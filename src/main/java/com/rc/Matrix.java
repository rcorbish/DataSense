
package com.rc ;

import java.lang.reflect.Array;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.Random;

import javax.management.RuntimeErrorException;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

public class Matrix {
    int M ;     // rows
    int N ;     // cols
    String name ;
    String labels[] ;

    static Compute Engine = Compute.getInstance() ;

    double data[] ;

    private boolean isVector ;

    public Matrix( int rows, int columns ) {
        this( rows, columns, new double[ rows * columns ] ) ;
    }

    public Matrix( int rows, int columns, double ... data ) {
        this.M = rows ;
        this.N = columns ;
        this.data = data ;
        isVector = rows==1 || columns==1 ;
    }

    public Matrix dup() {
        double copy[] = new double[data.length] ;
        System.arraycopy( data, 0, copy, 0, copy.length ) ;
        Matrix rc = new Matrix( M, N, copy ) ;
        if( name != null ) {
        	rc.name = name + " copy" ;
        }
        if( labels != null ) {
        	rc.labels = labels.clone() ;
        }
        return rc ;
    }

    public Matrix sub( Matrix O ) {
    	for( int i=0 ; i<length() ; i++ ) {
    		data[i] -= O.data[i] ;
    	}    		
        return this ;
    }

    public Matrix mmul( Matrix B ) {
        return Engine.mmul(this, B ) ;
    }

    //
    // X = A\B	( matlab/octave convention )
    //
    // Commonly used to solve linear equations
    //
    //
    public Matrix divLeft( Matrix B ) {
        return Engine.solve(this, B ) ;
    }
    
    //
    // X = B/A	( matlab/octave convention )
    //
    // Less commonly used 
    //
    //
    public Matrix divRight( Matrix B ) {
        return Engine.solve2(this, B ) ;
    }

    public Matrix transpose() {

		double data2[] = new double[ data.length] ;
    	
        for( int i=0 ; i<M ; i++ ) {
        	for( int j=0 ; j<N ; j++ ) {
        		int i1 = i + j*M ;
        		int i2 = j + i*N ;
        		
                data2[ i2 ] = data[ i1 ];
        	}
        }
        int tmp = this.M ;
        this.M = this.N ;
        this.N = tmp ;      
        this.data = data2 ;
        
        return this ;
    }

	public void reshape( int newM, int newN ) {

		if( newM != M || newN != N ) {
			double newData[] = new double[newM * newN] ;
			
			for( int i=0 ; i<newM ; i++ ) {
				for( int j=0 ; j<newN ; j++ ) {
					newData[ i + j*newM  ] = (j<N && i<M ) ? data[ i + j*M ] : 0 ; 
				}
			}
			data = newData ;
			M = newM ;
			N = newN ;
		}
	}
	
	public Matrix extractColumns( int ... cols ) {
		
		Matrix rc = new Matrix( M, cols.length ) ;
		
		for( int i=0 ; i<cols.length ; ++i ) {
			System.arraycopy( data, M*cols[i], rc.data, M*i, M ) ;
		}
		
		for( int i=0 ; i<cols.length ; ++i ) {
			int col = cols[i] ;
			int len = ( N-1 - col ) * M ;
			System.arraycopy( data, M*(col+1), data, M*col, len ) ;
			System.arraycopy( labels, (col+1), labels, col, (labels.length - 1 - col) ) ;
			N-- ;
		}
		
		return rc ;
	}

	public Matrix appendColumns( Matrix other ) {
		
		if( M != other.M ) {
			throw new RuntimeException( "Incompatible row counts for appendColumns" ) ;
		}
		
		Matrix rc = new Matrix( M, N+other.N ) ;
		
		System.arraycopy( data, 0, rc.data, 0, length() ) ;		
		System.arraycopy( other.data, 0, rc.data, M*N, other.length() ) ;

		rc.labels = new String[ rc.N ] ;
		System.arraycopy( labels, 0, rc.labels, 0, N ) ;		
		System.arraycopy( other.labels, 0, rc.labels, N, other.N ) ;
		
		return rc ;
	}

	public Matrix map( MatrixFunction func ) {
		for( int i=0 ; i<M ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				put( i, j, func.call( get(i,j), this, i, j ) ) ;
			}
		}
		return this ;
	}

	public Matrix mapColumn( MatrixColFunction func ) {
		
		for( int i=0 ; i<N ; i++ ) {
			putColumn(i, func.call( data, this, i*M, M ) ) ;
		}
		return this ;
	}
	
    public double get( int r, int c ) {
        return data[r + c*M]  ;
    }
    public void put( int r, int c, double v ) {
        data[r + c*M] = v ;  ;
    }
    public void putColumn( int c, double v[] ) {
        System.arraycopy(v, 0, data, c*M, M ) ;  ;
    }

    public int length() { return M*N ; }
    
    
    static public Matrix eye( int s ) {
    	Matrix rc = new Matrix( s, s ) ;
    	for( int i=0 ; i<rc.length() ; i+=(s+1) ) {
    		rc.data[i] = 1  ;
    	}
    	return rc ;
    }
	
	static public Matrix rand( int m,int n ) {
		Random rng = new Random() ;
    	Matrix rc = new Matrix( m, n ) ;
    	for( int i=0 ; i<rc.length() ; i++ ) {
    		rc.data[i] = rng.nextGaussian() ;
    	}
    	return rc ;
    }
    
    public String toString() {
        StringBuilder rc = new StringBuilder() ;
        if( name != null ) {
        	rc.append( name ) ;        	
        }

        if( labels != null ) {
	        for( int i=0 ; i<Math.min( 800, labels.length) ; i++ ) {
                rc.append( String.format("%10s", labels[i] ) );
	        }
            rc.append( '\n' ); 
        }
        for( int i=0 ; i<Math.min( 800, M) ; i++ ) {
            for( int j=0 ; j<Math.min( 800, N) ; j++ ) {
                rc.append( String.format( "%10.3f", get(i,j) ) );
            }
            rc.append( '\n' ); 
        }
        return rc.toString() ;
    }
    
    @FunctionalInterface
    static interface MatrixFunction {    	
    	public double call( double value, Matrix context, int r, int c ) ;
    }
    @FunctionalInterface
    static interface MatrixColFunction {    	
    	public double[] call( double values[], Matrix context, int offset, int len ) ;
    }
    
    static class Deserializer implements JsonSerializer<Matrix> {
    	@Override
    	public JsonElement serialize( Matrix src, Type typeOfSrc, JsonSerializationContext context) {
    		// This method gets involved whenever the parser encounters the Matrix
    		// object (for which this serializer is registered)
    		JsonObject object = new JsonObject();

    		object.addProperty("M", src.M);
    		object.addProperty("N", src.N);

    		JsonArray cols[] = new JsonArray[src.N] ;
    		
			JsonArray data = new JsonArray( src.M );
    		for( int i=0 ; i<src.N ; i++ ) {
    			JsonArray r = new JsonArray( src.N );
    			if( src.labels != null ) {
    				JsonObject row = new JsonObject() ;
    				row.add( src.labels[i], r );
    				data.add( row ) ;
    			} else {
    				data.add( r ) ; 
    			}
    			cols[i] = r ;
    		}
    		
    		for( int i=0 ; i<src.M ; i++ ) {
        		for( int j=0 ; j<src.N ; j++ ) {
        			cols[j].add( src.get( i, j ) ) ;
        		}
    		}
    		object.add( "data", data ) ;
    		return object;
    	}
    }
}