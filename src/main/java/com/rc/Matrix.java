
package com.rc ;

public class Matrix {
    int M ;     // rows
    int N ;     // cols


    static Compute Engine = Compute.getInstance() ;

    double data[] ;

    private boolean isVector ;

    public Matrix( int rows, int columns ) {
        this( rows, columns, new double[ rows * columns ] ) ;
    }

    public Matrix( int rows, int columns, double data[] ) {
        this.M = rows ;
        this.N = columns ;
        this.data = data ;
        isVector = rows==1 || columns==1 ;
    }

    public Matrix dup() {
        double copy[] = new double[data.length] ;
        System.arraycopy( data, 0, copy, 0, copy.length ) ;
        return new Matrix( M, N, copy ) ;
    }

    public Matrix mmul( Matrix B ) {
        return Engine.mmul(this, B ) ;
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



    public double get( int r, int c ) {
        return data[r + c*M]  ;
    }

    public int length() { return M*N ; }
    public String toString() {
        StringBuilder rc = new StringBuilder() ;

        for( int i=0 ; i<Math.min( 8, M) ; i++ ) {
            for( int j=0 ; j<Math.min( 8, N) ; j++ ) {
                rc.append( String.format( "%10.3f", get(i,j) ) );
            }
            rc.append( '\n' ); 
        }
        return rc.toString() ;
    }
}