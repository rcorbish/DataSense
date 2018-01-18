
package com.rc ;

import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

public class Matrix {
	final static Logger log = LoggerFactory.getLogger( Matrix.class ) ;
	
	/**
	 * Number of rows (e.g. observations in a dataset )
	 */
	int M ;
	
	/**
	 * Number of features in each observation
	 */
	int N ;     
	/**
	 * Friendly name - used in printing
	 */
	String name ;
	/**
	 * Column labels - e.g. name of each feature being worked on
	 */
	String labels[] ;

	/**
	 * We use a math engine for big matrix operations
	 */
	static Compute Engine = Compute.getInstance() ;

	/**
	 * Underlying data holding the values
	 */
	double data[] ;

	/**
	 * Is the Matrix a vector? (rows or columns = 1 ? )
	 */
	public boolean isVector ;

	/**
	 * Is the Matrix a triangular matrix?
	 */
	public boolean isTriangular ;
	

	/**
	 * Create an matrix with the given size and column labels
	 * Matrices are stored column order ( fortran order )
	 * 
	 * @param rows number of rows
	 * @param columns number of columns
	 * @param labels the column labels to assign
	 */
	public Matrix( int rows, int columns, String ... labels ) {
		this( rows, columns, new double[ rows * columns ] ) ;
		if( labels != null ) {
			this.labels = labels.clone() ;
		}
	}

	/**
	 * Create an matrix with the given size and data store
	 * The data store should be at least as big as rows x columns
	 * Matrices are stored column order ( fortran order )
	 * 
	 * @param rows number of rows
	 * @param columns number of columns
	 * @param data the data array underlying this Matrix
	 */
	public Matrix( int rows, int columns, double ... data ) {
		this.M = rows ;
		this.N = columns ;
		this.data = data ;
		isVector = rows==1 || columns==1 ;
	}

	/**
	 * Create an empty matrix with the given size
	 * Matrices are stored column order ( fortran order )
	 * 
	 * @param rows number of rows
	 * @param columns number of columns
	 */
	public Matrix( int rows, int columns ) {
		this( rows, columns, new double[ rows * columns ] ) ;
	}


	/**
	 * Create an empty column vector with the given size
	 * Matrices are stored column order ( fortran order )
	 * 
	 * @param rows number of ritems
	 */
	public Matrix( int rows ) {
		this( rows, 1 ) ;
	}

	/**
	 * Create a copy of the matrix
	 * 
	 * @return a new Matrix
	 */
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


	/**
	 * Copy the upper triangular matrix from the receiver to a new 
	 * Matrix. The original is unchanged.
	 * 
	 * @return A new upper triangular matrix
	 */
	public Matrix upperTriangle() {
		Matrix rc = dup() ;
		for( int i=0 ; i<Math.min(M,N) ; i++ ) {
			int ix = M*i + i + 1 ;
			Arrays.fill( rc.data, ix, ix+M-i-1, 0 ) ;
		}
		log.info( "RC = {}", rc.data ) ;
		rc.isTriangular = true ;
		return rc ;
	}
	

	/**
	 * Elementwise subtraction of each element in a matrix.
	 * Each element has the corresponding element from O subtracted
	 *  
	 * @param O the other matrix
	 * @return a new Matrix
	 */
	public Matrix sub( Matrix O ) {
		Matrix rc = dup() ;
		return rc.subi( O ) ;
	}


	/**
	 * Elementwise subtraction of each element in a matrix. The original 
	 * matrix is changed. Each element has the corresponding element from O subtracted
	 * @param O the other matrix
	 * @return this
	 */
	public Matrix subi( Matrix O ) {
		if( O.isVector && isVector ) { // vector add
			if( O.length() != length() ) {
				throw new RuntimeException( "Invalid subtraction of vectors: [" + O.length() + "] - [" + length() + "]") ;
			}
			for( int i=0 ; i<length() ; i++ ) {
				data[i] -= O.data[i] ;
			}
		} else if( isVector && O.M>1 ) { // can't add a matrix to a vector
			throw new RuntimeException( "Can't subtract a matrix from a vector" ) ;
		} else if( O.N==1 ) { // col vector vs matrix 
			if( O.M != M ) {
				throw new RuntimeException( "Invalid subtraction of column vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] -= O.data[i] ;
				}
			}
		} else if( O.M==1 ) { // col vector vs matrix 
			if( O.N != N ) {
				throw new RuntimeException( "Invalid addition of row vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] -= O.data[j] ;
				}
			}
		} else { // compatible matrix
			for( int i=0 ; i<length() ; i++ ) {
				data[i] -= O.data[i] ;
			}
		}
		return this ;
	}


	/**
	 * Elementwise addition of each element in a matrix. 
	 * @param O the other matrix
	 * @return a new Matrix
	 */
	public Matrix add( Matrix O ) {
		Matrix rc = dup() ;
		return rc.addi( O ) ;
	}


	/**
	 * Elementwise addition of each element in a matrix. The original 
	 * matrix is changed
	 * @param O the other matrix
	 * @return this
	 */
	public Matrix addi( Matrix O ) {
		if( O.isVector && isVector ) { // vector add
			if( O.length() != length() ) {
				throw new RuntimeException( "Invalid addition of vectors" ) ;
			}
			for( int i=0 ; i<length() ; i++ ) {
				data[i] += O.data[i] ;
			}
		} else if( isVector && O.M>1 ) { // can't add a matrix to a vector
			throw new RuntimeException( "Can't add a matrix to a vector" ) ;
		} else if( O.N==1 ) { // col vector vs matrix 
			if( O.M != M ) {
				throw new RuntimeException( "Invalid addition of column vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] += O.data[i] ;
				}
			}
		} else if( O.M==1 ) { // col vector vs matrix 
			if( O.N != N ) {
				throw new RuntimeException( "Invalid addition of row vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] += O.data[j] ;
				}
			}
		} else { // compatible matrix
			for( int i=0 ; i<length() ; i++ ) {
				data[i] += O.data[i] ;
			}
		}
		return this ;
	}


	/**
	 * Scalar addition of value to each element in a matrix. 
	 * @param x the scalar to add to every element
	 * @return a new Matrix
	 */
	public Matrix add( double x ) {
		Matrix rc = dup() ;
		return rc.addi(x) ;
	}


	/**
	 * Scalar addition of value to each element in a matrix. The original 
	 * matrix is changed
	 * @param x the scalar to add to every element
	 * @return this
	 */
	public Matrix addi( double x ) {
		for( int i=0 ; i<length() ; i++ ) {
			data[i] += x ;
		}    		
		return this ;
	}

	/**
	 * Scalar multiplication of each element in a matrix. The original 
	 * matrix is changed
	 * @param a the factor
	 * @return this
	 */
	public Matrix muli( double a ) {
		for( int i=0 ; i<length() ; i++ ) {
			data[i] *= a ;
		}    		
		return this ;
	}

	/**
	 * Scalar multiplication of each element in a matrix. 
	 * @param a the factor
	 * @return a new Matrix
	 */
	public Matrix mul( double a ) {
		Matrix rc = dup() ;
		return rc.muli( a )  ;
	}


	/**
	 * Scalar division of each element in a matrix. The original 
	 * matrix is changed
	 * @param a the factor
	 * @return this
	 */
	public Matrix divi( double a ) {
		for( int i=0 ; i<length() ; i++ ) {
			data[i] /= a ;
		}    		
		return this ;
	}

	/**
	 * Scalar division of each element in a matrix. 

	 * @param a the factor
	 * @return a new Matrix
	 */
	public Matrix div( double a ) {
		Matrix rc = dup() ;
		return rc.divi( a )  ;
	}

	/**
	 * Hadamard multiply - elementwise multiply of 2 matrices
	 * 
	 * @param O other matrix
	 * @return a new Matrix
	 */
	public Matrix hmul( Matrix O) {
		Matrix rc = dup() ;
		return rc.hmuli( O )  ;
	}

	/**
	 * Hadamard multiply - elementwise multiply of 2 matrices
	 * The matrix is updated with the product. If O is smaller
	 * than the receiver, the elements to multiply by starts
	 * from the beginning of O again. This allows us to 
	 * multiply a rectangular receiver by a column vector
	 * 
	 * @param O other matrix
	 * @return this
	 */
	public Matrix hmuli( Matrix O ) {
		if( O.M == 1 && M>1 ) { // row vector vs matrix
			if( O.N != N ) {
				throw new RuntimeException( "Invalid mpy of columns vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] *= O.data[j] ;
				}
			}
		} else if( O.N==1 && N>1 ) { // col vector  vs matrix 
			if( O.M != M ) {
				throw new RuntimeException( "Invalid m pyof columns vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] *= O.data[i] ;
				}
			}
		} else { // compatible matrix
			for( int i=0 ; i<length() ; i++ ) {
				data[i] *= O.data[i] ;
			}
		}
		return this ;
	}

	
	/**
	 * Hadamard divide - elementwise divide of 2 matrices
	 * 
	 * @param O other matrix
	 * @return a new Matrix
	 */
	public Matrix hdiv( Matrix O) {
		Matrix rc = dup() ;
		return rc.hdivi( O )  ;
	}

	/**
	 * Hadamard multiply - elementwise divide of 2 matrices
	 * The matrix is updated with the product. If O is smaller
	 * than the receiver, the elements to divide by starts
	 * from the beginning of O again. This allows us to 
	 * divide a rectangular receiver by a column vector
	 * 
	 * @param O other matrix
	 * @return this
	 */
	public Matrix hdivi( Matrix O ) {
		if( O.M == 1  ) { // row vector vs matrix
			if( O.N != N && M>1 ) {
				throw new RuntimeException( "Invalid div of columns vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] /= O.data[j] ;
				}
			}
		} else if( O.N==1 && N>1 ) { // col vector  vs matrix 
			if( O.M != M ) {
				throw new RuntimeException( "Invalid div of columns vector" ) ;
			}
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					data[ i + j*M  ] /= O.data[i] ;
				}
			}
		} else { // compatible matrix
			for( int i=0 ; i<length() ; i++ ) {
				data[i] /= O.data[i] ;
			}
		}
		return this ;
	}

	
	/**
	 * Total all values in the matrix
	 * 
	 * @return the total of all values
	 */
	public double total() {
		double rc = 0 ;
		for( int i=0 ; i<length() ; i++ ) {
			rc += data[i] ;
		}    		
		return rc ;
	}

	/**
	 * Matrix multiply
	 * 
	 * @param B the other matrix
	 * @return a new Matrix
	 */
	public Matrix mmul( Matrix B ) {
		return Engine.mmul(this, B ) ;
	}

	/**
	 * Find the dot product of 2 vectors
	 * 
	 * @param B the other vector
	 * @return the dot product
	 */
	public double dot( Matrix B ) {
		return Engine.dot(this, B ) ;
	}

	/**
	 * Find the dot product of a vector with itself
	 * @return the dot product
	 */
	public double dot() {
		double x = norm() ;
		return x * x ;
	}



	/**
	 * Find the kroenecker product of a matrix with another
	 * @return a new matrix
	 */
	public Matrix outer( Matrix O ) {
		Matrix rc = null ;
		if( O.isVector && isVector ) {
			rc = new Matrix( length(), O.length() ) ;
			for( int i=0 ; i<length() ; i++ ) {
				for( int j=0 ; j<O.length() ; j++ ) {
					rc.put( i, j, get(i) * O.get(j) ) ;
				}
			}
		}
		return rc ;
	}

	
	/**
	 * Return the length of a vector
	 * @return the Euclidean norm
	 */
	public double norm() {
		return Engine.norm(this) ;
	}

	
	/**
	* X = A\B	( matlab/octave convention )
	*
	* Commonly used to solve linear equations
	*
	*/
	public Matrix divLeft( Matrix B ) {
		return Engine.solve(this, B ) ;
	}

	/**
	* X = B/A	( matlab/octave convention )
	*
	* Less commonly used 
	*
	*/
	public Matrix divRight( Matrix B ) {
		return Engine.solve2(this, B ) ;
	}

	/**
	 * Find the determinant of a matrix. This uses
	 * LU decomp to find a triangular factor from which
	 * we can use the trace to get the determinant value
	 */
	public double det() {
		if( M != N ) {
			throw new IllegalArgumentException( "Determinant of non-square matrix requested" ) ;
		}
		double rc = 1 ;
		if( isTriangular ) {
			for( int i=0 ; i<Math.min(M,N) ; i++ ) {
				rc *= get(i,i) ;
			}
		} else {
			int ipiv[] = new int[ Math.min( M, N ) ] ;
			Matrix lu = Engine.lud( dup(), ipiv ) ;
			for( int i=0 ; i<Math.min(M,N) ; i++ ) {
				rc *= lu.get(i,i) ;
			}
			for( int i=0 ; i<ipiv.length ; i++ ) {
				rc *= ipiv[i] == (i+1) ? 1 : -1 ;
			}
		}
		return rc ;
	}
	
/**
 * Transpose a matrix
 * 
 * @return  a new Matrix
 */
	public Matrix transpose() {

		Matrix rc ;
		if( isVector ) {
			rc = new Matrix( this.N, this.M, data.clone() ) ;	
		} else {
			double data2[] = new double[ data.length] ;
			for( int i=0 ; i<M ; i++ ) {
				for( int j=0 ; j<N ; j++ ) {
					int i1 = i + j*M ;
					int i2 = j + i*N ;

					data2[ i2 ] = data[ i1 ];
				}
			}
			rc = new Matrix( this.N, this.M, data2 ) ;
		}
		return rc ;
	}

	/**
	 * Return a Matrix of the count of distinct values in each column. 
	 * A value is considered distinct to the precision given.
	 * This probably only has real meaning for integers (e.g mapped columns)
	 * 
	 * @param precision how to determine whether a value is similar to another value 
	 * @return a vector of each column's minimum
	 */
	public Matrix countBuckets( double precision ) {
		Matrix rc = new Matrix( 1, N ) ;

		Set<String> buckets = new HashSet<>() ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			buckets.clear() ;
			for( int j=0 ; j<M ; j++ ) {
				buckets.add( String.valueOf( (int)( data[ix] / precision ) ) ) ;
				ix++ ;
			}
			rc.data[i] = buckets.size() ;
		}		
		return rc ;
	}

	/**
	 * Return a Matrix of the mean value in each column
	 * 
	 * @return a vector of each column's geometric mean
	 */
	public Matrix mean() {
		Matrix rc = new Matrix( 1, N ) ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			double sum = 0 ;
			for( int j=0 ; j<M ; j++ ) {
				sum += data[ix] ;
				ix++ ;
			}
			double mean = sum / M ;
			rc.data[i] = mean ;
		}		
		return rc ;
	}

	/**
	 * Return a Matrix of the mean value in each column
	 * 
	 * @return a vector of each column's geometric mean
	 */
	public Matrix sum() {
		Matrix rc = new Matrix( 1, N ) ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			double sum = 0 ;
			for( int j=0 ; j<M ; j++ ) {
				sum += data[ix] ;
				ix++ ;
			}
			rc.data[i] = sum ;
		}		
		return rc ;
	}

	
	
	/**
	 * Return a Matrix of the median value in each column
	 * 
	 * @return a vector of each column's median
	 */
	public Matrix median() {
		Matrix rc = new Matrix( 1, N ) ;

		double col[] = new double[ M ] ; 
	
		for( int i=0 ; i<N ; i++ ) {
			System.arraycopy( data, i*M, col, 0, M ) ;
			Arrays.sort( col ) ;
			rc.data[i] = col[ col.length/2 ] ;
		}		
		return rc ;
	}

	
	
	/**
	 * Return a Matrix of the minimum value in each column
	 * 
	 * @return a vector of each column's minimum
	 */
	public Matrix min() {
		Matrix rc = new Matrix( 1, N ) ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			double min = data[ix++] ;
			for( int j=1 ; j<M ; j++ ) {
				min = Math.min( min, data[ix++] ) ;
			}
			rc.data[i] = min ;
		}		
		return rc ;
	}

	/**
	 * Return a Matrix of the maximum value in each column
	 * 
	 * @return a vector of each column's maximum
	 */
	public Matrix max() {
		Matrix rc = new Matrix( 1, N ) ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			double min = data[ix++] ;
			for( int j=1 ; j<M ; j++ ) {
				min = Math.max( min, data[ix++] ) ;
			}
			rc.data[i] = min ;
		}		
		return rc ;
	}


	/**
	 * Return a Matrix of the standard deviation of each column
	 * 
	 * @param means the column means
	 * @return a vector of each column's standard deviation
	 */
	public Matrix stddev( Matrix means ) {
		Matrix rc = variance( means ) ;

		for( int i=0 ; i<rc.length() ; i++ ) {
			rc.data[i] = Math.sqrt( rc.data[i] ) ;
		}

		return rc ;
	}

	/**
	 * Return a Matrix of the variance of each column
	 * 
	 * @param means the column means
	 * @return a vector of each column's variance
	 */
	public Matrix variance( Matrix means ) {
		Matrix rc = new Matrix( 1, N ) ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			double sum = 0 ;
			double mean = means.data[i] ;
			for( int j=0 ; j<M ; j++ ) {
				double d  = ( data[ix] - mean ) ;
				sum += d * d ;
				ix++ ;
			}
			rc.data[i] = sum / ( M-1 ) ;
		}
		return rc ;
	}

	/**
	 * Return a Matrix of the skewness of each column
	 * 
	 * @param means the column means
	 * @return a vector of each column's skewness
	 */
	public Matrix skewness( Matrix means ) {
		Matrix rc = new Matrix( 1, N ) ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			double sumN = 0 ;
			double sumD = 0 ;
			double mean = means.data[i] ;
			for( int j=0 ; j<M ; j++ ) {
				double d  = ( data[ix] - mean ) ;
				sumN += d * d * d ;
				sumD += d * d ;
				ix++ ;
			}
			double n = sumN / M ;
			double d = Math.pow( ( sumD / M ), 1.5 ) ;
			rc.data[i] = n/d ;
		}
		return rc ;
	}


/**
 * Return a Matrix of the kurtosis of each column
 * 
 * @param means the column means
 * @return a vector of each column's kurtosis
 */
	public Matrix kurtosis( Matrix means ) {
		Matrix rc = new Matrix( 1, N ) ;

		int ix = 0 ;
		for( int i=0 ; i<N ; i++ ) {
			double sumN = 0 ;
			double sumD = 0 ;
			double mean = means.data[i] ;
			for( int j=0 ; j<M ; j++ ) {
				double d  = data[ix] - mean ;
				d = d * d ;
				sumN += d * d ;
				sumD += d ;
				ix++ ;
			}
			double n = sumN / M ;
			double d = sumD / M  ;
			rc.data[i] = ( n / (d*d) ) - 3.0 ;
		}		
		return rc ;
	}

/**
 * Subtract the column mean from each value in the matrix
 * @return a new Matrix
 */
	public Matrix zeroMeanColumns() {
		Matrix rc = new Matrix( M, N, labels ) ;

		for( int i=0 ; i<N ; i++ ) {
			// Mean
			double sum = 0 ;
			int ix = i*M ;
			for( int j=0 ; j<M ; j++ ) {
				sum += data[ix] ;
				ix++ ;
			}
			double mean = sum / M ;
			ix = i*M ;
			for( int j=0 ; j<M ; j++ ) {
				rc.data[ix] = data[ix] - mean ;
				ix++ ;
			}
		}
		return rc ;
	}

/**
 * For each column set all values to have zero mean & unit variance.
 * This can be useful in some algos.
 * 
 * @return a new Matrix of mormalized cols
 */
	public Matrix normalizeColumns() {
		Matrix rc = new Matrix( M, N, labels ) ;

		for( int i=0 ; i<N ; i++ ) {
			// Mean
			double sum = 0 ;
			int ix = i*M ;
			for( int j=0 ; j<M ; j++ ) {
				sum += data[ix] ;
				ix++ ;
			}
			double mean = sum / M ;

			// Std deviation
			ix = i*M ;
			sum = 0 ; 
			for( int j=0 ; j<M ; j++ ) {
				double d = data[ix] - mean ;
				sum += d * d ;
				ix++ ;
			}
			double sig = Math.sqrt( sum / (M - 1) ) ;

			// ( x - m ) / s    zero mean, unit variance
			ix = i*M ;
			for( int j=0 ; j<M ; j++ ) {
				rc.data[ix] = ( data[ix] - mean ) / sig ;
				ix++ ;
			}
		}
		return rc ;
	}
	
	
	/**
	 * Create a new matrix, of the same shape, with values set to 1.0 where
	 * the original matrix had a value equal to the given test, within a given 
	 * precision
	 * 
	 * @param test corresponding elements are set to one where their values equal this
	 * @param precision allows real number comparison within this precision
	 * @return a new Matrix
	 */
	public Matrix oneIfEquals( double test, double precision ) {
		Matrix rc = dup() ;
		for( int i=0 ; i<rc.length() ; i++ ) {
			double d = rc.data[i] - test ;
			rc.data[i] = d>precision || d<-precision ? 0.0 : 1.0 ;
		}
		return rc  ;
	}
	
	
	/**
	 * Return a column vector containing the index of the max value in each
	 * row. Used to determine logisitic regresion meaning.
	 * 
	 * @return a new Matrix
	 */
	public Matrix maxIndexOfRows() {
		Matrix rc = new Matrix( M, 1 ) ;
		for( int i=0 ; i<M ; i++ ) {
			double max = get( i, 0 ) ;
			int ix = 0 ;
		
			for( int j=1 ; j<N ; j++ ) {
				if( get(i,j) > max ) {
					ix = j ;
					max = get(i,j) ;
				}
			}
			rc.put( i,  ix ) ;
		}
		return rc  ;
	}

	/**
	 * Return a column vector containing the index of the max value in each
	 * row. Used to determine logisitic regresion meaning.
	 * 
	 * @return a new Matrix
	 */
	public Matrix minIndexOfRows() {
		Matrix rc = new Matrix( M, 1 ) ;
		for( int i=0 ; i<M ; i++ ) {
			double min = get( i, 0 ) ;
			int ix = 0 ;
		
			for( int j=1 ; j<N ; j++ ) {
				if( get(i,j) < min ) {
					ix = j ;
					min = get(i,j) ;
				}
			}
			rc.put( i,  ix ) ;
		}
		return rc  ;
	}

	
	
	/**
	 * Change the shape of a matrix. As best as possible the 
	 * original data is left intact. So if rows & columns are  less
	 * than the original - the upper left corner of the matrix is preserved.
	 * If the new values are larger zeros are inserted.
	 * 
	 * This affects the matrix in place. 
	 * 
	 * @param newM new rows
	 * @param newN new cols
	 */
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

	/**
	 * Remove some columns from a matrix, they should be copied to a new Matrix
	 * The original Matrix is changed by this operation
	 * 
	 * @param cols which columns to pull out
	 * @return a new Matrix containing the removed columns
	 */
	public Matrix extractColumns( int ... cols ) {

		Matrix rc = new Matrix( M, cols.length, new String[cols.length] ) ;

		for( int i=0 ; i<cols.length ; ++i ) {
			System.arraycopy( data, M*cols[i], rc.data, M*i, M ) ;
		}

		for( int i=0 ; i<cols.length ; ++i ) {
			int col = cols[i] ;
			int len = ( N-1 - col ) * M ;
			System.arraycopy( data, M*(col+1), data, M*col, len ) ;
			// handle label move
			if( labels != null ) {
				rc.labels[i] = labels[col] ;
				System.arraycopy( labels, (col+1), labels, col, (labels.length - 1 - col) ) ;
			}
			N-- ;
		}

		return rc ;
	}

	

	/**
	 * Copy some columns from a matrix, they should be copied to a new Matrix
	 * The original Matrix is unchanged by this operation
	 * 
	 * @param cols which columns to pull out
	 * @return a new Matrix containing the removed columns
	 */
	public Matrix copyColumns( int ... cols ) {

		Matrix rc = new Matrix( M, cols.length, new String[cols.length] ) ;
		rc.labels = labels==null ? null : new String[ cols.length ] ;

		for( int i=0 ; i<cols.length ; ++i ) {
			System.arraycopy( data, M*cols[i], rc.data, M*i, M ) ;
			if( labels != null ) {
				rc.labels[i] = labels[cols[i]] ;
			}
		}

		return rc ;
	}


	/**
	 * Swap two columns in a matrix
	 * The original Matrix is hanged by this operation
	 * 
	 * @param col1 swap this (0 based) column ...
	 * @param col2 ... with this (0 based) column
	 * @return the changed matrix
	 */
	public Matrix swapColumns( int col1, int col2 ) {
		if( col1 != col2 ) {
			double tmp[] = new double[M] ;
			System.arraycopy( data, M*col1, tmp, 0, M ) ;
			System.arraycopy( data, M*col2, data, M*col1, M ) ;
			System.arraycopy( tmp, 0, data, M*col2, M )  ;
			
			if( labels != null ) {
				String tmp2 = labels[col1] ;
				labels[col1] = labels[col2] ;
				labels[col2] = tmp2 ;
			}
		}
		return this ;
	}


	/**
	 * Add new columns to the'rhs' of the matrix.  The original matrix is unchanged.
	 * 
	 * @param other the new columns to add
	 * @return a new Matrix with the extra columns tacked on
	 */
	public Matrix appendColumns( Matrix other ) {

		if( M != other.M ) {
			throw new RuntimeException( "Incompatible row counts for appendColumns" ) ;
		}

		Matrix rc = new Matrix( M, N+other.N, new String[ N+other.N ] ) ;

		System.arraycopy( data, 0, rc.data, 0, length() ) ;		
		System.arraycopy( other.data, 0, rc.data, M*N, other.M * other.N ) ;

		if( labels != null && other.labels != null ) {
			System.arraycopy( labels, 0, rc.labels, 0, N ) ;		
			System.arraycopy( other.labels, 0, rc.labels, N, other.N ) ;
		}		
		return rc ;
	}

	/**
	 * Remove rows from the receiver. The rows are copied to a new Matrix. The
	 * original matrix is changed.
	 * 
	 * @param rows which rows to remove
	 * @return a new Matrix containing the removed rows
	 */
	public Matrix extractRows( int ... rows ) {

		Matrix rc = new Matrix( rows.length, N, labels ) ;

		for( int i=0 ; i<rows.length ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				rc.put( i,j, get(rows[i],j) ) ;
			}
		}
		for( int i=0 ; i<rows.length ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				int startCopy = j*M + rows[i] ;
				System.arraycopy( data, startCopy+1, data, startCopy, M-rows[i]-1 ) ;
			}
		}
		reshape( M-rc.M, N ) ;

		return rc ;
	}


	/**
	 * Copy rows from the receiver. The rows are copied to a new Matrix. The
	 * original matrix is unchanged.
	 * 
	 * @param rows which rows to copied
	 * @return a new Matrix containing the copied rows
	 */
	public Matrix copyRows( int ... rows ) {

		Matrix rc = new Matrix( rows.length, N, labels ) ;

		for( int i=0 ; i<rows.length ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				rc.put( i,j, get( rows[i], j ) ) ;
			}
		}

		return rc ;
	}
		

	/**
	 * Add a Matrix to the 'bottom' of the receiver. The number of
	 * columns of both matrices should match.
	 * 
	 * @param other add this to the receiver
	 * @return a new Matrix
	 */
	public Matrix appendRows( Matrix other ) {

		if( N != other.N ) {
			throw new RuntimeException( "Incompatible column counts for appendRows" ) ;
		}

		Matrix rc = new Matrix( M+other.M, N, labels ) ;

		for( int i=0 ; i<M ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				rc.put( i,j, get(i,j) ) ;
			}
		}
		for( int i=0 ; i<other.M ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				rc.put( i+M,j, other.get(i,j) ) ;
			}
		}

		return rc ;
	}

	/**
	 * Call a function on each element of a Matrix. This is the 
	 * full version of the {@link SimpleMatrixFunction} version. The Matrix 
	 * is changed in place - use dup() if you need a changed copy
	 * before calling a map() 
	 * 
	 * @param func the lambda to change a value
	 * @return this
	 */
	public Matrix map( MatrixFunction func ) {
		for( int i=0 ; i<M ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				put( i, j, func.call( get(i,j), this, i, j ) ) ;
			}
		}
		return this ;
	}

	/**
	 * Call a function on each element of a Matrix. This is the 
	 * short version of the {@link MatrixFunction} version. The Matrix 
	 * is changed in place - use dup() if you need a changed copy
	 * before calling a map() 
	 * 
	 * @param func the lambda to change a value
	 * @return this
	 */
	public Matrix map( DoubleUnaryOperator func ) {
		// Arrays.stream( data ).parallel().map( func ) ;
		for( int i=0 ; i<length() ; i++ ) {
			put( i, func.applyAsDouble( get(i) ) ) ;
		}
		return this ;
	}

	/**
	 * Call a function on each column of a Matrix. The Matrix 
	 * is changed in place - use dup() if you need a changed copy
	 * before calling a map()
	 * 
	 * @see MatrixColFunction
	 * 
	 * @param func the lambda to call on each column 
	 * @return this 
	 */
	public Matrix mapColumn( MatrixColFunction func ) {
		Matrix rc = new Matrix( M, N, labels ) ;

		for( int i=0 ; i<N ; i++ ) {
			rc.putColumn( i, func.call( data, this, i*M, M ) ) ;
		}
		return this ;
	}


	/**
	 * Call a function on each element of a Matrix. The Matrix 
	 * is not changed. The function takes the previous value 
	 * and the current value as arguments.  
	 * 
	 * @param func the lambda to call a value
	 * @return the result of the last function call for each column
	 */
	public Matrix reduce( DoubleBinaryOperator func ) {
		Matrix rc = new Matrix( N, 1 ) ;
		for( int j=0 ; j<N ; j++ ) {
			rc.put( j, get(0,j) ) ;
		}
		for( int i=1 ; i<M ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				rc.put( j, func.applyAsDouble( rc.get(j), get(i,j) ) ) ;
			}
		}
		return rc ;
	}

	/**
	 * Call a function on each element of a Matrix. The Matrix 
	 * is not changed. The function takes the previous value 
	 * and the current value as arguments.  
	 * 
	 * @param func the lambda to call a value
	 * @param start the initial value for all items
	 * @return the result of the last function call for each column
	 */
	public Matrix reduce( DoubleBinaryOperator func, double start ) {
		Matrix rc = new Matrix( N, 1 ) ;
		for( int j=0 ; j<N ; j++ ) {
			rc.put( j, start ) ;
		}
		for( int i=0 ; i<M ; i++ ) {
			for( int j=0 ; j<N ; j++ ) {
				rc.put( j, func.applyAsDouble( rc.get(j), get(i,j) ) ) ;
			}
		}
		return rc ;
	}


	/**
	 * Update labels with a prefix. Add the given prefix to each label 
	 * This is usually used when concatenating matrices.
	 * 
	 * @param prefix prefix to prepend to each label
	 */
	public void prefixLabels( String prefix ) {
		if( labels == null ) { 
			labels = new String[N] ;
			for( int i=0 ; i<N ; i++ ) {
				labels[i] = String.valueOf(i) ;
			}
		}
		for( int i=0 ; i<N ; i++ ) {
			labels[i] = prefix + labels[i] ;
		}
	}


	/**
	 * Add a value to an address in the array. Address is row,column
	 * No error checking is done ( args should be sensible to avoid exceptions )
	 * 
	 * @param r row ( 0 based )
	 * @param c column ( 0 based )
	 * @param v value to add to the address
	 * @return the value at the given address
	 */
	public void add( int r, int c, double v ) {
		data[ r + c*M ] += v ;
	}


	/**
	 * Get a value from an address in the array. Address is row,column
	 * No error checking is done ( args should be sensible to avoid exceptions )
	 * 
	 * @param r row ( 0 based )
	 * @param c column ( 0 based )
	 * @return the value at the given address
	 */
	public double get( int r, int c ) {
		return get( r + c*M ) ;
	}

	/**
	 * Get a value from the data
	 * @param ix index in the raw data ( 0 based )
	 * @return the value
	 */
	public double get( int ix ) {
		return data[ix]  ;
	}

	/**
	 * Set a value in the array at row,column address
	 * @param r row ( 0 based )
	 * @param c column ( 0 based )
	 * @param v value to set
	 */
	public void put( int r, int c, double v ) {
		put( r + c*M, v ) ;
	}
	/**
	 * Set a single value in the data
	 * @param ix index in the array ( 0 based )
	 * @param v value to set
	 */
	public void put( int ix, double v ) {
		data[ix] = v ;  ;
	}

	/**
	 * Overwrite a column in the array with new data. 
	 * 
	 * @param c column index to overwrite ( 0 based )
	 * @param v the new column data
	 */
	public void putColumn( int c, double v[] ) {
		if( v.length < M ) {
			throw new RuntimeException( "Column data to add is too short - need " + M + " elements." ) ; 
		}
		System.arraycopy(v, 0, data, c*M, M ) ;  ;
	}

	/**
	 * Is the matrix triangular?
	 * 
	 * @returns whether the matrix is triangular
	 */
	public boolean isTriangular() {
		isTriangular = true ;
		for( int i=1 ; i<M ; i++ ) {
			for( int j=Math.min(i,N)-1 ; j>0 ; --j ) {
				isTriangular &= get(i,j) == 0 ;
				if( !isTriangular ) return false ;
			}
		}
		return isTriangular ;
	}
	/**
	 * Total available size of the data
	 * Note - this may be larger than M*N (if we created a special matrix to hold more than rows x columns)
	 * Don't do the above unless you really need to, we don't support LDA !
	 *  
	 * @return number of values stored
	 */
	public int length() { return data.length ; }


	/**
	 * Create a square identity matrix
	 * 
	 * @param s rows & columns 
	 * @return a new matrix
	 */
	static public Matrix eye( int s ) {
		Matrix rc = new Matrix( s, s ) ;
		for( int i=0 ; i<rc.length() ; i+=(s+1) ) {
			rc.data[i] = 1  ;
		}
		return rc ;
	}

	/**
	 * Create a random matrix. Random is zero mean, unit variance.
	 * 
	 * @param m rows
	 * @param n columns
	 * @return a new Matrix
	 */
	static public Matrix rand( int m,int n ) {
		Random rng = new Random() ;
		Matrix rc = new Matrix( m, n ) ;
		for( int i=0 ; i<rc.length() ; i++ ) {
			rc.data[i] = rng.nextGaussian() ;
		}
		return rc ;
	}

	/**
	 * Create a new Matrix filled with a constant. Label the columns
	 * with given values
	 * 
	 * @param m rows
	 * @param n cols
	 * @param v value to fill
	 * @param labels names of the columns
	 * @return a new Matrix
	 */
	static public Matrix fill( int m,int n, double v, String ... labels ) {
		Matrix rc = fill( m, n, v ) ;
		rc.labels = labels ;
		return rc ;
	}

	/**
	 * Create a new Matrix filled with a single value
	 * @param m rows
	 * @param n columns 
	 * @param v value to fill the matrix
	 * @return a new Matrix
	 */
	static public Matrix fill( int m,int n, double v ) {
		Matrix rc = new Matrix( m, n ) ;
		Arrays.fill( rc.data, v);
		return rc ;
	}
	
	/**
	 * Create a new Vector filled with a single value
	 * @param m rows
	 * @param v value to fill the matrix
	 * @return a new Matrix
	 */
	static public Matrix fill( int m, double v ) {
		Matrix rc = new Matrix( m ) ;
		Arrays.fill( rc.data, v);
		return rc ;
	}

	/**
	 * Create a new Vector shuffled integer values of 0..N
	 * @param m rows
	 * @return a new Matrix
	 */
	static public Matrix shuffle( int m ) {
		Matrix rc = new Matrix( m ) ;
		int d[] = new int[m] ;
		for( int i=0 ; i<m ; i++ ) {
			d[i] = i ;
		}
		Random rng = new Random() ;
		
		for( int i=0 ; i<m ; i++ ) {
			int n = rng.nextInt( m - i ) ;
			rc.data[i] = d[n] ;
			d[n] = d[m-i-1] ;
		}
		return rc ;
	}

	/**
	 * Nice means to print a matrix. We only print some of the data
	 * otherwise this would go crazy. Change the MAX_PRINT value
	 * to adjust the max matrix values to print
	 * 
	 */
	public String toString() {
		final int MAX_PRINT = 10 ;
		StringBuilder rc = new StringBuilder() ;
		if( name != null ) {
			rc.append( name ) ;        	
		}

		if( labels != null ) {
			for( int i=0 ; i<Math.min( MAX_PRINT, labels.length) ; i++ ) {
				rc.append( String.format("%10s", labels[i] ) );
			}
			rc.append( '\n' ); 
		}
		if( isVector ) {
			for( int i=0 ; i<Math.min( MAX_PRINT, length() ) ; i++ ) {
				rc.append( String.format( "%10.4f", get(i) ) );
			}			
		} else {
			for( int i=0 ; i<Math.min( MAX_PRINT, M) ; i++ ) {
				for( int j=0 ; j<Math.min( MAX_PRINT, N) ; j++ ) {
					rc.append( String.format( "%10.4f", get(i,j) ) );
				}
				rc.append( '\n' ); 
			}
		}
		rc.deleteCharAt( rc.length()-1 ) ;
		return rc.toString() ;
	}

	//
	// Functional interfaces for lambda methods
	// 
	@FunctionalInterface
	static interface MatrixFunction {    	
		public double call( double value, Matrix context, int r, int c ) ;
	}
	// @FunctionalInterface
	// static interface SimpleMatrixFunction {    	
	// 	public double call( double value ) ;
	// }
	@FunctionalInterface
	static interface MatrixColFunction {    	
		public double [] call( double values[], Matrix context, int offset, int len ) ;
	}

	/**
	 * JSON deserializer - used by gson.
	 * If we send a Matrix to a client (esp. web page) this will print it our nicely
	 * 
	 * @author richard
	 *
	 */
	static class Deserializer implements JsonSerializer<Matrix> {
		@Override
		public JsonElement serialize( Matrix src, Type typeOfSrc, JsonSerializationContext context) {
			// This method gets involved whenever the parser encounters the Matrix
			// object (for which this serializer is registered)
			JsonObject object = new JsonObject();

			object.addProperty("M", src.M);
			if( src.name != null ) {
				object.addProperty("name", src.name);
			}

			if( src.isVector ) {
				object.addProperty("M", src.length() );
				if( src.labels == null ) {
					JsonArray data = new JsonArray( src.length() ) ;
					for( int i=0 ; i<src.length() ; i++ ) {
						data.add( src.data[i] ) ;
					}
					object.add( "data", data ) ;
				} else if( src.labels.length >= src.length() ) {
					for( int i=0 ; i<src.length() ; i++ ) {
						object.addProperty( src.labels[i] , src.data[i] ) ;
					}
				} else if( src.labels.length == 1 ) {
					JsonArray data = new JsonArray( src.length() ) ;
					for( int i=0 ; i<src.length() ; i++ ) {
						data.add( src.data[i] ) ;
					}
					object.add( src.labels[0], data ) ;
				}
			} else {
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
			}
			return object;
		}
	}
}