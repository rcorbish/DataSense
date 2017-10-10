package com.rc;

import java.nio.file.Paths;
import java.util.Random;



public class Main {

	public static void main(String[] args) {

		try {

			Random rng = new Random( 100 ) ;

			int rows  = 4 ;		// M
			int cols  = 4 ;		// N
			int mid   = 4 ;		// K
			int numFeatures = 4 ;

			double A[] = new double[rows*mid] ; 	// M x K   
			double A2[] = new double[rows*mid] ; 	// M x K   
			double B[] = new double[mid*cols] ;  	// K x N
			double b[] = new double[rows*numFeatures] ; 	 	// M

			for( int i=0 ; i<rows*mid ; i++ ) {
				A[i] = rng.nextGaussian() ;
				A2[i] = A[i] ;
			}
			for( int i=0 ; i<mid*cols ; i++ ) {
				B[i] = rng.nextGaussian() ;
			}
			for( int i=0 ; i<rows ; i++ ) {
				//b[i] = 0 ;
				for( int j=0 ; j<mid ; j++ ) {
					int ix = i+ j*rows ;
					b[i] += A[ix] * -(j+1) ; //+ rng.nextGaussian() / 100.0 ;
					b[i+rows] += A[ix] * (j+3) ; //+ rng.nextGaussian() / 100.0 ;
					b[i+rows+rows] += A[ix] * Math.E ;
					b[i+rows+rows+rows] += A[ix] * Math.PI ;
				}
			}

			/*
			System.out.println( "------ S A V E -----" );

			Loader.saveToCsv( rows, A, Paths.get("A.csv") ) ;
			Loader.saveToCsv( mid , B, Paths.get("B.csv") ) ;
			Loader.saveToCsv( rows, b, Paths.get("b.csv") ) ;

			System.out.println( "------ L O A D -----" );
			rows = 3_000 ;
			A = Loader.loadFromCsv( rows, Paths.get("A.csv") ) ;
			B = Loader.loadFromCsv( mid, Paths.get("B.csv") ) ;
			b = Loader.loadFromCsv( rows, Paths.get("b.csv") ) ;
			for( int i=0 ; i<rows ; i++ ) {
				b[i] = 0 ;
				for( int j=0 ; j<cols ; j++ ) {
					int ix = i+ j*rows ;
					b[i] += A[ix] * -(j+1) + rng.nextGaussian() / 100.0 ;
				}
			}
			 */

			System.out.println( "--------- A --------" );
			printMatrix(rows, mid, A);

			System.out.println( "--------- B --------" );
			printMatrix(mid, cols, B);

			System.out.println( "--------- b --------" );
			printMatrix(rows, numFeatures, b);

			System.out.println( "----- A U T O ------" );
			try ( Compute comp = Compute.getInstance() ) {
				Matrix x = new Matrix( mid, numFeatures, comp.solve2(rows, mid, A, b, numFeatures) ) ;
				System.out.println(x);
				x.transpose(); 
				System.out.println(x);
				System.out.println(x.dup());
				
				double C[] = comp.mmul(rows, numFeatures, x.data, A2 ) ;
				printMatrix( rows, numFeatures, C ) ;

				// System.out.println();
				// printMatrix( rows, numFeatures, b ) ;
			} catch( Throwable ignore ) {
				ignore.printStackTrace();
			}
/*
			System.out.println( "\n---- C U D A ------" );

			try ( Compute comp =new Cuda() ) {
				double C[] = comp.mmul(rows, cols, A, B) ;
				printMatrix( rows, cols, C ) ;

				double x[] = comp.solve(rows, mid, A, b, numFeatures) ;
				printMatrix(mid, numFeatures, x);
			} catch( Throwable ignore ) {
				ignore.printStackTrace();
			}

			System.out.println( "----- B L A S ------" );
			try ( Compute comp = new Blas() ) {
				double C[] = comp.mmul(rows, cols, A, B) ;
				printMatrix( rows, cols, C ) ;

				double x[] = comp.solve(rows, mid, A, b, numFeatures) ;
				printMatrix(mid, numFeatures, x);
			} catch( Throwable ignore ) {
				ignore.printStackTrace();
			}
*/
		} catch( Throwable t ) {
			t.printStackTrace(); 
			System.exit( 2 ); ;
		}
	}

	static void printMatrix( int M, int N, double A[] ) {
		for( int i=0 ; i<Math.min( 8, M) ; i++ ) {
			for( int j=0 ; j<Math.min( 8, N) ; j++ ) {
				int ix = i + j*M ;
				System.out.print( String.format( "%10.2f", A[ix] ) );
			}
			System.out.println(); 
		}
	}
}
