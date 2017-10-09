package com.rc;

import java.nio.file.Paths;
import java.util.Random;



public class Main {

	public static void main(String[] args) {

		try {
			
			Random rng = new Random( 100 ) ;

			int rows  = 19_000 ;		// M
			int cols  = 10 ;		// N
			int mid   = 100 ;		// K

			double A[] = new double[rows*mid] ; 	// M x K   
			double B[] = new double[mid*cols] ;  	// K x N
			double b[] = new double[rows] ; 	 	// M

			for( int i=0 ; i<rows*mid ; i++ ) {
				A[i] = rng.nextGaussian() ;
			}
			for( int i=0 ; i<mid*cols ; i++ ) {
				B[i] = rng.nextGaussian() ;
			}
			for( int i=0 ; i<rows ; i++ ) {
				//b[i] = 0 ;
				for( int j=0 ; j<cols ; j++ ) {
					int ix = i+ j*rows ;
					b[i] += A[ix] * -(j+1) + rng.nextGaussian() / 10.0 ;
				}
			}

			System.out.println( "------ S A V E -----" );

			Loader.saveToCsv( rows, A, Paths.get("A.csv") ) ;
			Loader.saveToCsv( mid , B, Paths.get("B.csv") ) ;
			Loader.saveToCsv( rows, b, Paths.get("b.csv") ) ;
			
			System.out.println( "------ L O A D -----" );
			rows = 500 ;
			A = Loader.loadFromCsv( rows, Paths.get("A.csv") ) ;
			B = Loader.loadFromCsv( mid, Paths.get("B.csv") ) ;
			b = Loader.loadFromCsv( rows, Paths.get("b.csv") ) ;
			for( int i=0 ; i<rows ; i++ ) {
				b[i] = 0 ;
				for( int j=0 ; j<cols ; j++ ) {
					int ix = i+ j*rows ;
					b[i] += A[ix] * -(j+1) + rng.nextGaussian() / 10.0 ;
				}
			}
			
					
			System.out.println( "--------- A --------" );
			printMatrix(rows, mid, A);

			//		System.out.println( "--------- B --------" );
			//		printMatrix(mid, cols, B);

			System.out.println( "--------- b --------" );
			printMatrix(1, rows, b);


			System.out.println( "\n---- C U D A ------" );

			try ( Cuda cuda = new Cuda() ) {
				double C[] = cuda.mmul(rows, cols, A, B) ;
				printMatrix( rows, cols, C ) ;

				double x[] = cuda.solve(rows, mid, A, b) ;
				printMatrix(1, cols, x);
			} catch( Throwable ignore ) {
				ignore.printStackTrace();
			}


			System.out.println( "----- B L A S ------" );
			try ( Blas blas = new Blas(8) ) {
				double C[] = blas.mmul(rows, cols, A, B) ;
				printMatrix( rows, cols, C ) ;

				double x[] = blas.solve(rows, mid, A, b) ;
				printMatrix(1, cols, x);
			} catch( Throwable ignore ) {
				ignore.printStackTrace();
			}

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
