package com.rc;

import java.util.Random;



public class Main {

	public static void main(String[] args) {

		try {

			Random rng = new Random( 120 ) ;

			int rows  = 3 ;		// M
			int cols  = 5 ;		// N
			int mid   = 4 ;		// K
			int numFeatures = 4 ;

			Matrix A = new Matrix( rows, mid ) ; 	// M x K   
			Matrix B = new Matrix( mid, cols ) ;  	// K x N
			Matrix b = new Matrix( numFeatures,mid ) ; 	 	// M

			for( int i=0 ; i<A.length() ; i++ ) {
				A.data[i] = rng.nextGaussian() ;
			}
			for( int i=0 ; i<B.length() ; i++ ) {
				B.data[i] = rng.nextGaussian() ;
			}
			for( int i=0 ; i<b.length() ; i++ ) {
				b.data[i] = rng.nextGaussian() ;
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
			System.out.println( A );

			System.out.println( "--------- B --------" );
			System.out.println( B );

			System.out.println( "--------- b --------" );
			System.out.println( b );

			try ( Compute comp = Compute.getInstance() ) {
				Matrix A2 = A.dup();
				Matrix x = comp.solve2( A, b)  ;
				System.out.println( x );
				
				Matrix C = x.mmul( A2 ) ;
				System.out.println( C );

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
