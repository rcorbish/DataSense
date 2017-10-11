package com.rc;

import java.nio.file.Paths;
import java.util.Random;



public class Main {

	public static void main(String[] args) {

		try {

			Random rng = new Random( 100 ) ;

			int rows  = 3 ;		// M
			int cols  = 4 ;		// N
			int mid   = 4 ;		// K
			int numFeatures = 4 ;

			double Ad[] = new double[rows*mid] ; 	// M x K   
			double Bd[] = new double[mid*cols] ;  	// K x N
			double bd[] = new double[rows*numFeatures] ; 	 	// M

			for( int i=0 ; i<rows*mid ; i++ ) {
				Ad[i] = rng.nextGaussian() ;
			}
			for( int i=0 ; i<mid*cols ; i++ ) {
				Bd[i] = rng.nextGaussian() ;
			}
			for( int i=0 ; i<rows ; i++ ) {
				//b[i] = 0 ;
				for( int j=0 ; j<mid ; j++ ) {
					int ix = i+ j*rows ;
					bd[i] += Ad[ix] * -(j+1) ; //+ rng.nextGaussian() / 100.0 ;
					bd[i+rows] += Ad[ix] * (j+3) ; //+ rng.nextGaussian() / 100.0 ;
					bd[i+rows+rows] += Ad[ix] * Math.E ;
					bd[i+rows+rows+rows] += Ad[ix] * Math.PI ;
				}
			}
			
			Matrix A = new Matrix(rows,  mid, Ad ) ;
			Matrix B = new Matrix( mid, cols, Bd ) ;
			Matrix b = new Matrix(rows,  numFeatures, bd ) ;

			System.out.println( A ) ;
			A.transpose();
			System.out.println( A ) ;
			A.transpose();
			
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
				Matrix x = comp.solve2( A.transpose(), b.transpose(), numFeatures)  ;
				System.out.println( x );
				
				double Cd[] = comp.mmul(rows, numFeatures, x.data, A2.data ) ;
				Matrix C = new Matrix( rows, numFeatures, Cd ) ;
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
