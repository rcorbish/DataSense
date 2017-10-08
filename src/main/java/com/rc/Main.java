package com.rc;

import java.util.Random;



public class Main {

	public static void main(String[] args) {

		Random rng = new Random( 100 ) ;

		int rows  = 5_000 ;		// M
		int cols  = 1_000 ;			// N
		int mid   = 1_000 ;			// K

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

		System.out.println( "--------- A --------" );
		printMatrix(rows, mid, A);

//		System.out.println( "--------- B --------" );
//		printMatrix(mid, cols, B);

		System.out.println( "--------- b --------" );
		printMatrix(1, rows, b);


		System.out.println( "\n---- C U D A ------" );

		try ( Cuda cuda = new Cuda() ) {
			long start = System.nanoTime() ;
			double C[] = cuda.mmul(rows, cols, A, B) ;
			long delta = System.nanoTime() - start ;
			printMatrix( rows, cols, C ) ;
			System.out.println( "Elapsed " + String.format( "%,d uS", (delta/1000) )) ;
		} catch( Throwable ignore ) {

		}

		try ( Cuda cuda = new Cuda() ) {
			long start = System.nanoTime() ;
			double x[] = cuda.solve(rows, mid, A, b) ;
			long delta = System.nanoTime() - start ;
			printMatrix(1, cols, x);
			System.out.println( "Elapsed " + String.format( "%,d uS", (delta/1000) )) ;
		} catch( Throwable ignore ) {
			ignore.printStackTrace();
		}
		

		System.out.println( "----- B L A S ------" );
		try ( Blas blas = new Blas(8) ) {
			long start = System.nanoTime() ;
			double C[] = blas.mmul(rows, cols, A, B) ;
			long delta = System.nanoTime() - start ;
			printMatrix( rows, cols, C ) ;
			System.out.println( "Elapsed " + String.format( "%,d uS", (delta/1000) )) ;
		} catch( Throwable ignore ) {

		}

		try ( Blas blas = new Blas(8) ) {
			long start = System.nanoTime() ;
			double x[] = blas.solve(rows, mid, A, b) ;
			long delta = System.nanoTime() - start ;
			printMatrix(1, cols, x);
			System.out.println( "Elapsed " + String.format( "%,d uS", (delta/1000) )) ;
		} catch( Throwable ignore ) {
			ignore.printStackTrace();
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
