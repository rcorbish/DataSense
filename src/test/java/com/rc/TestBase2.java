package com.rc;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class TestBase2 {

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}


	
	@Test
	public void testData() {
		Matrix A = new Matrix( 5, 5 ) ;
		for( int i=0 ; i<A.length() ; i++ ) {
			A.data[i] = i ;
		}

		int ix = 0 ;
		for( int j=0 ; j<A.N ; j++ ) {
			for( int i=0 ; i<A.M ; i++ ) {
				assertEquals( "Unexpected value in A", ix, A.get(i, j), 1e-6) ;
				ix++ ;
			}
		}
	}

	@Test
	public void testPutcol() {
		Matrix A = Matrix.eye(4) ;
		for( int i=0 ; i<A.N ; i++ ) {
			A.putColumn(i, new double[] { i,i,i,i } ) ;
		}

		for( int j=0 ; j<A.N ; j++ ) {
			for( int i=0 ; i<A.M ; i++ ) {
				assertEquals( "Unexpected value in A", j, A.get(i, j), 1e-6) ;
			}
		}
	}

	@Test
	public void testGetcol() {
		Matrix A = Matrix.eye(4) ;

		for( int j=0 ; j<A.N ; j++ ) {
			Matrix C = A.extractColumns( 0 ) ;
			assertEquals( "Unexpected # rows in extract", A.M, C.M ) ;

			for( int i=0 ; i<C.M ; i++ ) {
				assertEquals( "Unexpected value in A", i==j?1:0, C.get(i, 0), 1e-6) ;
			}
		}
	}



	@Test
	public void testCopyCol() {
		Matrix A = Matrix.eye(4) ;

		for( int j=0 ; j<A.N ; j++ ) {
			Matrix C = A.copyColumns( j ) ;
			assertEquals( "Unexpected # rows in copy", A.M, C.M ) ;

			for( int i=0 ; i<C.M ; i++ ) {
				assertEquals( "Unexpected value in A", i==j?1:0, C.get(i, 0), 1e-6) ;
			}
		}
	}



	@Test
	public void testGetrow1() {
		Matrix A = Matrix.eye(4) ;

		for( int j=0 ; j<4 ; j++ ) {
			Matrix R = A.extractRows( 0 ) ;
			assertEquals( "Unexpected # cols in extract", A.N, R.N ) ;
			for( int i=0 ; i<R.N ; i++ ) {
				assertEquals( "Unexpected value in extract", i==j?1:0, R.get(0,i), 1e-6) ;
			}
		}
	}

	@Test
	public void testGetrow2() {
		Matrix A = Matrix.rand(5,3) ;

		Matrix R = A.dup().extractRows( 1,2 ) ;
		assertEquals( "Unexpected # cols in extract", A.N, R.N ) ;
		assertEquals( "Unexpected # rows in extract", 2, R.M ) ;

		for( int i=0 ; i<R.M ; i++ ) {
			for( int j=0 ; j<R.N ; j++ ) {
				assertEquals( "Unexpected value in extract", A.get( i+1,  j), R.get(i,j), 1e-6) ;
			}
		}
	}



	@Test
	public void testTriangularSquare() {
		Matrix A = Matrix.rand(4,4) ;
		Matrix B = A.upperTriangle() ;
		for( int i=1 ; i<A.M ; i++ ) {
			for( int j=Math.min( i, A.N) ; j>0 ; j-- ) {
				A.put( i,j-1,0 ) ;
			}
		}
		assertArrayEquals("Upper triangle failed on square", A.data, B.data, 1e-6 );
	}




	@Test
	public void testTriangularTall() {
		Matrix A = Matrix.rand(6,4) ;
		Matrix B = A.upperTriangle() ;
		for( int i=1 ; i<A.M ; i++ ) {
			for( int j=Math.min( i, A.N) ; j>0 ; j-- ) {
				A.put( i,j-1,0 ) ;
			}
		}
		assertArrayEquals("Upper triangle failed on tall", A.data, B.data, 1e-6 );
	}


	@Test
	public void testTriangularI() {
		Matrix A = Matrix.eye(4) ;
		Matrix B = A.upperTriangle() ;
		assertArrayEquals("Upper triangle failed on identity", A.data, B.data, 1e-6 );
	}

	@Test
	public void testTriangularSquare2() {
		Matrix A = new Matrix( 5, 5 ) ;
		for( int i=0 ; i<A.length() ; i++ ) {
			A.data[i] = i ;
		}
		Matrix U = A.upperTriangle() ;
		assertTrue( "Matrix U is not triangular", U.isTriangular );
		
		for( int i=0 ; i<A.M ; i++ ) {
			for( int j=0 ; j<A.N ; j++ ) {
				if( i>j ) A.put( i, j, 0 ) ;
			}
		}
		assertTrue( "Matrix A is not triangular", U.isTriangular() );
		assertArrayEquals( "Matrix data does not match", U.data, A.data, 1e-4 ) ;
	}

	

	@Test
	public void testTriangularTall2() {
		Matrix A = new Matrix( 10, 5 ) ;
		for( int i=0 ; i<A.length() ; i++ ) {
			A.data[i] = i ;
		}
		Matrix U = A.upperTriangle() ;
		assertTrue( "Matrix U is not triangular", U.isTriangular );
		
		for( int i=0 ; i<A.M ; i++ ) {
			for( int j=0 ; j<A.N ; j++ ) {
				if( i>j ) A.put( i, j, 0 ) ;
			}
		}
		assertTrue( "Matrix A is not triangular", A.isTriangular() );
		assertArrayEquals( "Matrix data does not match", U.data, A.data, 1e-4 ) ;
	}


	@Test
	public void testTriangularWide2() {
		Matrix A = new Matrix( 5, 15 ) ;
		for( int i=0 ; i<A.length() ; i++ ) {
			A.data[i] = i ;
		}
		Matrix U = A.upperTriangle() ;
		assertTrue( "Matrix U is not triangular", U.isTriangular );
		
		for( int i=0 ; i<A.M ; i++ ) {
			for( int j=0 ; j<A.N ; j++ ) {
				if( i>j ) A.put( i, j, 0 ) ;
			}
		}
		assertTrue( "Matrix A is not triangular", A.isTriangular() );
		assertArrayEquals( "Matrix data does not match", U.data, A.data, 1e-4 ) ;
	}



	@Test
	public void testMap() {
		Matrix A = Matrix.rand(4,4) ;
		A.map( (v,m,r,c) -> r+10*c ) ;

		for( int j=0 ; j<A.N ; j++ ) {
			for( int i=0 ; i<A.M ; i++ ) {
				assertEquals( "Unexpected value in A", i+10*j, A.get(i, j), 1e-6) ;
			}
		}
	}


	@Test
	public void testReduce() {
		Matrix A = Matrix.fill( 4, 4, 17 ) ;
		A.put( 0, 0, 20 ) ;
		A.put( 0, 1, 21 ) ;
		A.put( 0, 2, 22 ) ;
		A.put( 0, 3, 23 ) ;

		Matrix mx = A.reduce( (a,b) -> Math.max(a,b) ) ; 
		assertArrayEquals("Incorrect maximum", new double[] { 20, 21, 22, 23 }, mx.data, 1e-6 ) ;
	}


	@Test
	public void testReduceStart() {
		Matrix A = Matrix.rand( 4, 4 ) ;
		A.put( 0, 0, 20 ) ;
		A.put( 0, 1, 21 ) ;
		A.put( 0, 2, 22 ) ;
		A.put( 0, 3, 23 ) ;

		Matrix ct1 = A.reduce( (a,b) -> (a + ( b>170 ? 1 : 0)), 0 ) ; 
		assertArrayEquals("Incorrect maximum", new double[] { 0,0,0,0 }, ct1.data, 1e-6 ) ;


		Matrix ct2 = A.reduce( (a,b) -> (a + ( b>20 ? 1 : 0)), 2 ) ; 
		assertArrayEquals("Incorrect maximum", new double[] { 2,3,3,3 }, ct2.data, 1e-6 ) ;
	}


	@Test
	public void testIdentity() {
		Matrix A = Matrix.eye( 4 ) ;
		assertEquals( "Unexpected M size in identity", 4, A.M ) ;
		assertEquals( "Unexpected M size in identity", 4, A.N ) ;

		double t = 0 ;
		for( int i=0 ; i<A.length() ; i++ ) {
			t += A.data[i] ;
		} 

		assertEquals( "Unexpected values in identity matrix", 4, t, 1e-10 ) ;

		for( int i=0 ; i<A.M ; i++ ) {
			assertEquals( "Unexpected value in identity matrix on diagonal " + i, 1.0, A.get(i, i), 1e-10 ) ;
		} 
	}

}
