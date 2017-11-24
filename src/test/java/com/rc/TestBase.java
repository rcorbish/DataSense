package com.rc;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

@Ignore
public class TestBase {

	Compute test ;

	@Before
	public void setUp() throws Exception {
		test = Compute.getInstance() ;
	}

	@After
	public void tearDown() throws Exception {
		test.close(); 
	}


	@Test
	public void testAdd1() {
		Matrix A = Matrix.rand(4,4) ;
		Matrix B = Matrix.rand(4,4) ;

		Matrix C = A.add( B ) ;
		double tst[] = new double[16] ;
		for( int i=0 ; i<A.length() ; i++ ) {
			tst[i] = A.data[i] + B.data[i] ;
		}		
		assertArrayEquals( "Unexpected value in A+B", tst, C.data, 1e-6) ;
	}


	@Test
	public void testAdd2() {
		Matrix A = Matrix.rand(4,4) ;
		Matrix B = Matrix.rand(1,4) ;

		Matrix C = A.add( B ) ;
		double tst[] = new double[16] ;
		for( int i=0 ; i<A.length() ; i++ ) {
			tst[i] = A.data[i] + B.data[i/4] ;
		}		
		assertArrayEquals( "Unexpected value in A+B", tst, C.data, 1e-6) ;
	}


	@Test
	public void testAdd3() {
		Matrix A = Matrix.rand(4,4) ;
		Matrix B = Matrix.rand(4,1) ;

		Matrix C = A.add( B ) ;
		
		double tst[] = new double[16] ;
		for( int i=0 ; i<A.length() ; i++ ) {
			tst[i] = A.data[i] + B.data[i%4] ;
		}		
		assertArrayEquals( "Unexpected value in A+B", tst, C.data, 1e-6) ;
	}




	@Test
	public void testOuter() {
		Matrix A = new Matrix( 4, 1,    1, 2, 3, 4  ) ; 
		Matrix B = new Matrix( 4, 1,    -1, 2, -3, 4  ) ; 

		Matrix kp = A.outer( B ) ;

		double real[] = new double[] {
			-1 ,  -2 ,  -3,   -4,
			2  ,  4  ,  6 ,   8,
		   -3  , -6  , -9 , -12,
			4  ,  8 ,  12 ,  16
		} ;
		assertArrayEquals( "Incorrect outer result", real, kp.data, 1e-6 ) ;
	}


	@Test
	public void testDetSquare() {
		Matrix A = new Matrix( 3, 3,    1, 2, 3,   -7, 9,-2,   8,0,4  ) ; 
		double d = A.det() ;
		assertEquals( "Incorrect determinant", -156, d, 1e-6 ) ;
	}



	@Test
	public void testDetTriangle() {
		Matrix A = new Matrix( 3, 3,    1, 0, 0,   -7, 9,0,   8,0,4  ) ; 
		double d = A.det() ;
		assertEquals( "Incorrect determinant", 36, d, 1e-6 ) ;

		A.isTriangular = true ;
		double d2 = A.det() ;
		assertEquals( "Incorrect determinant (fast)", d, d2, 1e-6 ) ;
	}



	@Test
	public void testDetNonSquare() { 
		try {
			Matrix A = new Matrix( 4, 3,    1, 2, 3,0.3,   -7, 9,-2,-4,   8,0,4,0  ) ; 
			A.det() ;
			fail( "Non-square determinant is not alloweed" ) ;
		} catch( IllegalArgumentException ok ) {
			// expect an error
		}
	}




	@Test
	public void testDot() {
		Matrix A = new Matrix( 4, 1,    1, 2, 3, 4  ) ; 
		Matrix B = new Matrix( 4, 1,    -1, 2, -3, 4  ) ; 

		double dot = A.dot( B ) ;

		double real = 0 ;
		for( int i=0 ; i<A.length() ; i++ ) {
			real += A.get(i, 0) * B.get( i, 0 ) ;
		}

		assertEquals( "Incorrect dot result", real, dot, 1e-6 ) ;
	}



	@Test
	public void testMmul() {
		Matrix A = new Matrix( 4, 3,    1.0, 2.0, -2, 0.7,    3.0, -3.0, -1.0, 7.0,    0.5, 3, 8, -0.3 ) ; 
		Matrix B = new Matrix( 3, 2,    6,-3,4,   7,8,-2  ) ; 

		Matrix C = A.mmul( B ) ;
		assertEquals( "Unexpected M size in mmul", A.M, C.M ) ;
		assertEquals( "Unexpected N size in mmul", B.N, C.N ) ;

		Matrix D = new Matrix( 4,2 ) ;

		for( int i=0 ; i<A.M ; i++ ) {
			for( int j=0 ; j<B.N ; j++ ) {
				double v = 0 ;
				for( int k=0 ; k<A.N ; k++ ) {
					v +=  A.get( i, k ) * B.get( k, j ) ;
				}
				D.put( i, j, v );
			}
		}

		assertArrayEquals( "Incorrect mmul result", D.data, C.data, 1e-6 ) ;
	}

	@Test
	public void testAddiVV() {
		Matrix A = new Matrix( 1, 4, 1,2,3,4 ) ;
		Matrix B = new Matrix( 4, 1, .1, .2, .3, .4 ) ;

		A.addi( B ) ;
		double expected[] = new double[] { 1.1, 2.2, 3.3, 4.4 } ;
		assertArrayEquals( "Incorrect addi (vector/vector) result", expected, A.data, 1e-6 ) ;
	}


	@Test
	public void testAddiVM() {
		Matrix A = new Matrix( 2, 4, 1,5, 2,6, 3,7, 4,8 ) ;
		Matrix B = new Matrix( 1, 4, .1, .2, .3, .4 ) ;

		A.addi( B ) ;
		double expected[] = new double[] { 1.1, 5.1, 2.2, 6.2, 3.3, 7.3, 4.4, 8.4 } ;
		assertArrayEquals( "Incorrect addi (Matrix/row vector) result", expected, A.data, 1e-6 ) ;
	}

	@Test
	public void testAddiVM2() {
		Matrix A = new Matrix( 2, 4, 1,5, 2,6, 3,7, 4,8 ) ;
		Matrix B = new Matrix( 2, 1, .1, .2 ) ;

		A.addi( B ) ;
		double expected[] = new double[] { 1.1, 5.2, 2.1, 6.2, 3.1, 7.2, 4.1, 8.2 } ;
		assertArrayEquals( "Incorrect addi (Matrix/col Vector) result", expected, A.data, 1e-6 ) ;
	}

	@Test
	public void testAddiMM() {
		Matrix A = Matrix.rand( 6,6 ) ;
		Matrix B = Matrix.rand( 6,6 ) ;

		double expected[] = new double[A.M * A.N];
		for( int i=0 ; i<expected.length ; i++ ) {
			expected[i] = A.get(i) + B.get(i) ;
		}

		A.addi( B ) ;
		assertArrayEquals( "Incorrect addi (Matrix/Matrix) result", expected, A.data, 1e-6 ) ;
	}



	@Test
	public void testSubiVV() {
		Matrix A = new Matrix( 1, 4, 1,2,3,4 ) ;
		Matrix B = new Matrix( 4, 1, .1, .2, .3, .4 ) ;

		A.subi( B ) ;
		double expected[] = new double[] { 0.9, 1.8, 2.7, 3.6 } ;
		assertArrayEquals( "Incorrect subi (vector/vector) result", expected, A.data, 1e-6 ) ;
	}


	@Test
	public void testSubiVM() {
		Matrix A = new Matrix( 2, 4, 1,5, 2,6, 3,7, 4,8 ) ;
		Matrix B = new Matrix( 1, 4, .1, .2, .3, .4 ) ;

		A.subi( B ) ;
		double expected[] = new double[] { 0.9, 4.9, 1.8, 5.8, 2.7, 6.7, 3.6, 7.6 } ;
		assertArrayEquals( "Incorrect subi (Matrix/row vector) result", expected, A.data, 1e-6 ) ;
	}

	@Test
	public void testSubiVM2() {
		Matrix A = new Matrix( 2, 4, 1,5, 2,6, 3,7, 4,8 ) ;
		Matrix B = new Matrix( 2, 1, .1, .2 ) ;

		A.subi( B ) ;
		double expected[] = new double[] { 0.9, 4.8, 1.9, 5.8, 2.9, 6.8, 3.9, 7.8 } ;
		assertArrayEquals( "Incorrect subi (Matrix/col Vector) result", expected, A.data, 1e-6 ) ;
	}

	@Test
	public void testSubiMM() {
		Matrix A = Matrix.rand( 6,6 ) ;
		Matrix B = Matrix.rand( 6,6 ) ;

		double expected[] = new double[A.M * A.N];
		for( int i=0 ; i<expected.length ; i++ ) {
			expected[i] = A.get(i) - B.get(i) ;
		}

		A.subi( B ) ;
		assertArrayEquals( "Incorrect subi (Matrix/Matrix) result", expected, A.data, 1e-6 ) ;
	}



	@Test
	public void testMatrixError() {
		try {
			Matrix A = new Matrix( 3, 3,    1.0, 2.0,-2,     3.0, -3.0, -1.0,    0.5, 3, 8, -0.3 ) ; 
			Matrix B = new Matrix( 2, 3,    6,-3,4,   7,8,-2  ) ; 

			A.mmul( B ) ;
			fail( "Did not trap incompatible matrix sizes" ) ;
		} catch( RuntimeException expected ) {

		}
	}


	@Test
	public void testSolveSquare() {

		Matrix A = new Matrix( 3, 3,    1.0, 2.0,-2,     3.0, -3.0, -1.0,    0.5, 3, 8 ) ; 
		Matrix B = new Matrix( 3, 3,    2,4,-4,   3,-3,-1,   3,18,48  ) ; 
		Matrix B2 = B.dup();
		Matrix A2 = A.dup();

		Matrix X = test.solve(A, B) ;
		Matrix C = A2.mmul(X) ;

		for( int i=0 ; i<C.length() ; i++ ) {
			assertEquals( "Solver (AX=B) incorrect solution for square matrix", C.data[i], B2.data[i], 1e-6 ) ;
		}
	}

	@Test
	public void testSolveTall() {
		Matrix A = new Matrix( 5, 3,    1.0, 2.0,-2,-3,5,     3.0, -3.0, -1.0,2,8,    0.5, 3, 8,16,-9 ) ; 
		Matrix B = new Matrix( 5, 3,    0.40303 ,   3.49899 ,   4.58930,    9.09309 ,  -4.78863,
				2.33787 ,   2.93620 ,  -1.97189,   -1.46035 ,   8.09298,
				6.58104 ,   2.86530 ,  19.70198,   46.38339 , -10.15150  ) ; 
		Matrix B2 = B.dup();
		Matrix A2 = A.dup();

		Matrix X = test.solve(A, B) ;

		Matrix C = A2.mmul(X) ;

		for( int i=0 ; i<C.length() ; i++ ) {
			assertEquals( "Solver (AX=B) incorrect solution for tall matrix", C.data[i], B2.data[i], 1e-5 ) ;
		}
	}

	@Test
	public void testSolveVector() {
		Matrix A = new Matrix( 5, 3,    
				0.57978 ,  0.93114 ,  0.20273 ,  0.58061 ,  0.74229 ,
				0.48296 ,  0.19064 ,  0.47324 ,  0.37277 ,  0.75313 ,
				0.76184 ,  0.53417 ,  0.93904 ,  0.34721 ,  0.76354  ) ; 
		Matrix B = new Matrix( 5, 1,    0.464330,  0.837568,  0.083792,  0.509288,  0.746724 ) ; 

		Matrix X = test.solve(A, B) ;

		assertEquals( "Solver (AX=B) incorrect factor X[0]", 0.96543, X.data[0], 1e-5 ) ;
		assertEquals( "Solver (AX=B) incorrect factor X[1]", 0.21751, X.data[1], 1e-5 ) ;
		assertEquals( "Solver (AX=B) incorrect factor X[2]", -0.22744, X.data[2], 1e-5 ) ;

	}

	@Test 
	public void testConjugateGradientDescent() {
		Matrix A = new Matrix( 5, 4,
				1.000000,   1.000000,   1.000000,   1.000000,   1.000000,
			    0.747288,   0.806088,   0.292693,   0.079899,   0.630942,
			    0.500764,   0.380189,   0.464858,   0.645147,   0.490761,
			    0.124043,   0.330575,   0.880164,   0.982739,   0.723298  ) ; 
		Matrix B = new Matrix( 5, 1,     4.4930,  5.0760,  6.5010,  7.0270, 6.6270 ) ; 
		
		Cgd.CostFunction cost = new Cgd.CostFunction() {			
			@Override
			public double cost(Matrix X, Matrix y, Matrix theta, double lambda) {
				Matrix S = X.mmul( theta ).subi( y ) ;		
				double j1 = S.hmuli(S).total() ;

				Matrix t = theta.hmul( theta ).muli( lambda ) ;
				double cost = ( t.total() - t.get(0) + j1 ) / ( 2 * y.length() ) ;

				return cost ;
			}
		};
		
		Cgd.GradientsFunction grad = new Cgd.GradientsFunction() {			
			@Override
			public Matrix grad(Matrix X, Matrix y, Matrix theta, double lambda) {
				Matrix S = X.mmul( theta ).subi( y ) ;		
				Matrix G1 = X.transpose().mmul( S ) ;

				Matrix G2 = theta.mul( lambda ) ;
				G2.put( 0, 0.0 ) ;

				return G2.addi( G1 ).divi( y.length() ) ;
			}
		}; 
		
		double LAMBDA 	= 0.00002;	

		Cgd cgd = new Cgd( ) ; //(sc,it) -> { if( (it%500)==0 ){ System.out.println( it + " : " + sc ) ;} } ) ;

		double gradErr = cgd.checkGrad( cost, grad ) ;
		assertTrue( "Gradient functions are incorrect - error too large", gradErr<1e-8 ) ;

		Matrix X = cgd.solve( cost, grad, A, B, LAMBDA, 1_000 ) ;
		for( int i=0 ; i<X.length() ; i++ ) {
			assertEquals( "CGD solver (AX=B) incorrect factor X[" + i + "]", (i+1), X.data[i], 0.02 ) ;
		}
		Matrix Bpredict = A.mmul( X ) ;
		Bpredict.subi( B ) ;
		double err = Bpredict.map( v -> v*v ).total() ;
		assertTrue( "Total error too big", err<1e-5 ) ;
		
		for( int i=0 ; i<Bpredict.length() ; i++ ) {
			assertEquals( "CGD solver (AX=B) incorrect prediction[" + i + "]", 0, Bpredict.data[i], 0.001 ) ;
		}
	}
	

	@Test 
	public void testLogisticRegression() {
		Matrix A = new Matrix( 100, 2 ) ; 
		Matrix B = new Matrix( A.M, 1 ) ;
		Random rng = new Random( 10 ) ;
		for( int i=0 ; i<A.M ; i++ ) {
			A.put( i, 0, rng.nextInt( 100 )/10.0 ); 
			A.put( i, 1, rng.nextInt( 100 )/10.0 );
			
			B.put( i, A.get( i,1 ) > (3*A.get( i,0 )-150) ? 1 : 0 );
		}
		
		
		Cgd.CostFunction cost = new Cgd.CostFunction() {			
			@Override
			public double cost(Matrix X, Matrix y, Matrix theta, double lambda) {
				Matrix ht = X.mmul( theta ) ;
				ht.map( v -> sigmoid(v) ) ;
				Matrix loght = ht.dup().map( v -> Math.log(v) ) ;
				Matrix loght2 = ht.dup().map( v -> Math.log(1.0-v) ) ;

				double ts = theta.total() - theta.get(0) ;

				double J = loght.hmuli( y.mul(-1) ).subi( loght2.hmuli( y.mul(-1).add(1) ) ).total() / y.length() ;
				J += lambda * ts * ts / (2 * y.length() ) ;
				
				return J;
			}
		};
		
		Cgd.GradientsFunction grad = new Cgd.GradientsFunction() {			
			@Override
			public Matrix grad(Matrix X, Matrix y, Matrix theta, double lambda) {
				Matrix ht = X.mmul( theta ) ;
				ht.map( v -> sigmoid(v) ) ;

				Matrix G1 = X.hmul( ht.subi( y ) ).sum().muli( 1.0/y.length() ) ;
				Matrix G2 = theta.mul( lambda/y.length() ) ;
				G2.put(0,  0.0 ); 
				
				return G2.addi( G1.transpose() ) ;
			}
		}; 
		
		double LAMBDA 	= 0.001;	

		Matrix X = Matrix.fill( A.M, 1, 1.0 ).appendColumns( A ) ;

		
		Cgd cgd = new Cgd( ) ; 
		
		double gradErr = cgd.checkGrad( cost, grad ) ;
		assertTrue( "Gradient functions are incorrect - error too large", gradErr<1e-8 ) ;
		
		Matrix T = cgd.solve( cost, grad, X, B, LAMBDA, 300 ) ;

		assertEquals( "Theta has incorrect number of columns",  1, T.N ) ;
		assertEquals( "Theta has incorrect number of rows", X.N, T.M ) ;
		
		System.out.println( "Theta = " + T ) ;
		
		Matrix YH = X.mmul( T ) ;
		YH.map( v -> sigmoid(v) ) ;
		
//		for( int i=0 ; i<YH.length() ; i++ ) {
//			System.out.println( String.format("YH[%d] %6.4f",  i, YH.data[i] ) ) ;
//		}
		
		assertArrayEquals(  "Error is too big for this simple solution", B.data, YH.data, 0.1 ) ;
	}
	
	protected double sigmoid( double z ) {
		return  1.0 / ( Math.exp(-z) + 1 ) ;
	}
}
