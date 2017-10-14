package com.rc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;

public class Cuda extends Compute {
	final static Logger log = LoggerFactory.getLogger( Cuda.class ) ;

	// cublasFillMode_t;
	public final static int CUBLAS_FILL_MODE_LOWER=0 ; 
	public final static int CUBLAS_FILL_MODE_UPPER=1 ;

	// cublasDiagType_t; 
	public final static int CUBLAS_DIAG_NON_UNIT=0 ;
	public final static int CUBLAS_DIAG_UNIT=1 ;

	//cublasSideMode_t; 
	public final static int CUBLAS_SIDE_LEFT =0 ; 
	public final static int CUBLAS_SIDE_RIGHT=1 ;

	//cublasOperation_t;
	public final static int CUBLAS_OP_N=0 ;
	public final static int CUBLAS_OP_T=1 ;
	public final static int CUBLAS_OP_C=2 ;  

	public interface CuBlas extends Library {
		CuBlas INSTANCE = (CuBlas)
				Native.loadLibrary((Platform.isWindows() ? "msvcrt" : "cublas"),
						CuBlas.class);

		int cublasCreate_v2( Pointer handle[] ) ;
		int cublasDestroy_v2( Pointer handle) ;
		int cublasAlloc( int n, int sz, Pointer buf[] ) ;
		int cublasFree( Pointer buf ) ;
		int cublasSetMatrix(int rows, int cols, int elemSize, double A[], int lda, Pointer B, int ldb ) ;
		int cublasGetMatrix(int rows, int cols, int elemSize, Pointer A, int lda, double B[], int ldb ) ;
		int cublasGetVersion_v2(Pointer handle, int version[] ) ;

		int cublasDgemm_v2(Pointer handle,
				int transa, int transb,
				int m, int n, int k,
				double          alpha[],
				Pointer         A, int lda,
				Pointer			B, int ldb,
				double          beta[],
				Pointer         C, int ldc) ;

		int cublasDtrsm_v2 (
				Pointer  handle, 
				int side,
				int uplo,
				int trans,
				int diag,
				int m,
				int n,
				double alpha[], 	  
				Pointer A,
				int lda,
				Pointer B,
				int ldb
				) ;
	}

	public interface CuSolver extends Library {
		CuSolver INSTANCE = (CuSolver)
				Native.loadLibrary((Platform.isWindows() ? "msvcrt" : "cusolver"),
						CuSolver.class);

		int cusolverDnCreate( Pointer handle[] ) ;
		int cusolverDnDestroy( Pointer handle) ;
		int cusolverDnDgeqrf_bufferSize( Pointer handle, int m, int n, Pointer A, int lda, int work[] ) ;
		int cusolverDnDgeqrf( 
				Pointer handle,
				int m, int n, 
				Pointer A, int lda,
				Pointer tau,
				Pointer work,
				int workSize,
				Pointer devInfo 
				) ;
		int cusolverDnDormqr( 
				Pointer handle,
				int side,
				int trans,
				int m, int n, int k,
				Pointer A,
				int lda,
				Pointer tau,
				Pointer B, int ldb,
				Pointer work,
				int workSize,
				Pointer devInfo 
				) ;

	}

	private final Pointer cublasHandle ;
	private final Pointer cusolverHandle ;
	private final int version ;
	
	private final int DoubleSize ;  // size of double in bytes
	private final int IntSize ;		// size of int in bytes
	
	private final double one[] = { 1.0 } ;
	private final double zero[] = { 0.0  };
	
	public Cuda() {
		
		DoubleSize = ( Double.SIZE / 8 ) ;   // bits ! bytes
		IntSize = ( Integer.SIZE / 8 ) ;   // bits ! bytes
		
		Pointer handle[] = new Pointer[1] ;
		int rc = CuBlas.INSTANCE.cublasCreate_v2( handle ) ;
		checkrc( rc ) ;
		this.cublasHandle = handle[0] ;

		rc = CuSolver.INSTANCE.cusolverDnCreate( handle ) ;
		checkrc( rc ) ;
		this.cusolverHandle = handle[0] ;

		int version[] = new int[1] ;
		CuBlas.INSTANCE.cublasGetVersion_v2( handle[0], version ) ;
		this.version = version[0] ;
		
		log.info( "Cuda version: {}", getVersion() ) ;

	}

	@Override
	public String getVersion() {
		return String.valueOf( version ) ;
	}
	
	@Override
	public void close() {
		if( cublasHandle != null ) CuBlas.INSTANCE.cublasDestroy_v2( cublasHandle ) ;    	
		if( cusolverHandle != null ) CuSolver.INSTANCE.cusolverDnDestroy( cusolverHandle ) ;    	
		log.info( "Diconected from GPU" ) ;
	}

	public Matrix mmul( Matrix A, Matrix B ) {
		Matrix C = new Matrix( A.M, B.N ) ;

		log.info( "mpy {} x {}  *  {} x {}", A.M, A.N, B.M, B.N ) ;

		Pointer gpuA=null, gpuB=null, gpuC=null ;
		
		try {
			gpuA = getMemory(A.M*A.N) ;
			gpuB = getMemory(B.M*B.N) ;
			gpuC = getMemory(A.M*B.N) ;
	
			log.info( "Sending A and B to GPU" ) ;
			int rc = CuBlas.INSTANCE.cublasSetMatrix(A.M, A.N, DoubleSize, A.data, A.M, gpuA, A.M ) ;
			checkrc( rc ) ;
			rc = CuBlas.INSTANCE.cublasSetMatrix(B.M, B.N, DoubleSize, B.data, B.M, gpuB, B.M ) ;
			checkrc( rc ) ;
			
			log.info( "Execute multiply" ) ;
			rc = CuBlas.INSTANCE.cublasDgemm_v2( cublasHandle, 
					CUBLAS_OP_N,  CUBLAS_OP_N,
					A.M,B.N,B.M, 
					one, 
					gpuA, A.M, 
					gpuB, B.M,
					zero, 
					gpuC, A.M
					) ;
			checkrc( rc ) ;
	
			log.info( "Copying C from GPU" ) ;
			rc = CuBlas.INSTANCE.cublasGetMatrix(C.M, C.N, DoubleSize, gpuC, C.M, C.data, C.M ) ;
			checkrc( rc ) ;
		} finally {
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuB ) ;
			CuBlas.INSTANCE.cublasFree( gpuC ) ;
		}	
		log.info( "mpy complete");
		return C ;
	}

	//
	// Ax = B
	//
	// QRx = B
	// Rx = Q'B
	//
	// x = solve R \ Q'B 
	//
	public Matrix solve( Matrix A, Matrix B ) {
		log.info( "Solve Ax=b  {} x {} ", A.M, A.N ) ;

		if( A.M<A.N) throw ( new RuntimeException( "M must be >= N" ) ) ;
		Matrix x = null ;
		
		Pointer gpuD=null, gpuA=null, gpuB=null, gpuW=null, gpuT=null ;
		try {
			gpuD = getMemory(1);					// device return code
			gpuA = getMemory(A.M*A.N);					// A
			gpuB = getMemory(B.M*B.N);		// this will also hold Q' x b   
	
			log.info( "Sending A and B to GPU" ) ;
			int rc = CuBlas.INSTANCE.cublasSetMatrix( A.M, A.N, DoubleSize, A.data, A.M, gpuA, A.M ) ;
			checkrc(rc) ;
			rc = CuBlas.INSTANCE.cublasSetMatrix( B.M, B.N, DoubleSize, B.data, B.M, gpuB, B.M ) ;
			checkrc(rc) ;
	
			// workspace size
			int work[] = new int[1] ;		
	
			log.info( "Calculating work area on GPU" ) ;
			rc = CuSolver.INSTANCE.cusolverDnDgeqrf_bufferSize(
					cusolverHandle, 
					A.M, B.N, 
					gpuA, A.M, 
					work
					) ; 		
			checkrc( rc ) ;
			int lwork = work[0] ;
			gpuW = getMemory(lwork) ;
			log.debug( "Allocated double[{}] on GPU", lwork ) ;
	
			// QR ( step 1 )
			gpuT = getMemory( Math.min(A.M, A.N) ) ;
			log.info( "Perform QR = A" ) ;
			rc = CuSolver.INSTANCE.cusolverDnDgeqrf( 
					cusolverHandle, 
					A.M, B.N, 
					gpuA, A.M, 
					gpuT, 
					gpuW, lwork, 
					gpuD 
					) ; 
			checkrc( rc ) ;
	//		printMatrix( 1, N, gpuT ) ;
	
			/*  if we want Q - uncomment this
			// Q
			Pointer gpuQ = getMemory(M*M) ;
			double q[] = new double[M*M] ;
			for( int i=0 ; i<M ; i++ ) q[i*M+i] = 1 ;
			rc = CuBlas.INSTANCE.cublasSetMatrix(M, M, DoubleSize, q, M, gpuQ, M ) ;
			checkrc( rc ) ;
			
			rc = CuSolver.INSTANCE.cusolverDnDormqr(
					cuusolverHandle, 
					CUBLAS_SIDE_LEFT, CUBLAS_OP_N, 
					M, M, Math.min(M,N), 
					gpuA, M,
					gpuT, 
					gpuQ, M, 
					gpuW, 
					lwork, 
					gpuD
					) ; 
			checkrc( rc ) ;
			printMatrix( M, M, gpuQ ) ;
	*/
			
			// Q' x b   -> gpuB		
			log.info( "Perform Q' x b" ) ;
			rc = CuSolver.INSTANCE.cusolverDnDormqr(
					cusolverHandle, 
					CUBLAS_SIDE_LEFT, CUBLAS_OP_T, 
					A.M, B.N, Math.min(A.M,A.N), 
					gpuA, A.M,	
					gpuT, 
					gpuB, B.M, 
					gpuW, 
					lwork, 
					gpuD
					) ; 
			checkrc( rc ) ;


			//--------------------------------------
			// Solve R x = Q' x b   to find x
			// R is upper triangular 
	
			log.info( "Solve Rx = Q' x b" ) ;
			rc = CuBlas.INSTANCE.cublasDtrsm_v2(
					cublasHandle, 
					CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
					A.N, B.N, 
					one, 
					gpuA, A.M, 
					gpuB, B.M
					) ;
			checkrc( rc ) ;
//			printMatrix( M, numFeatures, gpuB ) ;
	
			log.info( "Copying x from GPU" ); 
			x = new Matrix( B.M, B.N ) ;
			rc = CuBlas.INSTANCE.cublasGetMatrix(A.M, B.N, DoubleSize, gpuB, B.M, x.data, B.M ) ;
			checkrc( rc ) ;
		} finally {		
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuB ) ;
			CuBlas.INSTANCE.cublasFree( gpuW ) ;
			CuBlas.INSTANCE.cublasFree( gpuD ) ;
			CuBlas.INSTANCE.cublasFree( gpuT ) ;
		}
		log.info( "Solved x" ) ;
		return x ;
	}

	//
	// xA = B
	// 
	// (xA)' = B'
	//
	// A'x'  = B'
	// QRx'  = B'
	// Rx'   = Q'B'
	//
	// x = solve R \ Q'B' 
	//
	public Matrix solve2( Matrix A, Matrix B ) {
		return null ;
	}

	protected void printMatrix( int M, int N, Pointer A ) {
		double a[] = new double[M*N] ;
		int rc = CuBlas.INSTANCE.cublasGetMatrix( M, N, DoubleSize, A, M, a, M ) ;
		System.out.println( ".... " + rc ) ;
		
		for( int i=0 ; i<Math.min( 8, M) ; i++ ) {
			for( int j=0 ; j<Math.min( 8, N) ; j++ ) {
				int ix = i + j*M ;
				System.out.print( String.format( "%10.3f", a[ix] ) );
			}
			System.out.println(); 
		}
		
		System.out.println( "...." ) ;
	}

	protected void checkrc( int rc ) {
		if( rc == 0 ) return ;
		StackTraceElement ste = Thread.currentThread().getStackTrace()[2] ;
		if( ste.getMethodName().equals( "getMemory") ) {
			ste = Thread.currentThread().getStackTrace()[3] ;
		}
		log.error( "Error {} at: {}",  rc, ste ) ;
		throw new RuntimeException( "Failed to check RC" ) ;
	}
	
	protected Pointer getMemory( int numDoubles ) {
		Pointer ptr[] = new Pointer[1] ;
		int rc = CuBlas.INSTANCE.cublasAlloc( numDoubles, DoubleSize, ptr ) ;
		checkrc(rc) ;		
		return ptr[0] ;
	}
}
	