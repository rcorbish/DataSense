package com.rc;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;

public class Cuda implements AutoCloseable {

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

	private final Pointer cuublasHandle ;
	private final Pointer cuusolverHandle ;
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
		this.cuublasHandle = handle[0] ;

		rc = CuSolver.INSTANCE.cusolverDnCreate( handle ) ;
		checkrc( rc ) ;
		this.cuusolverHandle = handle[0] ;

		int version[] = new int[1] ;
		CuBlas.INSTANCE.cublasGetVersion_v2( handle[0], version ) ;
		this.version = version[0] ;				
	}

	@Override
	public void close() {
		CuBlas.INSTANCE.cublasDestroy_v2( cuublasHandle ) ;    	
		CuSolver.INSTANCE.cusolverDnDestroy( cuusolverHandle ) ;    	
	}

	public double[] mmul( int rows, int cols, double A[], double B[] ) {
		int M = rows ;
		int K = A.length / M ;
		int N = cols ;
		double C[] = new double[M*N] ;

		Pointer gpuA=null, gpuB=null, gpuC=null ;
		
		try {
			gpuA = getMemory(M*K) ;
			gpuB = getMemory(K*N) ;
			gpuC = getMemory(M*N) ;
	
			int rc = CuBlas.INSTANCE.cublasSetMatrix(M, K, DoubleSize, A, M, gpuA, M ) ;
			checkrc( rc ) ;
			rc = CuBlas.INSTANCE.cublasSetMatrix(K, N, DoubleSize, B, K, gpuB, K ) ;
			checkrc( rc ) ;
			
			rc = CuBlas.INSTANCE.cublasDgemm_v2( cuublasHandle, 
					CUBLAS_OP_N,  CUBLAS_OP_N,
					M,N,K, 
					one, gpuA, M, 
					gpuB, K,
					zero, gpuC, M
					) ;
			checkrc( rc ) ;
	
			rc = CuBlas.INSTANCE.cublasGetMatrix(M, N, DoubleSize, gpuC, M, C, M ) ;
			checkrc( rc ) ;
		} finally {
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuB ) ;
			CuBlas.INSTANCE.cublasFree( gpuC ) ;
		}	
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
	public double[] solve( int rows, int cols, double A[], double B[] ) {
		int M = rows ;
		int N = cols ;

		if( M<N) throw ( new RuntimeException( "M must be >= N" ) ) ;
		double x[] = null ;
		
		Pointer gpuD=null, gpuA=null, gpuB=null, gpuW=null, gpuT=null ;
		try {
			gpuD = getMemory(1);		// device return code
			gpuA = getMemory(M*N);		// A
			gpuB = getMemory(M*1);		// this will also hold Q' x b   ( 1st column only ) 
	
			// Copy A and b to GPU
			int rc = CuBlas.INSTANCE.cublasSetMatrix(M, N, DoubleSize, A, M, gpuA, M ) ;
			checkrc(rc) ;
			rc = CuBlas.INSTANCE.cublasSetMatrix(M, 1, DoubleSize, B, M, gpuB, M ) ;
			checkrc(rc) ;
	
			// workspace size
			int work[] = new int[1] ;		
	
			rc = CuSolver.INSTANCE.cusolverDnDgeqrf_bufferSize(cuusolverHandle, M, 1, gpuA, M, work) ; 		
			checkrc( rc ) ;
			int lwork = work[0] ;
			gpuW = getMemory(lwork) ;
	
			// QR ( step 1 )
			gpuT = getMemory( Math.min(M, N) ) ;
			rc = CuSolver.INSTANCE.cusolverDnDgeqrf( 
					cuusolverHandle, 
					M, N, 
					gpuA, M, 
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
			rc = CuSolver.INSTANCE.cusolverDnDormqr(
					cuusolverHandle, 
					CUBLAS_SIDE_LEFT, CUBLAS_OP_T, 
					M, 1, Math.min(M,N), 
					gpuA, M,	
					gpuT, 
					gpuB, M, 
					gpuW, 
					lwork, 
					gpuD
					) ; 
			checkrc( rc ) ;
	//		printMatrix( M, 1, gpuB ) ;
	
			//--------------------------------------
			// Solve R x = Q' x b   to find x
			// R is upper triangular 
	
			rc = CuBlas.INSTANCE.cublasDtrsm_v2(
					cuublasHandle, 
					CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
					N, 1, one, 
					gpuA, M, 
					gpuB, N
					) ;
			checkrc( rc ) ;
	
			x = new double[N] ;
			rc = CuBlas.INSTANCE.cublasGetMatrix(N, 1, DoubleSize, gpuB, N, x, N ) ;
			checkrc( rc ) ;
		} finally {		
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuB ) ;
			CuBlas.INSTANCE.cublasFree( gpuW ) ;
			CuBlas.INSTANCE.cublasFree( gpuD ) ;
			CuBlas.INSTANCE.cublasFree( gpuT ) ;
		}
		return x ;
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
		System.err.println( "Error code [" + rc + "] at: " + ste ) ;
		throw new RuntimeException( "Failed to check RC" ) ;
	}
	
	protected Pointer getMemory( int numDoubles ) {
		Pointer ptr[] = new Pointer[1] ;
		int rc = CuBlas.INSTANCE.cublasAlloc( numDoubles, DoubleSize, ptr ) ;
		checkrc(rc) ;		
		return ptr[0] ;
	}
}
	