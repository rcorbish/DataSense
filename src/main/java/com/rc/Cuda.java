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
				Native.loadLibrary((Platform.isWindows() ? "libcublas" : "cublas"),
						CuBlas.class);

		int cublasCreate_v2( Pointer handle[] ) ;
		int cublasDestroy_v2( Pointer handle) ;
		int cublasAlloc( int n, int sz, Pointer buf[] ) ;
		int cublasFree( Pointer buf ) ;
		int cublasSetMatrix(int rows, int cols, int elemSize, double A[], int lda, Pointer B, int ldb ) ;
		int cublasGetMatrix(int rows, int cols, int elemSize, Pointer A, int lda, double B[], int ldb ) ;
		int cublasSetVector(int len, int elemSize, double A[], int lda, Pointer B, int ldb ) ;
		int cublasGetVector(int len, int elemSize, Pointer A, int lda, double B[], int ldb ) ;
		int cublasGetVector(int len, int elemSize, Pointer A, int lda, int B[], int ldb ) ;
		int cublasGetVersion_v2(Pointer handle, int version[] ) ;

		int cublasDgemm_v2(Pointer handle,
				int transa, int transb,
				int m, int n, int k,
				double          alpha[],
				Pointer         A, int lda,
				Pointer			B, int ldb,
				double          beta[],
				Pointer         C, int ldc) ;

		int cublasDdot_v2(
				Pointer	handle ,
				int n,
				Pointer x, int incx,
				Pointer y, int incy,
				double rc[]
                ) ;

		int cublasDnrm2_v2(
				Pointer	handle ,
				int n,
				Pointer x, int incx,
				double rc[]
                ) ;

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
		
		int cusolverDnDgetrf_bufferSize( Pointer handle, int m, int n, Pointer A, int lda, int work[] ) ;
		int cusolverDnDgetrf( 
				Pointer handle,
				int m, int n, 
				Pointer A, int lda,
				Pointer work,
				Pointer ipiv,
				Pointer devInfo 
				) ;

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

	@Override
	public double dot( Matrix A, Matrix B ) {
		if( !A.isVector || !B.isVector ) throw new RuntimeException( String.format( "Dot product requires vectors" ) )  ;
		if( A.length() != B.length() ) throw new RuntimeException( String.format( "Incompatible matrix sizes: %d  and %d", A.length(), B.length() ) )  ;

		Pointer gpuA = null, gpuB = null;
		double dotProduct[] = new double[1] ;
		try {
			gpuA = getMemory( A.length() ) ;
			gpuB = getMemory( B.length() ) ;
			
			int rc = CuBlas.INSTANCE.cublasSetVector(A.length(), DoubleSize, A.data, 1, gpuA, 1 ) ;
			checkrc( rc ) ;
			rc = CuBlas.INSTANCE.cublasSetVector(B.length(), DoubleSize, B.data, 1, gpuB, 1 ) ;
			checkrc( rc ) ;

			rc = CuBlas.INSTANCE.cublasDdot_v2( 
				cublasHandle,
				A.length(), 
				gpuA, 1, 
				gpuB, 1,
				dotProduct
				) ;
			checkrc( rc ) ;
			
		} finally {
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuB ) ;
		}
		return dotProduct[0] ;
	}


	@Override
	public double norm( Matrix A ) {
		if( !A.isVector ) throw new RuntimeException( String.format( "Norm requires a vector" ) )  ;

		Pointer gpuA = null ;
		double norm[] = new double[1] ;
		try {
			gpuA = getMemory( A.length() ) ;
			
			int rc = CuBlas.INSTANCE.cublasSetVector(A.length(), DoubleSize, A.data, 1, gpuA, 1 ) ;
			checkrc( rc ) ;

			rc = CuBlas.INSTANCE.cublasDnrm2_v2( 
				cublasHandle,
				A.length(), 
				gpuA, 1, 
				norm
				) ;
			checkrc( rc ) ;
			
		} finally {
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
		}
		return norm[0] ;
	}

	@Override
	public Matrix mmul( Matrix A, Matrix B ) {
		if( A.N != B.M ) throw new RuntimeException( String.format( "Incompatible matrix sizes: %d x %d  and %d x %d", A.M, A.N, B.M, B.N ) )  ;

		Matrix C = new Matrix( A.M, B.N ) ;

		Pointer gpuA=null, gpuB=null, gpuC=null ;
		
		try {
			gpuA = getMemory(A.M*A.N) ;
			gpuB = getMemory(B.M*B.N) ;
			gpuC = getMemory(A.M*B.N) ;
	
			log.debug( "Sending A and B to GPU" ) ;
			int rc = CuBlas.INSTANCE.cublasSetMatrix(A.M, A.N, DoubleSize, A.data, A.M, gpuA, A.M ) ;
			checkrc( rc ) ;
			rc = CuBlas.INSTANCE.cublasSetMatrix(B.M, B.N, DoubleSize, B.data, B.M, gpuB, B.M ) ;
			checkrc( rc ) ;
			
			log.debug( "Execute multiply" ) ;
			rc = CuBlas.INSTANCE.cublasDgemm_v2( cublasHandle, 
					CUBLAS_OP_N,  CUBLAS_OP_N,
					A.M,B.N,B.M, 
					one, 
					gpuA, A.M, 
					gpuB, B.M,
					zero, 
					gpuC, C.M
					) ;
			checkrc( rc ) ;
	
			log.debug( "Copying C from GPU" ) ;
			rc = CuBlas.INSTANCE.cublasGetMatrix(C.M, C.N, DoubleSize, gpuC, C.M, C.data, C.M ) ;
			checkrc( rc ) ;
		} finally {
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuB ) ;
			CuBlas.INSTANCE.cublasFree( gpuC ) ;
		}	
		log.debug( "mpy complete");
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
	//
	// IN
	// 	A is A.M x A.N  
	// 	B is B.M x B.N
	//
	// OUT
	// 	Q is A.M x A.M    	( square & orthogonal )
	// 	R is A.M x A.N		( same as A )
	//  Q'B' is A.M x B.N	 
	//  X is A.N x B.N		
	//
	public Matrix solve( Matrix A, Matrix B ) {
		log.debug( "Solve Ax=b  {} x {} ", A.M, A.N ) ;

		if( A.M<A.N) throw ( new RuntimeException( "M must be >= N" ) ) ;
		//if( A.M!=B.M) throw ( new RuntimeException( "M must be the same" ) ) ;
		Matrix x = null ;
		
		Pointer gpuD=null, gpuA=null, gpuB=null, gpuW=null, gpuT=null ;
		try {
			gpuD = getMemory(1);				// device return code
			gpuA = getMemory(A.M*A.N);			// A
			gpuB = getMemory(A.M*B.N);			// this will also hold Q' x b   
	
			log.debug( "Sending A and B to GPU" ) ;
			int rc = CuBlas.INSTANCE.cublasSetMatrix( A.M, A.N, DoubleSize, A.data, A.M, gpuA, A.M ) ;
			checkrc(rc) ;
			rc = CuBlas.INSTANCE.cublasSetMatrix( B.M, B.N, DoubleSize, B.data, B.M, gpuB, B.M ) ;
			checkrc(rc) ;
	
			// workspace size
			int work[] = new int[1] ;		
	
			log.debug( "Calculating work area on GPU" ) ;
			rc = CuSolver.INSTANCE.cusolverDnDgeqrf_bufferSize(
					cusolverHandle, 
					A.M, A.N, 
					gpuA, A.M, 
					work
					) ; 		
			checkrc( rc ) ;
			int lwork = work[0] ;
			gpuW = getMemory(lwork) ;
			log.debug( "Allocated double[{}] on GPU", lwork ) ;
	
			// QR ( step 1 )
			gpuT = getMemory( Math.min(A.M, A.N) ) ;
			rc = CuSolver.INSTANCE.cusolverDnDgeqrf( 
					cusolverHandle, 
					A.M, A.N, 
					gpuA, A.M, 
					gpuT, 
					gpuW, lwork, 
					gpuD 
					) ; 
			checkrc( rc ) ;
			log.debug( "factored QR <- A" ) ;
//			printMatrix(A.M, A.N, gpuA);
			
			// Q' x b   -> gpuB		
			log.debug( "Perform Q' x b" ) ;
			rc = CuSolver.INSTANCE.cusolverDnDormqr(
					cusolverHandle, 
					CUBLAS_SIDE_LEFT, CUBLAS_OP_T, 
					A.M, A.N, Math.min(A.M, A.N), 
					gpuA, A.M,	
					gpuT, 
					gpuB, A.M, 
					gpuW, 
					lwork, 
					gpuD
					) ; 
			checkrc( rc ) ;

//			printMatrix(A.M, B.N, gpuB);

			//--------------------------------------
			// Solve R x = Q' x b   to find x
			// R is upper triangular 
	
			log.debug( "Solve Rx = Q' x b" ) ;
			rc = CuBlas.INSTANCE.cublasDtrsm_v2(
					cublasHandle, 
					CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, 
					CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
					A.N, B.N, 
					one, 
					gpuA, A.M, 
					gpuB, B.M
					) ;
			checkrc( rc ) ;
//			printMatrix( 4, 4, gpuB ) ;
	
			log.debug( "Copying x from GPU" ); 
			x = new Matrix( A.N, B.N, B.labels ) ;
			rc = CuBlas.INSTANCE.cublasGetMatrix(x.M, x.N, DoubleSize, gpuB, B.M, x.data, x.M ) ;
			checkrc( rc ) ;
		} finally {		
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuB ) ;
			CuBlas.INSTANCE.cublasFree( gpuW ) ;
			CuBlas.INSTANCE.cublasFree( gpuD ) ;
			CuBlas.INSTANCE.cublasFree( gpuT ) ;
		}
		log.debug( "Solved x =", x ) ;
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

	public Matrix lud( Matrix A, int ipiv[] ) {
		log.debug( "Solve LU = A  {} x {} ", A.M, A.N ) ;

		Matrix X = null ;
		Pointer gpuD=null, gpuA=null, gpuI=null, gpuW=null ;
		try {
			gpuD = getMemory(1);				// device return code
			gpuA = getMemory(A.M*A.N);			// A
			gpuI = getMemory(A.M);				// IPIV
	
			log.debug( "Sending A to GPU" ) ;
			int rc = CuBlas.INSTANCE.cublasSetMatrix( A.M, A.N, DoubleSize, A.data, A.M, gpuA, A.M ) ;
			checkrc(rc) ;
	
			// workspace size
			int work[] = new int[1] ;		
	
			log.debug( "Calculating work area on GPU" ) ;
			rc = CuSolver.INSTANCE.cusolverDnDgetrf_bufferSize(
					cusolverHandle, 
					A.M, A.N, 
					gpuA, A.M, 
					work
					) ; 		
			checkrc( rc ) ;
			int lwork = work[0] ;
			gpuW = getMemory(lwork) ;
			log.debug( "Allocated double[{}] on GPU", lwork ) ;
	
			// LU ( step 1 )
			rc = CuSolver.INSTANCE.cusolverDnDgetrf( 
					cusolverHandle, 
					A.M, A.N, 
					gpuA, A.M, 
					gpuW, 
					gpuI, 
					gpuD 
					) ; 
			checkrc( rc ) ;
	
			log.debug( "Copying IPIV from GPU" ); 
			rc = CuBlas.INSTANCE.cublasGetVector(ipiv.length, IntSize, gpuI, 1, ipiv, 1) ;
			checkrc( rc ) ;

			log.debug( "Copying A from GPU" ); 
			X = new Matrix( A.M, A.N ) ;
			rc = CuBlas.INSTANCE.cublasGetMatrix(X.M, X.N, DoubleSize, gpuA, A.M, X.data, X.M ) ;			
			checkrc( rc ) ;
		} finally {		
			CuBlas.INSTANCE.cublasFree( gpuD ) ;
			CuBlas.INSTANCE.cublasFree( gpuA ) ;
			CuBlas.INSTANCE.cublasFree( gpuI ) ;
			CuBlas.INSTANCE.cublasFree( gpuW ) ;
		}
		log.debug( "Factored LU = {}", X ) ;
		return X ;
	}	

	protected void printMatrix( int M, int N, Pointer A ) {
		double a[] = new double[M*N] ;
		int rc = CuBlas.INSTANCE.cublasGetMatrix( M, N, DoubleSize, A, M, a, M ) ;
		System.out.println( ".... RC=" + rc ) ;
		
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
	