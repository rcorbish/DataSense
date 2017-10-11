package com.rc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;


public class Blas extends Compute {
	final static Logger log = LoggerFactory.getLogger( Blas.class ) ;
	
	public interface OpenBlas extends Library {
		OpenBlas INSTANCE = (OpenBlas)
				Native.loadLibrary((Platform.isWindows() ? "msvcrt" : "openblas"), OpenBlas.class ) ;

		void openblas_set_num_threads( int numThreads ) ;
		String openblas_get_corename() ;
		String openblas_get_config() ;
		
		int cblas_dgemm( int order, int transA, int transB , 
				int M, int N, int K, 
				double alpha, double A[], int LDA, 
				double B[], int LDB, 
				double beta, double C[], int LDC );
		void cblas_dtrsm( int Order, int side, int uplo, int transA, int diag,
				int M, int N, double alpha, 
				double A[], int lda, 
				double B[], int ldb
				) ;

	}

	public interface Lapacke extends Library {
		Lapacke INSTANCE = (Lapacke)
				Native.loadLibrary((Platform.isWindows() ? "msvcrt" : "lapacke"), Lapacke.class ) ;

		int LAPACKE_dgeqrf_work( 
				int matrix_layout, 
				int m, int n, 
				double a[],  int lda, 
				double tau[], 
				double work[], int lwork,
				int info[]
				);
		int LAPACKE_dormqr_work( 
				int matrix_layout, int side, int trans, 
				int m, int n, int k,
				double a[], int lda, 
				double tau[],
				double c[], int ldc ,
				double work[], int lwork,
				int info[]
				) ;

	}

	public final static int LAPACK_ROW_MAJOR 	= 101 ;
	public final static int LAPACK_COL_MAJOR 	= 102 ;
	public final static int CblasRowMajor	 	= 101 ;
	public final static int CblasColMajor		= 102 ;
	public final static int CblasNoTrans		= 111 ;
	public final static int CblasTrans			= 112 ; 
	public final static int CblasConjTrans		= 113 ;
	public final static int CblasConjNoTrans	= 114 ;
	public final static int CblasUpper			= 121 ;
	public final static int CblasLower			= 122 ;
	public final static int CblasNonUnit		= 131 ;
	public final static int CblasUnit			= 132 ;
	public final static int CblasLeft			= 141 ; 
	public final static int CblasRight			= 142 ;

	private final double one = 1.0  ;
	private final double zero = 0.0 ;


	public Blas() {
		this( Runtime.getRuntime().availableProcessors() ) ;
	}

	public Blas( int numThreads ) {
		log.info( "Creating BLAS instance with {} threads", numThreads ) ;
		log.info( "Openblas config: {}", getVersion() ) ;
		OpenBlas.INSTANCE.openblas_set_num_threads(numThreads);
	}

	@Override
	public String getVersion() {
		return OpenBlas.INSTANCE.openblas_get_config()  ;
	}
	
	@Override
	public double[] mmul( int rows, int cols, double A[], double B[] ) {

		int M = rows ;
		int K = A.length / M ;
		int N = cols ;
		double C[] = new double[M*N] ;

		log.info( "mpy {} x {}  *  {} x {}", M, K, K, N ) ;

		//		--------------
		//		 A [M x K]
		//		 B [K x N]
		//		 C [M x N]
		//		--------------
		int rc = OpenBlas.INSTANCE.cblas_dgemm( 
				CblasColMajor,
				CblasNoTrans, CblasNoTrans,
				M,N,K, 
				one, A, M, 
				B, K,
				zero, C, M
				) ;
		
		checkrc( rc ) ;
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
	public double[] solve( int rows, int cols, double A[], double B[], int numFeatures ) {
		int M = rows ;
		int N = cols ;

		log.info( "Solve Ax=b  {} x {} ", M, N ) ;

		if( M<N) throw ( new RuntimeException( "M must be >= N" ) ) ;

		int devinfo[] = new int[1] ;
		double work[] = new double[1] ;
		double tau[] = new double[ Math.min(M, N) ] ;
		int rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				M, N, 
				A, M,
				tau,
				work,
				-1,
				devinfo ) ;
		checkrc( rc ) ;

		int lwork = (int)work[0] ;
		log.info( "Allocated double[{}] for work area", lwork ) ;

		work = new double[lwork] ;
		rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				M, N, 
				A, M,
				tau,
				work,
				lwork,
				devinfo) ;
		checkrc( rc ) ;
		//		printMatrix(1,  N, tau);
		log.info( "Factored  QR = A" ) ;

		// Q' x b   -> B		
		rc = Lapacke.INSTANCE.LAPACKE_dormqr_work(
				LAPACK_COL_MAJOR,
				'L' , //CblasLeft,
				'T' , //CblasTrans,
				M, numFeatures, Math.min(M,N), 
				A, M,	
				tau, 
				B, M,
				work, lwork,
				devinfo
				) ; 
		checkrc( rc ) ;
		//		printMatrix(M, 1, B);

		log.info( "Created Q'b = Rx" ) ;

		//--------------------------------------
		// Solve R = Q' x b   
		// R is upper triangular 
		OpenBlas.INSTANCE.cblas_dtrsm(
				CblasColMajor,
				CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				N, numFeatures, 
				one, 
				A, M, 
				B, M
				) ;
//		printMatrix( M, numFeatures, B ) ;

		log.info( "Solved x" ) ;

		double x[] = new double[N*numFeatures] ;
		for( int f=0 ; f<numFeatures ; f++ ) {
			for( int i=0 ; i<N ; i++ ) {
				int xx = i + f*N ;
				int bx = i + f*M ; 
				x[xx] = B[bx] ;
			}
		}
		return x ;
	}



	//
	// xA = B
	//
	// This is eaasy if we can switch row/col ordering of matrices
	//
	public Matrix solve2( Matrix A, Matrix B, int numFeatures ) {

		
		log.info( "Solve xA=B  {} x {}  / {} x {} ", A.M, A.N, B.M, B.N) ;

		
//		if( M<N) throw ( new RuntimeException( "M must be >= N" ) ) ;

		int devinfo[] = new int[1] ;
		double work[] = new double[1] ;
		int tauSize = 4 ; //Math.min( A.N, A.M )  ;
		
		
		double tau[] = new double[ tauSize ] ;
		int rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				A.M, A.N, 
				A.data, A.M,
				tau,
				work,
				-1,
				devinfo ) ;
		checkrc( rc ) ;

		int lwork = (int)work[0] ;
		lwork=1000 ;
		log.info( "Allocated double[{}] for work area", lwork ) ;

		work = new double[lwork] ;
		rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				A.M, A.N, 
				A.data, A.M,
				tau,
				work,
				lwork,
				devinfo) ;
		checkrc( rc ) ;
		//		printMatrix(1,  N, tau);
		System.out.println( A ) ;
		log.info( "Factored  QR = A'" ) ;

		// Q' x b   -> B		
		rc = Lapacke.INSTANCE.LAPACKE_dormqr_work(
				LAPACK_ROW_MAJOR,
				'L' , //CblasLeft,
				'T' , //CblasTrans,
				B.M, B.N, tauSize, 
				A.data, A.M,	
				tau, 
				B.data, B.M,
				work, lwork,
				devinfo
				) ; 
		checkrc( rc ) ;
		//		printMatrix(M, 1, B);

		log.info( "Created Q'b = Rx'" ) ;

		//--------------------------------------
		// Solve R = Q' x b   
		// R is upper triangular 
		OpenBlas.INSTANCE.cblas_dtrsm(
				CblasRowMajor,
				CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				A.N, numFeatures, 
				one, 
				A.data, A.M, 
				B.data, A.M
				) ;
//		printMatrix( M, numFeatures, B ) ;

		log.info( "Solved x" ) ;

		double x[] = new double[A.N*numFeatures] ;
		for( int f=0 ; f<numFeatures ; f++ ) {
			for( int i=0 ; i<A.N ; i++ ) {
				int xx = i + f*A.N ;
				int bx = i + f*A.M ; 
				x[xx] = B.data[bx] ;
			}
		}
		return new Matrix( numFeatures, A.N, x )  ;
	}

	@Override
	public void close() {
		log.info( "Shutdown openblas" ) ;
	}

	protected void printMatrix( int M, int N, double A[] ) {
		System.out.println( " ==== "  ) ;

		for( int i=0 ; i<Math.min( 8, M) ; i++ ) {
			for( int j=0 ; j<Math.min( 8, N) ; j++ ) {
				int ix = i + j*M ;
				System.out.print( String.format( "%10.3f", A[ix] ) );
			}
			System.out.println(); 
		}

		System.out.println( " ====" ) ;
	}

	protected void checkrc( int rc ) {
		if( rc == 0 ) return ;
		StackTraceElement ste = Thread.currentThread().getStackTrace()[2] ;
		log.error( "Error {} at: {}",  rc, ste ) ;
		throw new RuntimeException( "Failed to check RC" ) ;
	}

}

