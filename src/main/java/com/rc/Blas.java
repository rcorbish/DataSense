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
				Native.loadLibrary((Platform.isWindows() ? "libopenblas" : "openblas"), OpenBlas.class ) ;

		void openblas_set_num_threads( int numThreads ) ;
		String openblas_get_corename() ;
		String openblas_get_config() ;

		double cblas_ddot( int N, 
				double A[], int LDA, 
				double B[], int LDB
				);
		
		double cblas_dnrm2( int N, 
				double A[], int LDA
				);

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
				Native.loadLibrary((Platform.isWindows() ? "liblapacke" : "lapacke"), Lapacke.class ) ;

		int LAPACKE_dgeqrf_work( 
				int matrix_layout, 
				int m, int n, 
				double a[],  int lda, 
				double tau[], 
				double work[], int lwork
				);
		int LAPACKE_dormqr_work( 
				int matrix_layout, int side, int trans, 
				int m, int n, int k,
				double a[], int lda, 
				double tau[],
				double c[], int ldc ,
				double work[], int lwork
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
	public double dot( Matrix A, Matrix B ) {
		if( !A.isVector || !B.isVector ) throw new RuntimeException( String.format( "Dot product requires vectors" ) )  ;
		if( A.length() != B.length() ) throw new RuntimeException( String.format( "Incompatible matrix sizes: %d  and %d", A.length(), B.length() ) )  ;

		return  OpenBlas.INSTANCE.cblas_ddot( 
				A.length(), 
				A.data, 1, 
				B.data, 1
				) ;
	}

	@Override
	public double norm( Matrix A ) {
		if( !A.isVector ) throw new RuntimeException( String.format( "Norm requires vectors" ) )  ;

		return  OpenBlas.INSTANCE.cblas_dnrm2( 
				A.length(), 
				A.data, 1
				) ;
	}

	
	@Override
	public Matrix mmul( Matrix A, Matrix B ) {

		Matrix C = new Matrix( A.M, B.N, B.labels ) ;

		if( A.N != B.M ) throw new RuntimeException( String.format( "Incompatible matrix sizes: %d x %d  and %d x %d", A.M, A.N, B.M, B.N ) )  ;
		//		--------------
		//		 A [M x K]
		//		 B [K x N]
		//		 C [M x N]
		//		--------------
		int rc = OpenBlas.INSTANCE.cblas_dgemm( 
				CblasColMajor,
				CblasNoTrans, CblasNoTrans,
				A.M,B.N,B.M, 
				one, 
				A.data, A.M, 
				B.data, B.M,
				zero, 
				C.data, C.M
				) ;
		
		checkrc( rc ) ;
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

		log.debug( "Solve Ax=b  {} x {} ", A.M, A.N ) ;

		if( A.M<A.N) {
			throw ( new RuntimeException( "M must be >= N" ) ) ;
		}
		//if( A.M!=B.M) throw ( new RuntimeException( "M must be the same" ) ) ;

		double work[] = new double[1] ;
		double tau[] = new double[ Math.min(A.M, A.N) ] ;
		int rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				A.M, A.N, 
				A.data, A.M,
				tau,
				work,
				-1 ) ;
		checkrc( rc ) ;

		int lwork = (int)work[0] ;
		log.debug( "Allocated double[{}] for work area", lwork ) ;

		work = new double[lwork] ;
		rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				A.M, A.N, 
				A.data, A.M,
				tau,
				work,
				lwork) ;
		checkrc( rc ) ;
		//		printMatrix(1,  N, tau);
		log.debug( "Factored  QR = A" ) ;

		// Q' x b   -> B		
		rc = Lapacke.INSTANCE.LAPACKE_dormqr_work(
				LAPACK_COL_MAJOR,
				'L' , //CblasLeft,
				'T' , //CblasTrans,
				A.M, B.N, Math.min(A.M,A.N), 
				A.data, A.M,	
				tau, 
				B.data, A.M,
				work, lwork
				) ; 
		checkrc( rc ) ;
		//		printMatrix(M, 1, B);

		log.debug( "Created Q'b = Rx" ) ;

		//--------------------------------------
		// Solve R = Q' x b   
		// R is upper triangular 
		OpenBlas.INSTANCE.cblas_dtrsm(
				CblasColMajor,
				CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				A.N, B.N, 
				one, 
				A.data, A.M, 
				B.data, B.M
				) ;
//		printMatrix( M, numFeatures, B ) ;

		Matrix x = new Matrix( A.N, B.N, B.labels ) ;
		for( int f=0 ; f<B.N ; f++ ) {
			for( int i=0 ; i<A.N ; i++ ) {
				int xx = i + f*A.N ;
				int bx = i + f*B.M ; 
				x.data[xx] = B.data[bx] ;
			}
		}
		log.debug( "Solved x = ", x ) ;
		return x ;
	}



	//
	// xA = B
	//
	// This is easy if we can switch row/col ordering of matrices
	//
	public Matrix solve2( Matrix A, Matrix B ) {

		
		log.info( "Solve xA=B  {} x {}  / {} x {} ", A.M, A.N, B.M, B.N) ;
		if( A.N != B.N ) {
			throw new RuntimeException( "Incompatible sizes - columns must match" ) ;
		}
		if( A.M > A.N ) {
			int missingCols = A.M - A.N ;
			Matrix Z = Matrix.fill(A.M, missingCols, 0.0) ;
			A = A.appendColumns( Z ) ;
			//throw new RuntimeException( "Incompatible input - M must be <= N" ) ;
			
		}
		//
		// IN
		// 	A is A.M x A.N  
		// 	B is B.M x B.N
		//
		// OUT
		// 	Q is A.M x A.M
		// 	R is A.M x A.N
		// 	B is B.N x A.N
		//  Q'B' is A.M x B.N
		//  X is B.N x A.M
		//

		// In place - destroys inputs!
		A = A.transpose() ;
		B = B.transpose() ;
		
		double work[] = new double[1] ;
		int tauSize =Math.min( A.M, A.N )  ;
		
		double tau[] = new double[ tauSize ] ;
		int rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				A.M, A.N, 
				A.data, A.M,
				tau,
				work,
				-1 ) ;
		checkrc( rc ) ;

		int lwork = (int)work[0] ;
		log.debug( "Allocated double[{}] for work area", lwork ) ;

		work = new double[lwork] ;
		rc = Lapacke.INSTANCE.LAPACKE_dgeqrf_work(
				CblasColMajor,
				A.M, A.N, 
				A.data, A.M,
				tau,
				work,
				lwork ) ;
		checkrc( rc ) ;
		log.debug( "Factored  QR = A' ... \n{}", A ) ;

		// Q' x b   -> B		
		rc = Lapacke.INSTANCE.LAPACKE_dormqr_work(
				LAPACK_COL_MAJOR,
				'L' , //CblasLeft,
				'T' , //CblasTrans,
				B.M, B.N, tauSize, 
				A.data, A.M,	
				tau, 
				B.data, B.M,
				work, lwork
				) ; 
		checkrc( rc ) ;
		
		log.debug( "Created Q'b' = Rx' ... \n{}",B ) ;

		//--------------------------------------
		// Solve R X = Q' x b'
		//       
		//      A is R		( result of QR )  A.M x A.N 
		//		B is Q'B	( Q * original input ) A.M * numFeatures
		OpenBlas.INSTANCE.cblas_dtrsm(
				CblasColMajor,
				CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				A.N, A.M, 
				one, 
				A.data, A.M,    
				B.data, A.M		
				) ;

		B = B.transpose() ;
		//B.M = A.M ;
		B.N = A.N ;
		log.debug( "Solved x' ...\n{}", B  ) ;

		// NB this is transpose copy
		// double x[] = new double[B.N*A.N] ;
		// for( int f=0 ; f<B.N ; f++ ) {
		// 	for( int i=0 ; i<B.N ; i++ ) {
		// 		int xx = i + f*B.N ;
		// 		int bx = i + f*B.N; 
		// 		double d = B.data[bx] ;
		// 		x[xx] = d ;
		// 	}
		// }
		return B ;
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

