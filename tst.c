
#include <stdio.h>
#include "lapacke.h"

int main( int argc, char **argv ) {
double *A = (double*)malloc( 16 * sizeof(double) ) ;
double *B = (double*)malloc( 16 * sizeof(double) ) ;
double *tau = (double*)malloc( 16 * sizeof(double) ) ;
double *work = (double*)malloc( 1024 * sizeof(double) ) ;
int rc = LAPACKE_dormqr_work(
				LAPACK_COL_MAJOR,
				'L' , //CblasLeft,
				'T' , //CblasTrans,
				3, 4, 3, 
				A, 3	,
				tau, 
				B, 4,
				work, 1024
				) ; 
printf( "Err %d\n", rc );
free( A ) ;
free( B ) ;
free( tau ) ;
free( work ) ;
return 0 ;
}
