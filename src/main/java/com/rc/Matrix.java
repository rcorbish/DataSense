
package com.rc ;

public class Matrix {
    int M ;     // rows
    int N ;     // cols

    double data[] ;

    private boolean isColumnMajor ;
    private boolean isVector ;

    public Matrix( int rows, int columns ) {
        this( rows, columns, new double[ rows * columns ] ) ;
    }

    public Matrix( int rows, int columns, double data[] ) {
        this.M = rows ;
        this.N = columns ;
        this.data = data ;
        isColumnMajor = true ;
        isVector = rows==1 || columns==1 ;
    }

    public Matrix dup() {
        double copy[] = new double[data.length] ;
        System.arraycopy( data, 0, copy, 0, copy.length ) ;
        Matrix rc = isColumnMajor ?
            new Matrix( M, N, copy ) :
            new Matrix( N, M, copy ) ;

        return rc ;
    }

    public void transpose() {
        int tmp = this.M ;
        this.M = this.N ;
        this.N = tmp ;
        isColumnMajor = !isColumnMajor ;
    }

    protected int index( int r, int c ) {
        return isColumnMajor ? (r + c*M) : (c + r*N) ;
    }

    public String toString() {
        StringBuilder rc = new StringBuilder() ;

        for( int i=0 ; i<Math.min( 8, M) ; i++ ) {
            for( int j=0 ; j<Math.min( 8, N) ; j++ ) {
                rc.append( String.format( "%10.3f", data[index(i,j)] ) );
            }
            rc.append( '\n' ); 
        }
        return rc.toString() ;
    }
}