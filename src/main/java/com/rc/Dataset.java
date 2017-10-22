
package com.rc ;

public class Dataset {
    Matrix train ;
    Matrix test ;
    
    public Dataset( Matrix A, Matrix B ) {
    	train = A ;
    	train.name = "Train" ;
    	test = B ;
    	test.name = "Test" ;
    }
}