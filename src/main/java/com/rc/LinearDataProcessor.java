package com.rc;

import java.io.IOException;
import java.io.InputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LinearDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( LinearDataProcessor.class ) ;

	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		Dataset dataset = Loader.load( DataProcessor.ROWS_TO_KEEP, data, options ) ;

		if( options.square ) {
			dataset.square( options.keepOriginal );
		}		
		
		if( options.log ) {
			dataset.log( options.keepOriginal );
		}
		
		if( options.reciprocal ) {
			dataset.reciprocal( options.keepOriginal );
		}

		if( options.normalize ) {
			dataset.normalize();
		}
		
		dataset.addBias() ;
		return dataset ;
	}

	
	public Object process( Dataset dataset ) {

		Matrix A  = dataset.train ;
		Matrix T  = dataset.test ;

		int feature = dataset.getFeatureColumnIndex() ;

		Matrix F = A.extractColumns( feature ) ;
		Matrix YR = T.extractColumns( feature ) ; 

		Matrix X = A.divLeft(F) ;			
		
		Matrix Y = T.mmul(X) ;
		//log.info( "Calculated Y={}", Y ) ;
		
		return score( YR, Y ) ;		
	}
}
