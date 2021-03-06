package com.rc;

import java.io.IOException;
import java.io.InputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CorrelationDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( CorrelationDataProcessor.class ) ;

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
				
		return dataset ;
	}

	public Object process( Dataset dataset ) {
		
		int feature = dataset.getFeatureColumnIndex() ;
		dataset.train.swapColumns(0, feature ) ;
		
		Matrix A = dataset.train.zeroMeanColumns() ;
		Matrix CO = A.transpose().mmul( A ) ;			
		CO.labels = dataset.train.labels ;
		CO.muli( 1.0 / ( dataset.train.M - 1 ) ) ;

		// Then correlation
		Matrix MX = dataset.train.mean() ;
		Matrix AS = dataset.train.stddev(MX) ;
		
		for( int i=0 ; i<dataset.train.N ; i++ ) {
			int ix = i*CO.M ;
			for( int j=0 ; j<dataset.train.N ; j++ ) {
				CO.data[ix] /= ( AS.data[i] * AS.data[j] ) ;
				ix++ ;
			}
		}
		CorrelationResults rc = new CorrelationResults() ;
		rc.labels = CO.labels ;
		rc.R = new double[CO.N][] ;
		
		for( int j=0 ; j<CO.N ; j++ ) {
			rc.R[j] = new double[CO.M] ;
			System.arraycopy(CO.data, j*CO.M, rc.R[j], 0, CO.M );
		}
		
		return rc ;
	}
}


class CorrelationResults {
	String labels[] ;
	double R[][] ;
}