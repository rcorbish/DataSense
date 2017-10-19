package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataProcessor {
	final static Logger log = LoggerFactory.getLogger( DataProcessor.class ) ;

	final static String FEATURE_LABELS[] = { "Score", "Feature", "Result" } ;
	public Object process( InputStream data, ProcessorOptions options ) {
		Object rc = null ;
		
		try {
			Matrix A = Loader.load( 1000, data, options ) ;
			A.name = "A" ;

			rc = "Unsupported method" ; 
			if( "linear".equals( options.method ) ) {
				int feature = 0 ;
				
				featureSearch:
				for( int i=0 ; i<FEATURE_LABELS.length ; i++ ) {
					for( int j=0 ; j<A.N; j++ ) {
						if( FEATURE_LABELS[i].equalsIgnoreCase( A.labels[j] ) ) {
							feature = j ;
							break featureSearch ;
						}
					}
				}
				Matrix B = A.extractColumns( feature ) ;
				B.name = "B" ;
				log.debug( "features {}", B ) ;		
				
				if( options.square ) {
					Matrix A2 = A.dup() ;
					A2.map( (value, context, r, c) ->  value * value ) ;			
					A2.prefixLabels( "sqr " ) ;
					A = A.appendColumns(A2) ;
				}
	
				if( options.addOnes ) {
					Matrix A2 = Matrix.fill( A.M, 1,  1.0, "bias" ) ;
					A = A.appendColumns(A2) ;
				}
				Matrix X = A.divLeft(B) ;			
				X.labels = A.labels ;
				rc = X.transpose() ;
			} else if( "covariance".equals( options.method ) ) {
				Matrix A4 = A.zeroMeanColumns() ;
				Matrix X = A4.transpose().mmul( A4 ) ;			
				X.labels = A.labels ;
				rc = X.muli( 1.0 / X.N ) ;
				log.debug( "Cov(A) =\n{}", rc ) ;
			} else if( "statistics".equals( options.method ) ) {
				Matrix AM = A.mean() ;
				Matrix AS = A.skewness( AM ) ;
				Matrix AK = A.kurtosis( AM ) ;
				Matrix X = AM.appendRows(AS).appendRows(AK);
				rc = X ;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return rc ;
	}
}
