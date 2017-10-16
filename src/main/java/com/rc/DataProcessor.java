package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataProcessor {
	final static Logger log = LoggerFactory.getLogger( DataProcessor.class ) ;

	final static String FEATURE_LABELS[] = { "Score", "Feature", "Result" } ;
	public Object process( InputStream data, Charset cs ) {
		Matrix rc = null ;
		try {
			Matrix A = Loader.load( 1000, data, cs ) ;
			A.name = "A" ;

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

			Matrix A2 = A.dup() ;
			Matrix Y = B.dup() ;

//			log.info( "data {}", A ) ;
//			log.info( "features {}", B ) ;
			
//			Matrix m2 = A.map( (value, context, r, c) ->  value * value )  ;			
//			Matrix m3 = A.appendColumns(m2) ;
			
			Matrix X = A.divLeft(B) ;
			
			rc = A2.mmul( X ).appendColumns(Y) ;
			X.labels = A.labels ;
			rc = X.transpose() ;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return rc ;
	}
}
