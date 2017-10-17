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
			Matrix A = Loader.load( 1000, data, options.cs ) ;
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

			Matrix A3 ;
			
			if( options.square ) {
				Matrix A2 = A.dup() ;
				A2.map( (value, context, r, c) ->  value * value )  ;			
				A3 = A.appendColumns(A2) ;
			} else {
				A3 = A ;
			}

			if( options.addOnes ) {
				Matrix A2 = Matrix.fill( A.M, 1,  1.0, "bias" ) ;
				A3 = A3.appendColumns(A2) ;
			} else {
				A3 = A ;
			}
			
			log.debug( "data {}", A ) ;
			log.debug( "features {}", B ) ;		

			rc = "Unsupported method" ; 
			if( "linear".equals( options.method ) ) {
				Matrix X = A3.divLeft(B) ;			
				X.labels = A3.labels ;
				rc = X.transpose() ;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return rc ;
	}
}

class ProcessorOptions {
	Charset cs ;
	boolean square ;
	boolean discrete ;
	boolean addOnes;
	String  method ;
}
