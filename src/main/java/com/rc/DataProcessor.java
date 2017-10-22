package com.rc;

import java.io.IOException;
import java.io.InputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataProcessor {
	final static Logger log = LoggerFactory.getLogger( DataProcessor.class ) ;

	final static String FEATURE_LABELS[] = { "Score", "Feature", "Result" } ;
	public Object process( InputStream data, ProcessorOptions options ) {
		Object rc = null ;
		
		try {
			Dataset dataset = Loader.load( 1000, data, options ) ;
			Matrix A  = dataset.train ;
			Matrix T  = dataset.test ;
			
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
				Matrix YR = T.extractColumns( feature ) ; 
				
				log.debug( "features {}", B ) ;		
				
				if( options.square ) {
					Matrix A2 = A.dup() ;
					A2.map( (value, context, r, c) ->  value * value ) ;			
					A2.prefixLabels( "sqr " ) ;
					A = A.appendColumns(A2) ;

					Matrix T2 = T.dup() ;
					T2.map( (value, context, r, c) ->  value * value ) ;			
					T2.prefixLabels( "sqr " ) ;
					T = T.appendColumns(T2) ;
				}
	
				if( options.addOnes ) {
					Matrix A2 = Matrix.fill( A.M, 1,  1.0, "bias" ) ;
					A = A.appendColumns(A2) ;

					Matrix T2 = Matrix.fill( T.M, 1,  1.0, "bias" ) ;
					T = T.appendColumns(T2) ;
				}
				
				Matrix X = A.divLeft(B) ;			
				X.labels = A.labels ;
				Matrix Y = T.mmul(X) ;
				Y.labels = new String[] { "Predicted" } ;
				Matrix YE = Y.sub(YR).map( (value, context, r, c) ->  value * value  ) ;
				YE.labels = new String[] { "MSE" } ;
				rc = new Dataset( X, Y.appendColumns( YR ).appendColumns(YE) ) ;
				
			} else if( "correlation".equals( options.method ) ) {
				// 1st covariance
				Matrix A4 = A.zeroMeanColumns() ;
				Matrix CO = A4.transpose().mmul( A4 ) ;			
				CO.labels = A.labels ;
				CO.muli( 1.0 / ( A.M - 1 ) ) ;

				// Then correlation
				Matrix MX = A.mean() ;
				Matrix AS = A.stddev(MX) ;
				
				for( int i=0 ; i<A.N ; i++ ) {
					int ix = i*CO.M ;
					for( int j=0 ; j<A.N ; j++ ) {
						CO.data[ix] /= ( AS.data[i] * AS.data[j] ) ;
						ix++ ;
					}
				}
				
				rc = CO ;
			} else if( "statistics".equals( options.method ) ) {
				Matrix AN = A.min() ;
				Matrix AX = A.max() ;
				Matrix AC = A.countBuckets( 1e-4 ) ;
				Matrix AM = A.mean() ;
				Matrix AD = A.stddev( AM ) ;
				Matrix AS = A.skewness( AM ) ;
				Matrix AK = A.kurtosis( AM ) ;
				Matrix X = AN
					.appendRows( AX )
					.appendRows( AM )
					.appendRows( AD )
					.appendRows( AS )
					.appendRows( AK )
					.appendRows( AC )
					;
				rc = X ;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return rc ;
	}
}
