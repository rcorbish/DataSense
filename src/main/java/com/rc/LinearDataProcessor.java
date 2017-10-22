package com.rc;

import java.io.IOException;
import java.io.InputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LinearDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( LinearDataProcessor.class ) ;

	final static String FEATURE_LABELS[] = { "Score", "Feature", "Result" } ;

	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		Dataset dataset = Loader.load( 1000, data, options ) ;

		if( options.square ) {
			Matrix A = dataset.train.dup() ;
			A.map( (value, context, r, c) ->  value * value ) ;			
			A.prefixLabels( "sqr " ) ;
			dataset.train = dataset.train.appendColumns(A) ;

			Matrix T = dataset.test.dup() ;
			T.hmuli( T ) ;			
			T.prefixLabels( "sqr " ) ;
			dataset.test = dataset.test.appendColumns(T) ;
		}

		if( options.addOnes ) {
			Matrix A = Matrix.fill( dataset.train.M, 1,  1.0, "bias" ) ;
			dataset.train = dataset.train.appendColumns(A) ;

			Matrix T = Matrix.fill( dataset.test.M, 1,  1.0, "bias" ) ;
			dataset.test = dataset.test.appendColumns(T) ;
		}

		return dataset ;
	}

	
	public Object process( Dataset dataset ) {
		Object rc = null ;

		Matrix A  = dataset.train ;
		Matrix T  = dataset.test ;

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

		Matrix X = A.divLeft(B) ;			
		X.labels = A.labels ;
		
		Matrix Y = T.mmul(X) ;
		Y.labels = new String[] { "Predicted" } ;
		
		Matrix YE = Y.sub(YR).map( (value, context, r, c) ->  value * value  ) ;
		YE.labels = new String[] { "MSE" } ;
		rc = new Dataset( X, Y.appendColumns( YR ).appendColumns(YE) ) ;

		return rc ;
	}
}
