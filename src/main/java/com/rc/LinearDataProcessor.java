package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

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

		// Prefix with a bias column
		Matrix A = Matrix.fill( dataset.train.M, 1,  1.0, "bias" ) ;
		dataset.train = A.appendColumns( dataset.train ) ;

		Matrix T = Matrix.fill( dataset.test.M, 1,  1.0, "bias" ) ;
		dataset.test = T.appendColumns(dataset.test) ;

		return dataset ;
	}

	
	public Object process( Dataset dataset ) {

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
		Matrix F = A.extractColumns( feature ) ;
		Matrix YR = T.extractColumns( feature ) ; 

		Map<Integer,Integer> featureKeys = new HashMap<>() ;
		for( int i=0 ; i<F.length() ; i++ ) {
			int f = (int)F.get(i) ;
			Integer n = featureKeys.get( f ) ;
			if( n == null ) {
				featureKeys.put( featureKeys.size(), f ) ;
			}
		}
		
		int numBuckets = featureKeys.size() ; // (int) Math.floor( F.countBuckets( 1 ).get( 0 ) ) ;

		
		Matrix X = A.divLeft(F) ;			
		X.labels = A.labels ;
		
		Matrix Y = T.mmul(X) ;
		Y.map( v -> Math.round(v) ) ;
		Y.labels = new String[] { "Predicted" } ;
		
		LinearResults rc = new LinearResults() ;

		Matrix precision = new Matrix( numBuckets, 1 ) ;
		Matrix recall = new Matrix( numBuckets, 1 ) ;
		
		for( int n=0 ; n<numBuckets ; n++ ) {
			int fn = featureKeys.get(n) ;
			
			int nfp = 0 ;
			int ntp = 0 ;
			int nfn  = 0 ;
			for( int i=0 ; i<Y.length() ; i++ ) {
				
				int yn = featureKeys.get( (int)Y.get(i) ) ;
				int yrn = featureKeys.get( (int)YR.get(i) ) ;
				
				if( yrn == fn && yn == fn ) {	// true positives
					ntp++ ;
				}
				if( yrn != fn && yn == fn ) {	// false positives
					nfp++ ;
				}

				if( yrn == fn && yn != fn ) {	// false negatives
					nfn++ ;
				}
				
			}
			precision.put( n, (double)ntp / (double)(ntp + nfp) ) ;
			recall.put( n, (double)ntp / (double)(ntp + nfn) ) ;
		}
		rc.precision = precision ;
		rc.recall = recall ;
		rc.f1 = precision.hmul( recall ).muli(2.0).hdivi( precision.add( recall ) ) ;
		return rc ;
	}
}

class LinearResults {
	Matrix precision ;
	Matrix recall ;
	Matrix f1 ;
}
