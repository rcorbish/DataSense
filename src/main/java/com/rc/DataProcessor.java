package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

abstract public class DataProcessor {
	final static Logger log = LoggerFactory.getLogger( DataProcessor.class ) ;
	final static public int ROWS_TO_KEEP = 5_000 ;
	
	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		options.testRatio = 0 ;
		Dataset dataset = Loader.load( ROWS_TO_KEEP, data, options ) ;
		return dataset ;
	}
	
	abstract public Object process( Dataset dataset ) ;

	public Object process( InputStream data, ProcessorOptions options ) {
		Object rc = null ;
		
		try {
			Dataset dataset = load( data, options ) ;	
			rc = process( dataset ) ;
		} catch (IOException e) {
			e.printStackTrace();
			rc = e.getMessage() ;
		}
		return rc ;
	}

	public ProcessScores score( Matrix Y, Matrix YH ) {
		return score( Y, YH, null ) ;
	}
	public ProcessScores score( Matrix Y, Matrix YH, Map<Integer,Integer> inverseFeatureKeys ) {
		
		ProcessScores rc = new ProcessScores() ;

		// Logistic
		if( inverseFeatureKeys != null ) {
			int numBuckets = (int)YH.dup().map( v -> Math.round(v) ).countBuckets(0.5).get(0) ;
			Matrix precision = new Matrix( numBuckets, 1 ) ;
			Matrix recall = new Matrix( numBuckets, 1 ) ;
			
			for( int n=0 ; n<numBuckets ; n++ ) {
				int f = inverseFeatureKeys.get( n ) ;

				int nfp = 0 ;
				int ntp = 0 ;
				int nfn  = 0 ;
				for( int i=0 ; i<YH.length() ; i++ ) {
					int yn = (int)Math.round( YH.get(i) ) ;
					int yrn = (int)Math.round( Y.get(i) ) ;
					if( yrn == f && yn == f ) {	// true positives
						ntp++ ;
					} else if( yrn != f && yn == f ) {	// false positives
						nfp++ ;
					} else if( yrn == f && yn != f ) {	// false negatives
						nfn++ ;
					}
				}
				precision.put( n, (double)ntp / (double)(ntp + nfp) ) ;
				recall.put( n, (double)ntp / (double)(ntp + nfn) ) ;
			}
			rc.precision = precision ;
			rc.recall = recall ;
			rc.f1 = precision.hmul( recall ).muli(2.0).hdivi( precision.add( recall ) ) ;
		}
		// Linear 
		double accuracy = 0.0 ;
		double harmonicError = 0.0 ;
		double correct = 0.0 ;
		
		for( int i=0 ; i<Y.length() ; i++ ) {
			double err = Y.get(i) - YH.get(i) ;
			accuracy += err * err ;
			harmonicError += 1.0 / ( 1.0 + err ) ;
			if( Math.abs(err)<0.5 ) correct++ ;
		}

		rc.accuracy = 1.0 - Math.sqrt( accuracy ) / Y.length() ;
		rc.correctRatio = correct / Y.length() ;
		rc.harmonic = Y.length() / harmonicError ;
		return rc ;
	}

}

class ProcessScores {
	Matrix precision ;
	Matrix recall ;
	Matrix f1 ;

	double accuracy ;
	double harmonic ;
	double correctRatio ;
}


