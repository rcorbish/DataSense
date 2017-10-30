package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CentroidDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( CentroidDataProcessor.class ) ;

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

		Matrix A  = dataset.train ;
		Matrix T  = dataset.test ;

		// value in dataset -> zero based index
		Map<Integer,Integer> featureKeys = dataset.getFeatureKeys() ;
		Map<Integer,Integer> inverseFeatureKeys = new HashMap<>() ;
		for( Entry<Integer, Integer> e :featureKeys.entrySet() ) {
			inverseFeatureKeys.put( e.getValue(), e.getKey() ) ;
		}
		int feature = dataset.getFeatureColumnIndex() ;

		Matrix F = A.extractColumns( feature ) ;
		Matrix YR = T.extractColumns( feature ) ; 

		
		int numBuckets = featureKeys.size() ;
		
		Matrix centroids = Matrix.fill( numBuckets, A.N, 0.0 ) ;
		Matrix counts    = Matrix.fill( numBuckets, A.N, 0.0 ) ;
		Matrix means     = A.mean() ;
		Matrix stddevs   = A.stddev(means) ;
		Matrix AN = A.normalizeColumns() ;
		for( int i=0 ; i<AN.M ; i++ ) {
			int f = featureKeys.get( (int)F.get(i) ) ;
			for( int j=0 ; j<AN.N ; j++ ) {
				centroids.put( f, j, centroids.get( f, j ) + AN.get(i,j) )  ;
				counts.add( f,j, 1 ) ;
			}
		}
		centroids.hdivi( counts ) ;
		Matrix Y = new Matrix( T.M, 1 ) ;
		
		//Matrix m = T.sub( centroids ) ;
		for( int i=0 ; i<T.M ; i++ ) {
			Matrix row = T.extractRows( 0 ) ;
			row.subi( means ) ;
			row.hdivi( stddevs ) ;
			Matrix D = centroids.sub( row ) ;
			D.hmuli( D ) ;
			Matrix euclideanDistanceToCentroid = D.transpose().sum().map( v -> Math.sqrt(v) ) ;
			Y.put( i, 0, euclideanDistanceToCentroid.minIndexOfRows().get(0) ) ;			
		}
		
		CentroidResults rc = new CentroidResults() ;

		Matrix precision = new Matrix( numBuckets, 1 ) ;
		Matrix recall = new Matrix( numBuckets, 1 ) ;
		
		for( int n=0 ; n<numBuckets ; n++ ) {
			int f = inverseFeatureKeys.get( n ) ;

			int nfp = 0 ;
			int ntp = 0 ;
			int nfn  = 0 ;
			for( int i=0 ; i<Y.length() ; i++ ) {
				int yn = (int)Y.get(i) ;
				int yrn = (int)YR.get(i) ;
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
		return rc ;

	}
}

class CentroidResults {
	Matrix precision ;
	Matrix recall ;
	Matrix f1 ;
}
