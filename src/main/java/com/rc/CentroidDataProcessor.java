package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CentroidDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( CentroidDataProcessor.class ) ;

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
		Map<Integer,Integer> featureKeys = new HashMap<>() ;
		for( int i=0 ; i<F.length() ; i++ ) {
			int f = (int)F.get(i) ;
			Integer n = featureKeys.get( f ) ;
			if( n == null ) {
				featureKeys.put( featureKeys.size(), f ) ;
			}
		}
		Matrix Y = T.extractColumns( feature ) ;
		
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
		Matrix YR = new Matrix( T.length(), 1 ) ;
		
		//Matrix m = T.sub( centroids ) ;
		for( int i=0 ; i<T.M ; i++ ) {
			Matrix row = T.extractRows( 0 ) ;
			row.subi( means ) ;
			row.hdivi( stddevs ) ;
			Matrix D = centroids.sub( row ) ;
			D.hmuli( D ) ;
			Matrix euclideanDistanceToCentroid = D.transpose().sum().map( v -> Math.sqrt(v) ) ;
			YR.put( i, 0, euclideanDistanceToCentroid.minIndexOfRows().get(0) ) ;			
		}
		
		CentroidResults rc = new CentroidResults() ;

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

class CentroidResults {
	Matrix precision ;
	Matrix recall ;
	Matrix f1 ;
}
