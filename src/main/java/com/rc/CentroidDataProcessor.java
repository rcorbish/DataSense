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
			
		if( options.normalize ) {
			dataset.normalize();
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
		
		return score( YR, Y, inverseFeatureKeys )  ;
	}
}

