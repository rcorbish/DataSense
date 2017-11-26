
package com.rc ;

import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Dataset {
	public final static String FEATURE_LABELS[] = { "Score", "Feature", "Result" } ;
	public final static String BIAS = "Bias" ;
	final static Logger log = LoggerFactory.getLogger( Dataset.class ) ;
	
	Matrix train ;
	Matrix test ;

	public Dataset( Matrix A, Matrix B ) {
		train = A ;
		train.name = "Train" ;
		test = B ;
		test.name = "Test" ;
	}

	public void addBias() {
		// Prefix with a bias column
		Matrix A = Matrix.fill( train.M, 1,  1.0, Dataset.BIAS ) ;
		train = A.appendColumns( train ) ;

		Matrix T = Matrix.fill( test.M, 1,  1.0, Dataset.BIAS ) ;
		test = T.appendColumns(test) ;

	}
	
	public int getFeatureColumnIndex() {
		int feature = 0 ;   // don't cache - sometimes columns move !

		featureSearch:
			for( int i=0 ; i<FEATURE_LABELS.length ; i++ ) {
				for( int j=0 ; j<train.N; j++ ) {
					if( FEATURE_LABELS[i].equalsIgnoreCase( train.labels[j] ) && !BIAS.equalsIgnoreCase(train.labels[j]) ) {
						feature = j ;
						break featureSearch ;
					}
				}
			}

		return feature ;
	}
	
	public Map<Integer,Integer> getFeatureKeys() {
		int feature = getFeatureColumnIndex() ;
		Matrix F = train.copyColumns( feature ) ;
		Map<Integer,Integer> featureKeys = new HashMap<>() ;
		for( int i=0 ; i<F.length() ; i++ ) {
			int f = (int)F.get(i) ;
			Integer n = featureKeys.get( f ) ;
			if( n == null ) {
				featureKeys.put( f, featureKeys.size() ) ;
			}
		}
		return featureKeys ;
	}

	public void normalize() {
		int feature = getFeatureColumnIndex() ;
		Matrix M = train.mean() ;
		// log.info( "Mean : {}", M ) ;
		Matrix S = train.stddev(M) ;
		M.put( feature, 0 ) ;
		S.put( feature, 1.0 ) ;
		
		train.subi(M).hdivi(S) ;
		test.subi(M).hdivi(S) ;
//		train.prefixLabels( "nrm " ) ;
//		test.prefixLabels( "nrm " ) ;
	}		

	public void square( boolean keepOriginals ) {
		int feature = getFeatureColumnIndex() ;
		Matrix A = train.dup() ;
		Matrix F = A.extractColumns(feature) ;
		A.map( v -> v*v ) ;			
		A.prefixLabels( "sqr " ) ;
		train = keepOriginals ? train.appendColumns(A) : F.appendColumns(A) ;

		Matrix T = test.dup() ;
		F = T.extractColumns(feature) ;
		
		T.map( v -> v*v ) ;			
		T.prefixLabels( "sqr " ) ;
		test = keepOriginals ? test.appendColumns(T) : F.appendColumns(T) ;
	}		
	
	public void log( boolean keepOriginals ) {
		int feature = getFeatureColumnIndex() ;
		Matrix A = train.dup() ;
		Matrix F = A.extractColumns(feature) ;
		
		A.map( v ->  Math.log1p( v*v ) ) ;			
		A.prefixLabels( "logsqr " ) ;
		train = keepOriginals ? train.appendColumns(A) : F.appendColumns(A) ;

		// Test data 
		Matrix T = test.dup() ;
		F = T.extractColumns(feature) ;
		
		T.map( v ->  Math.log1p( v*v ) ) ;			
		T.prefixLabels( "logsqr " ) ;
		test = keepOriginals ? test.appendColumns(T) : F.appendColumns(T) ;
	}
	
	public void reciprocal( boolean keepOriginals ) {
		int feature = getFeatureColumnIndex() ;
		Matrix A = train.dup() ;
		Matrix F = A.extractColumns(feature) ;
		
		A.map( v -> 1.0 / (v + 1.0) ) ;			
		A.prefixLabels( "recip. " ) ;
		train = keepOriginals ? train.appendColumns(A) : F.appendColumns(A) ;

		// Test data 
		Matrix T = test.dup() ;
		F = T.extractColumns(feature) ;
		
		T.map( v -> 1.0 / (v + 1.0) ) ;			
		T.prefixLabels( "recip. " ) ;
		test = keepOriginals ? test.appendColumns(T) : F.appendColumns(T) ;
	}

}