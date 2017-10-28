
package com.rc ;

import java.util.HashMap;
import java.util.Map;

public class Dataset {
	public final static String FEATURE_LABELS[] = { "Score", "Feature", "Result" } ;
	public final static String BIAS = "Bias" ;

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
}