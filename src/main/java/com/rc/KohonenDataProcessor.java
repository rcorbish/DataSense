package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KohonenDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( KohonenDataProcessor.class ) ;

	final int TARGET_SPACE_SIZE = 20 ;

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

		int numInputs = A.N  ;

		Matrix targetSpace[] = new Matrix[ TARGET_SPACE_SIZE * TARGET_SPACE_SIZE ] ;
		for( int i=0 ; i<targetSpace.length ; i++ ) {
			targetSpace[i] = Matrix.rand( numInputs, 1 ) ;
		}

		double learningRate = 0.1 ;
		for( int iteration=0 ; iteration<100 ; iteration++ ) {
			Matrix shuffle = Matrix.shuffle( A.M ) ;
			for( int m=0 ; m<A.M ; m++ ) {
				int ix = (int)shuffle.get(m) ;
				Matrix observation = A.copyRows(ix).transpose() ;
				double closestDistance = observation.sub( targetSpace[0] ).norm() ;
				int closestIndex = 0 ;
				for( int i=1 ; i<targetSpace.length ; i++ ) {
					double distance = observation.sub( targetSpace[i] ).norm() ;
					if( distance < closestDistance ) {
						closestIndex = i ;
						closestDistance = distance ;
					}
				}
			
			// Now move all vectors towards the answer
			// depending on how near they are to the 'closest'
				// Matrix closest = targetSpace[ closestIndex ] ;
				int x0 = closestIndex % TARGET_SPACE_SIZE ;
				int y0 = closestIndex / TARGET_SPACE_SIZE ;
				
				for( int i=0 ; i<targetSpace.length ; i++ ) {
	
					// calc distance between closest & each vector in the target
					int x1 = closestIndex % TARGET_SPACE_SIZE ;
					int y1 = closestIndex / TARGET_SPACE_SIZE ;
					double proximity = Math.sqrt( (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) ) ;
					double rate = learningRate / (1.0 + proximity) ;
					targetSpace[i].subi( observation.mul( rate ) )  ;
				}
			}
		}

		Matrix Y = YR ;
		return score( YR, Y, inverseFeatureKeys )  ;
	}
}

