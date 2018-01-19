package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KohonenDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( KohonenDataProcessor.class ) ;


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

		int feature = dataset.getFeatureColumnIndex() ;

		Matrix F = A.extractColumns( feature ) ;
		Matrix YR = T.extractColumns( feature ) ; 

		int numInputs = A.N  ;
		final int TARGET_SPACE_SIZE = 10 ;

		Matrix targetSpace[] = new Matrix[ TARGET_SPACE_SIZE * TARGET_SPACE_SIZE ] ;
		for( int i=0 ; i<targetSpace.length ; i++ ) {
			targetSpace[i] = Matrix.rand( numInputs, 1 ) ;
		}

		double learningRate = 1.0 ;
		final int ITERATIONS = 100 ;

		double MAP_RADIUS = TARGET_SPACE_SIZE / 2.0 ;
		double RADIUS_LAMBDA = ITERATIONS / Math.log( MAP_RADIUS ) ;
		
		for( int iteration=0 ; iteration<ITERATIONS ; iteration++ ) {
			double radius = MAP_RADIUS * Math.exp( -iteration/RADIUS_LAMBDA ) ;
			log.info( "Iteration {} - radius = {}", iteration, radius ) ;
					
			Matrix shuffle = Matrix.shuffle( A.M ) ;
			for( int m=0 ; m<A.M ; m++ ) {
				int ix = (int)shuffle.get(m) ;
				Matrix observation = A.copyRows(ix) ;

				int closestIndex = findClosestIndex(observation, targetSpace ) ;
			
			// Now move all vectors towards the answer
			// depending on how near they are to the 'closest'
				// Matrix closest = targetSpace[ closestIndex ] ;
				int x0 = closestIndex % TARGET_SPACE_SIZE ;
				int y0 = closestIndex / TARGET_SPACE_SIZE ;
	
				for( int i=0 ; i<targetSpace.length ; i++ ) {
	
					// calc distance between closest & each vector in the target
					int x1 = i % TARGET_SPACE_SIZE ;
					int y1 = i / TARGET_SPACE_SIZE ;
					double proximity = Math.sqrt( (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) ) ;
					if( proximity <= radius ) {
						double theta = Math.exp( -(proximity*proximity) / ( 2 * radius ) ) ;
						double rate = learningRate * theta ;
						Matrix delta = observation.sub( targetSpace[i] ).mul( rate ) ;
//						log.debug( "Moving\n{} by\n{} due to\n{}",targetSpace[i], delta, observation ) ;
						targetSpace[i].addi( delta ) ;
					}
				}
			}
			log.debug( "Iteration {} complete", iteration ) ;
		}
		log.info( "Iterations complete" ) ;
		
		// value in dataset -> zero based index
		Map<Integer,Integer> featureKeys = new HashMap<>() ;

		for( int m=0 ; m<A.M ; m++ ) {
			Matrix observation = A.copyRows(m) ;
			int closestIndex = findClosestIndex(observation, targetSpace ) ;
			featureKeys.put( closestIndex, (int)F.get(m) ) ;
			log.info( "{} :  {} -> {}", m, F.get(m), targetSpace[closestIndex] ) ;
		}
		
		
		log.info( "Feature keys in target: {}", featureKeys ) ;
		
		Matrix idealTargets[] = new Matrix[ featureKeys.size() ] ;
		int idealTargetIndices[] = new int[idealTargets.length] ;
		int ix = 0 ;
		for( int fk :featureKeys.keySet() ) {
			idealTargets[ix] = targetSpace[ fk ] ;
			idealTargetIndices[ix] = fk ;
			ix++ ;
		}

		Matrix Y = new Matrix( YR.M )  ;
		for( int m=0 ; m<T.M ; m++ ) {
			Matrix observation = A.copyRows(m) ;
			int closestIndex = findClosestIndex( observation, idealTargets ) ;
			log.info( "Mapping {} using {}", closestIndex, featureKeys ) ;
			Y.put( m, idealTargetIndices[closestIndex] ) ;
		}

		Map<Integer,Integer> inverseFeatureKeys = new HashMap<>() ;
		
		for( int i=0 ; i<idealTargetIndices.length ; i++ ) {
			inverseFeatureKeys.put( i, idealTargetIndices[i] ) ;
		}

		return score( YR, Y, inverseFeatureKeys )  ;
	}


	protected int findClosestIndex( Matrix v, Matrix w[] ) {
		double closestDistance = v.sub( w[0] ).norm() ;
		int closestIndex = 0 ;
		for( int i=1 ; i<w.length ; i++ ) {
			double distance = v.sub( w[i] ).norm() ;
			if( distance < closestDistance ) {
				closestIndex = i ;
				closestDistance = distance ;
			}
		}
		return closestIndex ;
	}

}

