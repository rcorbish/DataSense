package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rc.clustering.Cluster;
import com.rc.clustering.DPMM;
import com.rc.clustering.MultinominalDPMM;
import com.rc.clustering.Point;


public class DpmmcDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( DpmmcDataProcessor.class ) ;

    private DPMM dpmm ;
    
	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		options.discrete = true ;
		Dataset dataset = Loader.load( DataProcessor.ROWS_TO_KEEP, data, options ) ;
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

		List<com.rc.clustering.Point> pointList = new ArrayList<>();
		//add records in pointList

        int dimensionality = A.N ;
		for( int i=0 ; i<A.M ; i++ ) {
			pointList.add( new com.rc.clustering.Point( i, A.copyRows(i).transpose() ) ) ;
		}

		log.info( "Processing {} data items", pointList.size() ) ;
		
        double alpha = 1 ;
        double words = 100 ;

        dpmm = new MultinominalDPMM(dimensionality, alpha, words);
        
        int maxIterations = 30 ;
		int performedIterations = dpmm.cluster(pointList, maxIterations);
        log.info( "Created {} clusters in {} iterations", dpmm.getClusterList().size(), performedIterations ) ;

		int n=dpmm.getClusterList().size() ;

		Map<Integer, Integer> zi = dpmm.getPointAssignments();
		log.info( "Points: {}", zi ) ;

		Matrix YH = new Matrix( YR.length() ) ;
		for( int i=0 ; i<T.M ; i++ ) {
			com.rc.clustering.Point xi = 
				new com.rc.clustering.Point( i, T.copyRows(i).transpose() ) ;
				double prob[] = dpmm.clusterProbabilities(xi, n) ;
				double mx = prob[0] ;
				int mxi = 0 ;
				for( int j=1 ; j<prob.length ; j++ ) {
					if( prob[j]>mx ) {
						prob[j] = mx ;
						mxi = j ;
					}
				}
				Cluster c = dpmm.getClusterList().get( mxi ) ;
				List<Point> pl = c.getPointList() ;
				double f = 0 ;
				for( Point p : pl ) {
					f += F.get( p.id ) ; 
				}
				f /= pl.size() ;
				YH.put(i, f ) ;
			}

		return score(YR, YH, inverseFeatureKeys ) ;
	}
	
	protected int[] getClusters() {
		//get a list with the point ids and their assignments
		Map<Integer, Integer> zi = dpmm.getPointAssignments();
		
		int clusters[] = new int[ zi.size() ] ;
		for( int id : zi.keySet() ) {
			clusters[id] = zi.get( id ) ;
		}	
		return clusters ;
	}
}


