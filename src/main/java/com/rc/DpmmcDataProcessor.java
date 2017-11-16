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

import com.datumbox.opensource.clustering.DPMM;
import com.datumbox.opensource.clustering.GaussianDPMM;


public class DpmmcDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( DpmmcDataProcessor.class ) ;

    private DPMM dpmm ;
    
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

		List<com.datumbox.opensource.dataobjects.Point> pointList = new ArrayList<>();
		//add records in pointList

        int dimensionality = A.N ;
		for( int i=0 ; i<A.M ; i++ ) {
			pointList.add( new com.datumbox.opensource.dataobjects.Point( i, A.copyRows(i).transpose() ) ) ;
		}

		log.info( "Processing {} data items", pointList.size() ) ;
		
        double alpha = 0 ;
        //Hyper parameters of Base Function
        
        int kappa0 = 0 ;
        int nu0 = 0 ;
        Matrix mu0 = new Matrix( dimensionality );       
        Matrix psi0 = Matrix.eye( dimensionality ) ;

        dpmm = new GaussianDPMM(dimensionality, alpha, kappa0, nu0, mu0, psi0);
        
        int maxIterations = 30 ;
		int performedIterations = dpmm.cluster(pointList, maxIterations);
        log.info( "Created {} clusters in {} iterations", dpmm.getClusterList().size(), performedIterations ) ;

		int n=dpmm.getClusterList().size() ;

		Map<Integer, Integer> zi = dpmm.getPointAssignments();
		log.debug( "Points: {}", zi ) ;

		Matrix YH = new Matrix( YR.length() ) ;
		for( int i=0 ; i<T.M ; i++ ) {
			com.datumbox.opensource.dataobjects.Point xi = 
				new com.datumbox.opensource.dataobjects.Point( i, T.copyRows(i).transpose() ) ;
				double p[] = dpmm.clusterProbabilities(xi, n) ;
				log.debug( "Cum probs: {}", p ) ;
				double mx = p[0] ;
				int mxi = 0 ;
				for( int j=1 ; j<p.length ; j++ ) {
					if( p[j]>mx ) {
						p[j] = mx ;
						mxi = j ;
					}
				}
				YH.put(i, mxi ) ;
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


