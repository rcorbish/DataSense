
package com.rc.clustering;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.LoggerFactory;

/**
 * Dirichlet Process Mixture Model class.
 * 
 */
public abstract class DPMM  {
	private static org.slf4j.Logger log = LoggerFactory.getLogger(DPMM.class);

	/**
	 * Dimensionality of the data (the number of variables)
	 */
	protected final int dimensionality;

	/**
	 * Alpha value of Dirichlet process
	 */
	protected final double alpha;

	/**
	 * List of active clusters
	 */
	protected List<Cluster> clusterList;

	/**
	 * Constructor of DPMM
	 * 
	 * @param dimensionality    The dimensionality of the data that we cluster
	 * @param alpha     The alpha of the Dirichlet Process
	 */
	public DPMM(int dimensionality, double alpha) {
		this.dimensionality = dimensionality;
		this.alpha = alpha;
		clusterList = new ArrayList<>();
	}

	/**
	 * Getter method that returns the list of activate clusters
	 * 
	 * @return  List of Clusters
	 */
	public List<Cluster> getClusterList() {
		return clusterList;
	}

	/**
	 * Returns a map with pointId to Cluster Ids.
	 * 
	 * @return  Map with pointId => Cluster Ids
	 */
	public Map<Integer, Integer> getPointAssignments() {        
		int kn = clusterList.size();
		HashMap<Integer, Integer> pointAssignments = new HashMap<>( kn );

		for(int clusterId=0;clusterId<kn;++clusterId) {
			Cluster c=clusterList.get(clusterId);
			List<Point> pointList=c.getPointList();
			for(Point p : pointList) {
				pointAssignments.put(p.id,clusterId);
			}
		}
		return pointAssignments;
	}

	/**
	 * It calculates the clusters of a list of points.
	 * 
	 * @param pointList     The list of points
	 * @param maxIterations The maximum number of iterations
	 * @return  The actual number of iterations required
	 */
	public int cluster(List<Point> pointList, int maxIterations) {
		clusterList.clear();

		//run Collapsed Gibbs Sampling
		int performedIterations = collapsedGibbsSampling(pointList, maxIterations);
		return performedIterations;
	}

	/**
	 * Internal method used to instantiate a new cluster.
	 * 
	 * @return The new instance of the cluster object
	 */
	protected abstract Cluster generateCluster();

	/**
	 * Internal method used to create a new cluster and add it in the cluster list.
	 * 
	 * @return  The id of the new cluster.
	 */
	private Cluster createNewCluster() {
		//create new cluster
		Cluster c = generateCluster();

		//add the new cluster in our list
		clusterList.add(c);
		return c;
	}

	/**
	 * Implementation of Collapsed Gibbs Sampling algorithm.
	 * 
	 * @param pointList The list of points that we want to cluster
	 * @param maxIterations The maximum number of iterations
	 * @return  The actual number of iterations required
	 */
	private int collapsedGibbsSampling(List<Point> pointList, int maxIterations) {
		int n = pointList.size() ;
		int actualPointNumber = n ;
		for( Cluster c : clusterList ) {
			n += c.size();
		}

		//Initialize clusters, create a cluster for every xi
		for(Point xi : pointList) {
			Cluster cluster = createNewCluster() ; 
			cluster.addPoint(xi);
			xi.cluster = cluster ;
		}

		boolean noChangeMade=false;
		int iteration=0;

		while(iteration<maxIterations && noChangeMade==false) {
			log.info( "Iteration: {} has {} clusters", iteration, clusterList.size() ) ;

			noChangeMade=true;
			for(int i=0;i<actualPointNumber;++i) {
				Point xi = pointList.get(i);
				Cluster ci = xi.cluster ;

				//remove the point from the cluster
				ci.removePoint(xi);
				xi.cluster = null ;
				//if empty cluster remove it
				if( ci.size()==0 ) {
					clusterList.remove(ci);
				}

				int totalClusters = clusterList.size();
				double[] condProbCiGivenXiAndOtherCi = new double[totalClusters+1]; //plus one for the new possible cluster
				double[] currentClustersCondProb = clusterProbabilities(xi, n);

				//copy the values of the currentClusterCondProb to the larger array
				System.arraycopy(currentClustersCondProb, 0, condProbCiGivenXiAndOtherCi, 0, currentClustersCondProb.length);

				//Calculate the probabilities of assigning the point to a new cluster
				Cluster cNew = generateCluster();
				double priorLogPredictive = cNew.posteriorLogPdf(xi);

				double probNewCluster = alpha/(alpha+n-1);
				condProbCiGivenXiAndOtherCi[totalClusters]=priorLogPredictive+Math.log(probNewCluster);

				//normalize probabilities
				double max = condProbCiGivenXiAndOtherCi[0];
				for(int k=1;k<totalClusters+1;++k) {
					if(condProbCiGivenXiAndOtherCi[k]>max) {
						max=condProbCiGivenXiAndOtherCi[k];
					}
				}

				double sum = 0.0;
				for(int k=0;k<totalClusters+1;++k) {
					condProbCiGivenXiAndOtherCi[k]=Math.exp(condProbCiGivenXiAndOtherCi[k]-max);
					sum += condProbCiGivenXiAndOtherCi[k];
				}

				for(int k=0;k<totalClusters+1;++k) {
					condProbCiGivenXiAndOtherCi[k]/=sum; 
				}                    
				//sample a cluster according to the probability
				int ix = SRS.weightedProbabilitySampling( condProbCiGivenXiAndOtherCi ) ;
				
				Cluster sampledCluster = ix>=clusterList.size() ? 
							createNewCluster() : clusterList.get( ix ) ;                    

				sampledCluster.addPoint(xi);
				xi.cluster = sampledCluster ;
				if( ci != sampledCluster ) {
					noChangeMade=false;
				}
			}
			++iteration;
		}

		return iteration;
	}

	/**
	 * Estimate the probabilities of assigning the point to every cluster.
	 * 
	 * @param xi    The xi Point
	 * @param n     The number of activated Clusters
	 * @return      Array with the probabilities
	 */
	public double[] clusterProbabilities(Point xi, int n) {
		int totalClusters = clusterList.size();
		double[] condProbCiGivenXiAndOtherCi = new double[totalClusters];

		//Probabilities that appear on https://www.cs.cmu.edu/~kbe/dp_tutorial.posteriorLogPdf
		//Calculate the probabilities of assigning the point for every cluster
		for(int k=0;k<totalClusters;++k) {
			Cluster ck = clusterList.get(k);
			double marginalLogLikelihoodXi = ck.posteriorLogPdf(xi);
			double mixingXi = ck.size()/(alpha+n-1);

			condProbCiGivenXiAndOtherCi[k]=marginalLogLikelihoodXi+Math.log(mixingXi);
		}

		return condProbCiGivenXiAndOtherCi;
	}
}
