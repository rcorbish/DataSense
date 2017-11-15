/* 
 * Copyright (C) 2014 Vasilis Vryniotis <bbriniotis at datumbox.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package com.datumbox.opensource.clustering;

import com.datumbox.opensource.dataobjects.Point;
import com.datumbox.opensource.sampling.SRS;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.LoggerFactory;

/**
 * Dirichlet Process Mixture Model class.
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 * @param <CL>
 */
public abstract class DPMM  {
	private static org.slf4j.Logger log = LoggerFactory.getLogger(DPMM.class);

    /**
     * The abstract Cluster calls defines all the methods that must be implemented 
     * by a particular Cluster of Mixtrure Model. Every method that depends on the mathematical
     * model, the posterior and the likelihood must be implemented here.
     * 
     * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
     */
    public abstract class Cluster {
        /**
         * List of points assigned to this cluster
         */
        protected List<Point> pointList;

        /**
         * Dimensionality of the data, the number of variables
         */
        protected int dimensionality;

        /**
         * Constructor of Cluster.
         * 
         * @param dimensionality    The dimensionality of the data that we cluster
         */
        public Cluster(int dimensionality) {
            this.dimensionality = dimensionality;
            pointList = new ArrayList<>();
        }

        /**
         * Getter method that returns the list of points of cluster
         * 
         * @return  List of points
         */
        public List<Point> getPointList() {
            return pointList;
        }

        /**
         * Returns the number of points that are stored in the cluster.
         * 
         * @return  Number of points in cluster
         */
        public int size() {
            return pointList.size();
        }

        /**
         * Adds a single point in the cluster.
         * 
         * @param xi    The point that we wish to add in the cluster.
         */
        public abstract void addPoint(Point xi);

        /**
         * Removes a point from the cluster.
         * 
         * @param xi    The point that we wish to remove from the cluster
         */
        public abstract void removePoint(Point xi);

        /**
         * Updates the cluster's internal parameters based on the stored information.
         */
        protected abstract void updateClusterParameters();

        /**
         * Returns the log posterior PDF of a particular point xi, to belong to this
         * cluster.
         * 
         * @param xi    The point for which we want to estimate the PDF.
         * @return      The log posterior PDF
         */
        public abstract double posteriorLogPdf(Point xi);
    }

    
    /**
     * Dimensionality of the data (the number of variables)
     */
    protected final Integer dimensionality;
    
    /**
     * Alpha value of Dirichlet process
     */
    protected final double alpha;
    
    /**
     * List of active clusters
     */
    protected List<DPMM.Cluster> clusterList;
    
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
    public List<DPMM.Cluster> getClusterList() {
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
            DPMM.Cluster c=clusterList.get(clusterId);
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
    protected abstract DPMM.Cluster generateCluster();
    
    /**
     * Internal method used to create a new cluster and add it in the cluster list.
     * 
     * @return  The id of the new cluster.
     */
    private int createNewCluster() {
        int clusterId=clusterList.size(); //the ids start enumerating from 0
        
        //create new cluster
        DPMM.Cluster c = generateCluster();
        
        //add the new cluster in our list
        clusterList.add(clusterId, c);
        
        return clusterId;
    }
    
    /**
     * Implementation of Collapsed Gibbs Sampling algorithm.
     * 
     * @param pointList The list of points that we want to cluster
     * @param maxIterations The maximum number of iterations
     * @return  The actual number of iterations required
     */
    private int collapsedGibbsSampling(List<Point> pointList, int maxIterations) {
        int n = pointList.size();
        int actualPointNumber = n;
        if(clusterList.size()>0) { //if we have already clusters then measure the points that are also assigned in them
            for(DPMM.Cluster c : clusterList) {
                n+=c.size();
            }
        }
        
        Map<Integer, DPMM.Cluster> pointId2Cluster = new HashMap<>(); //pointId=>cluster
        

        //Initialize clusters, create a cluster for every xi
        for(Point xi : pointList) {
            int clusterId=createNewCluster();
            clusterList.get(clusterId).addPoint(xi);
            pointId2Cluster.put(xi.id, clusterList.get(clusterId));
        }
        
        boolean noChangeMade=false;
        int iteration=0;
        
        while(iteration<maxIterations && noChangeMade==false) {
            log.info( "Iteration: {} has {} clusters", iteration, clusterList.size() ) ;
            
            noChangeMade=true;
            for(int i=0;i<actualPointNumber;++i) {
                Point xi = pointList.get(i);
                DPMM.Cluster ci = pointId2Cluster.get(xi.id);
                
                boolean wasInClusterOfOne = false ;
                //remove the point from the cluster
                ci.removePoint(xi);
                //if empty cluster remove it
                if( ci.size()==0 ) {
                    clusterList.remove(ci);
                    pointId2Cluster.remove(xi.id);
                    wasInClusterOfOne = true ;
                }
                
                int totalClusters = clusterList.size();
                double[] condProbCiGivenXiAndOtherCi = new double[totalClusters+1]; //plus one for the new possible cluster
                double[] currentClustersCondProb = clusterProbabilities(xi, n);
                
                //copy the values of the currentClusterCondProb to the larger array
                System.arraycopy(currentClustersCondProb, 0, condProbCiGivenXiAndOtherCi, 0, currentClustersCondProb.length);
                
                //Calculate the probabilities of assigning the point to a new cluster
                DPMM.Cluster cNew = generateCluster();
                
                double priorLogPredictive = cNew.posteriorLogPdf(xi);
                cNew=null;
                
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
                
                int sampledClusterId;
                if(sum!=0.0) {
                    for(int k=0;k<totalClusters+1;++k) {
                        condProbCiGivenXiAndOtherCi[k]/=sum; 
                    }
                    
                    //sample a cluster according to the probability
                    sampledClusterId=SRS.weightedProbabilitySampling(condProbCiGivenXiAndOtherCi);
                    condProbCiGivenXiAndOtherCi=null;
                }
                else {
                    //if all probabilities are 0 then assign it to a new cluster
                    sampledClusterId=totalClusters;
                }
                
                
                //Add Xi back to the sampled Cluster
                if(sampledClusterId==totalClusters) { //if new cluster
                    int newClusterId=createNewCluster();
                    clusterList.get(newClusterId).addPoint(xi);
                    if( wasInClusterOfOne ) {
                        noChangeMade=false ;
                    }
                }
                else {
                    clusterList.get(sampledClusterId).addPoint(xi);
                    if(noChangeMade && ci!=clusterList.get(sampledClusterId)) {
                        noChangeMade=false;
                    }
                }
                pointId2Cluster.put(xi.id,clusterList.get(sampledClusterId));
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
    private double[] clusterProbabilities(Point xi, int n) {
        int totalClusters = clusterList.size();
        double[] condProbCiGivenXiAndOtherCi = new double[totalClusters];

        //Probabilities that appear on https://www.cs.cmu.edu/~kbe/dp_tutorial.posteriorLogPdf
        //Calculate the probabilities of assigning the point for every cluster
        for(int k=0;k<totalClusters;++k) {
            DPMM.Cluster ck = clusterList.get(k);
            double marginalLogLikelihoodXi = ck.posteriorLogPdf(xi);
            double mixingXi = ck.size()/(alpha+n-1);

            condProbCiGivenXiAndOtherCi[k]=marginalLogLikelihoodXi+Math.log(mixingXi);
        }
        
        return condProbCiGivenXiAndOtherCi;
    }
}
