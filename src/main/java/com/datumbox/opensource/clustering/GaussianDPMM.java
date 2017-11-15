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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.datumbox.opensource.dataobjects.Point;
import com.rc.Matrix;



/**
 *
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class GaussianDPMM extends DPMM {
	final static Logger log = LoggerFactory.getLogger( GaussianDPMM.class ) ;
    
    /**
     * Multivariate Normal with Normal-Inverse-Wishart prior.
     * References:
     *      http://snippyhollow.github.io/blog/2013/03/10/collapsed-gibbs-sampling-for-dirichlet-process-gaussian-mixture-models/
     *      http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/
     *      http://www.cs.princeton.edu/courses/archive/fall07/cos597C/scribe/20070921.pdf
     * 
     * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
     */
    public class Cluster extends DPMM.Cluster {
        
        //hyper parameters
        private final int       kappa0;
        private final int       nu0;
        private final Matrix    mu0;
        private final Matrix    psi0;
        
        //cluster parameters
        private Matrix mean;
        private Matrix covariance;
        
        //validation - confidence interval vars
        private Matrix meanError;
        private int meanDf;
        
        /**
         * Sum of observations used in calculation of cluster clusterParameters such 
         * as mean.
         */
        private Matrix xi_sum;
        /**
         * Sum of squared of observations used in calculation of cluster 
         * clusterParameters such as variance.
         */
        private Matrix xi_square_sum;

        /**
         * Cached value of Covariance determinant used only for speed optimization
         */
        private double cache_covariance_determinant;


        /**
         * Constructor of Gaussian Mixture Cluster
         * 
         * @param dimensionality    The dimensionality of the data that we cluster
         * @param kappa0    Mean fraction parameter
         * @param nu0   Degrees of freedom for Inverse-Wishart
         * @param mu0   Mean vector for Normal
         * @param psi0  Pairwise deviation product of Inverse Wishart
         */
        public Cluster(int dimensionality, int kappa0, int nu0, Matrix mu0, Matrix psi0) {
            super(dimensionality);
            
            if( nu0<dimensionality ) {
                nu0 = dimensionality;
            }
            if(mu0==null) {
                mu0 = new Matrix(dimensionality) ; //0 vector
            }

            if(psi0==null) {
                psi0 = Matrix.eye(dimensionality); //identity matrix
            }
            
            this.kappa0 = kappa0;
            this.nu0 = nu0;
            this.mu0 = mu0;
            this.psi0 = psi0;
            
            mean = new Matrix(dimensionality);
            covariance = Matrix.eye(dimensionality);
            cache_covariance_determinant = 1.0 ;

            meanError = calculateMeanError(psi0, kappa0, nu0);
            meanDf = Math.max(0, nu0-dimensionality+1);
        }
        
        /**
         * Adds a single point in the cluster.
         * 
         * @param xi    The point that we wish to add in the cluster.
         */
        @Override
        public void addPoint(Point xi) {
            addSinglePoint(xi);
            updateClusterParameters();
        }

        /**
         * Internal method that adds the point int cluster and updates clusterParams
         * 
         * @param xi    The point that we wish to add in the cluster.
         */
        private void addSinglePoint(Point xi) {
            int nk= pointList.size();

            //update cluster clusterParameters
            if(nk==0) {
                xi_sum=xi.data;
                xi_square_sum=xi.data.outer(xi.data);
            }
            else {
                xi_sum=xi_sum.add(xi.data);
                xi_square_sum=xi_square_sum.add(xi.data.outer(xi.data));
            }

            pointList.add(xi);
        }

        /**
         * Removes a point from the cluster.
         * 
         * @param xi    The point that we wish to remove from the cluster
         */
        @Override
        public void removePoint(Point xi) {
            int index = pointList.indexOf(xi);
            if(index==-1) {
                return;
            }

            //update cluster clusterParameters
            xi_sum=xi_sum.sub(xi.data);
            xi_square_sum=xi_square_sum.sub(xi.data.outer(xi.data));

            pointList.remove(index);

            updateClusterParameters();
        }

        private Matrix calculateMeanError(Matrix Psi, int kappa, int nu) {
            //Reference: page 18, equation 228 at http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
            return Psi.mul(1.0/(kappa*(nu-dimensionality+1.0)));
        }

        /**
         * Updates the cluster's internal parameters based on the stored information.
         */
        @Override
        protected void updateClusterParameters() {
            int n = pointList.size();
            if(n<=0) {
                return;
            }

            int kappa_n = kappa0 + n;
            int nu = nu0 + n;

            Matrix mu = xi_sum.div(n);
            log.debug( "mu {}", mu ) ;
            log.debug( "mu0 {}", mu0 ) ;
            Matrix mu_mu_0 = mu.sub(mu0);

            Matrix C = xi_square_sum.sub( ( mu.outer(mu) ).mul(n) );

            Matrix psi = psi0.add( C.add( ( mu_mu_0.outer(mu_mu_0) ).mul(kappa0*n/(double)kappa_n) ));
            C = null;
            mu_mu_0 = null;

            mean = ( mu0.mul(kappa0) ).add( mu.mul(n) ).div(kappa_n);
            covariance = psi.mul( (kappa_n+1.0)/(kappa_n*(nu - dimensionality + 1.0)) );
            cache_covariance_determinant=covariance.det() ;
     
            log.debug( "Covariance {}", covariance ) ;
            meanError = calculateMeanError(psi, kappa_n, nu);
            meanDf = Math.max(0, nu-dimensionality+1);
        }

        /**
         * Returns the log posterior PDF of a particular point xi, to belong to this
         * cluster.
         * 
         * @param xi    The point for which we want to estimate the PDF.
         * @return      The log posterior PDF
         */
        @Override
        public double posteriorLogPdf(Point xi) {
            Matrix x_mu = xi.data.sub(mean);
 
            log.debug( "Covariance {}", covariance ) ;
            log.debug( "Mu {}", x_mu ) ;
            
            double x_muInvSx_muT = covariance.divLeft( x_mu ).dot( x_mu ) ;

            double normConst = 1.0/( Math.pow(2*Math.PI, dimensionality/2.0) * Math.pow(cache_covariance_determinant, 0.5) );

            //double pdf = Math.exp(-0.5 * x_muInvSx_muT)*normConst;
            double logPdf = -0.5 * x_muInvSx_muT + Math.log(normConst);
            return logPdf;
        }
        
        /**
         * Getter for the mean of the cluster.
         * 
         * @return  The mean vector
         */
        public Matrix getMean() {
            return mean;
        }
        
        /**
         * Getter for the covariance of the cluster.
         * 
         * @return  The Covariance Matrix
         */
        public Matrix getCovariance() {
            return covariance;
        }

        /**
         * Getter for Mean Error of the cluster.
         * 
         * @return  The Mean Error Matrix
         */
        public Matrix getMeanError() {
            return meanError;
        }
        
        /**
         * The degrees of freedom of Student's Distribution.
         * http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf Equation 228
         * 
         * @return 
         */
        public int getMeanDf() {
            return meanDf;
        }

    }

   
    private final int kappa0;
    private final int  nu0;
    private final Matrix mu0;
    private final Matrix psi0;
    
    /**
     * Constructor of Gaussian DPMM.
     * 
     * @param dimensionality    The dimensionality of the data that we cluster
     * @param alpha     The alpha of the Dirichlet Process
     * @param kappa0    Mean fraction parameter
     * @param nu0   Degrees of freedom for Inverse-Wishart
     * @param mu0   Mean vector for Normal
     * @param psi0  Pairwise deviation product of Inverse Wishart
     */
    public GaussianDPMM(int dimensionality, double alpha, int kappa0, int  nu0, Matrix mu0, Matrix psi0) {
        super(dimensionality, alpha);
            
        this.kappa0 = kappa0;
        this.nu0 = nu0;
        this.mu0 = mu0;
        this.psi0 = psi0;
    }

    @Override
    protected Cluster generateCluster() {
        return new GaussianDPMM.Cluster(dimensionality, kappa0, nu0, mu0, psi0);        
    }
}
