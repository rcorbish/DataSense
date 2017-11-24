package com.rc.clustering;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rc.Matrix;

public class GaussianDPMM extends DPMM {
	final static Logger log = LoggerFactory.getLogger( GaussianDPMM.class ) ;
   
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
        return new GaussianCluster(dimensionality, kappa0, nu0, mu0, psi0);        
    }
}
