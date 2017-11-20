
package com.datumbox.opensource.clustering;

/**
 *
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class MultinominalDPMM extends DPMM {
    
    private final double alphaWords;
    
    /**
     * Constructor of Multinomial DPMM.
     * 
     * @param dimensionality    The dimensionality of the data that we cluster
     * @param alpha             The alpha of the Dirichlet Process
     * @param alphaWords        The second alpha of the Dirichlet Process for Words
     */
    public MultinominalDPMM(int dimensionality, double alpha, double alphaWords) {
        super(dimensionality, alpha);            
        this.alphaWords = alphaWords;
    }

    @Override
    protected Cluster generateCluster() {
        return new MultinomialCluster(dimensionality, alphaWords);        
    }
}
