
package com.rc.clustering;

import java.util.Random;

/**
 * Simple Random Sampling class.
 * 
 */
public class SRS {
    
    static Random rng = new Random() ;
    /**
     * Samples an observation based on a probability Table.
     * 
     * @param probabilityTable  The probability table
     * @return  The index that was selected based on sampling
     */

    public static int weightedProbabilitySampling(double probabilityTable[] ) {
        int sampledId=0;
        double randomNumber = rng.nextDouble() ;
        
        double probabilitySumSelector = 0.0;
        
        for(int i=0;i<probabilityTable.length;++i) {
            probabilitySumSelector+=probabilityTable[i];
            if(randomNumber<probabilitySumSelector) {
                sampledId=i;
                break;
            }
        }
        return sampledId;
    }
}
