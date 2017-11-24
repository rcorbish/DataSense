
package com.rc.clustering;

import com.rc.Matrix;

/**
 * Point Object is used to store the id and the data of the xi record.
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class Point {
    /**
     * The id variable is used to identify the xi record.
     */
    public int id;

    public Cluster cluster ;
    
    /**
     * The data variable is a RealVector which stores the information of xi record.
     */
    public Matrix data;
    
    /**
     * Point Constructor which accepts a RealVector input.
     * 
     * @param id    The integer id of the point
     * @param data  The data of the point
     */
    public Point(int id, Matrix data )  {
        this.id = id;
        this.data = data;
    }
}
