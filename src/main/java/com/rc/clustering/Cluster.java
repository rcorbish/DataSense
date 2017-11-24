package com.rc.clustering;

import java.util.ArrayList;
import java.util.List;

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
