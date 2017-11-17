
package com.datumbox.opensource.clustering;

import com.datumbox.opensource.dataobjects.Point;
import com.rc.Matrix;


public class MultinomialCluster extends Cluster {
	/**
	 * Sum of word counts of observations; it is used in calculation of cluster.
	 */
	protected Matrix Ni_sum;

	/**
	 * Cached value of WordCountsPlusAlpha used only for speed optimization
	 */
	private double cache_wordcounts_plusalpha;
	private Matrix wordCountsPlusAlpha ;
	
	//Alpha parameter of Dirichlet for the words
	private final double alphaWords;

	//cluster parameters
	private Matrix wordCounts;

	/**
	 * Constructor of Mutinomial Mixture Cluster
	 * 
	 * @param dimensionality    The dimensionality of the data that we cluster
	 * @param alphaWords        The alpha parameter of the Dirichlet Process for Words
	 */
	public MultinomialCluster(int dimensionality, double alphaWords) {
		super(dimensionality);

		if(alphaWords<0) {
			alphaWords = 50.0; //effectively we set alphaWords = 50. The alphaWords controls the amount of words in each cluster. In most notes it is notated as alpha.
		}
		this.alphaWords = alphaWords;

		wordCounts = new Matrix(dimensionality);
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
			Ni_sum=xi.data;
		}
		else {
			Ni_sum=Ni_sum.add(xi.data);
		}

		pointList.add(xi);
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
		Ni_sum=Ni_sum.sub(xi.data);

		pointList.remove(index);

		updateClusterParameters();
	}


	/**
	 * Updates the cluster's internal parameters based on the stored information.
	 */
	@Override
	protected void updateClusterParameters() {
		Matrix aVector = Matrix.fill(dimensionality, alphaWords);
		wordCountsPlusAlpha = wordCounts.add(aVector);

		cache_wordcounts_plusalpha=C(wordCountsPlusAlpha);

		wordCounts = Ni_sum;
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
		double cOfWordCountsPlusAlpha=cache_wordcounts_plusalpha;

		double logPdf= C(wordCountsPlusAlpha.add(xi.data))-cOfWordCountsPlusAlpha;
		return logPdf;
	}

	/**
	 * Internal method that estimates the value of C(a).
	 * 
	 * @param alphaVector   Vector with alpha values
	 * @return              Returns the value of C(a)
	 */
	private double C(Matrix alphaVector) {
		double sumAi=0.0;
		double sumLogGammaAi=0.0;

		int aLength=alphaVector.length();
		for(int i=0;i<aLength;++i) {
			double tmp=alphaVector.get(i);
			sumAi+= tmp;
			sumLogGammaAi+=logGamma(tmp);
		}

		double Cvalue = sumLogGammaAi-logGamma(sumAi);

		return Cvalue;
	}

	/**
	 * It estimates a numeric approximation of LogGamma function.
	 * Modified code from http://introcs.cs.princeton.edu/java/91float/Gamma.java.html
	 * 
	 * @param x     The input x
	 * @return      The value of LogGamma(x)
	 */
	private double logGamma(double x) {
		double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
		double ser = 1.0 + 76.18009173    / (x + 0)   - 86.50532033    / (x + 1)
				+ 24.01409822    / (x + 2)   -  1.231739516   / (x + 3)
				+  0.00120858003 / (x + 4)   -  0.00000536382 / (x + 5);
		return tmp + Math.log(ser * Math.sqrt(2 * Math.PI));
	}

}

