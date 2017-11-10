package com.rc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StatisticsDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( StatisticsDataProcessor.class ) ;

	public Object process( Dataset dataset ) {
		// dataset.normalize();
		StatisticsResults X = new StatisticsResults() ; 
		X.minimum = dataset.train.min() ;
		X.maximum = dataset.train.max() ;
		X.countDistinct = dataset.train.countBuckets( 1e-4 ) ;
		X.median = dataset.train.median() ;
		X.mean = dataset.train.mean() ;
		X.stddev = dataset.train.stddev( X.mean ) ;
		X.skewness = dataset.train.skewness( X.mean ) ;
		X.kurtosis = dataset.train.kurtosis( X.mean ) ;
		X.labels = dataset.train.labels ;
		return X ;
	}
}

class StatisticsResults {
	String labels[] ;
	Matrix minimum ;
	Matrix maximum ;
	Matrix mean ;
	Matrix median ;
	Matrix countDistinct ;
	Matrix stddev ;
	Matrix skewness ;
	Matrix kurtosis ;
}