package com.rc;

public class DataProcessorFactory {

	private DataProcessorFactory() {
		throw new UnsupportedOperationException() ;
	}
	
	public static DataProcessor getInstance( String method ) {
		
		if( "linear".equals( method ) ) {
			return new LinearDataProcessor() ;
		}
		if( "logistic".equals( method ) ) {
			return new LogisticDataProcessor() ;
		}
		if( "centroids".equals( method ) ) {
			return new CentroidDataProcessor() ;
		}
		if( "statistics".equals( method ) ) {
			return new StatisticsDataProcessor() ;
		}
		if( "correlation".equals( method ) ) {
			return new CorrelationDataProcessor() ;
		}
		throw new UnsupportedOperationException( method + " is not a supported processor type." ) ;
	}
}
