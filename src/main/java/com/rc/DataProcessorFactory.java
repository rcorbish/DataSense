package com.rc;

public class DataProcessorFactory {

	private DataProcessorFactory() {
		throw new UnsupportedOperationException() ;
	}
	
	public static DataProcessor getInstance( String method ) {
		
		if( "svmc".equals( method ) ) {
			return new SvmcDataProcessor() ;
		}
		if( "svmr".equals( method ) ) {
			return new SvmrDataProcessor() ;
		}
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
		if( "dpmmg".equals( method ) ) {
			return new DpmmgDataProcessor() ;
		}
		if( "dpmmc".equals( method ) ) {
			return new DpmmcDataProcessor() ;
		}
		throw new UnsupportedOperationException( method + " is not a supported processor type." ) ;
	}
}
