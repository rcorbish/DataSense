package com.rc;

import java.io.IOException;
import java.io.InputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

abstract public class DataProcessor {
	final static Logger log = LoggerFactory.getLogger( DataProcessor.class ) ;
	final static public int ROWS_TO_KEEP = 5_000 ;
	
	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		options.testRatio = 0 ;
		Dataset dataset = Loader.load( ROWS_TO_KEEP, data, options ) ;
		return dataset ;
	}
	
	abstract public Object process( Dataset dataset ) ;

	public Object process( InputStream data, ProcessorOptions options ) {
		Object rc = null ;
		
		try {
			Dataset dataset = load( data, options ) ;	
			rc = process( dataset ) ;
		} catch (IOException e) {
			e.printStackTrace();
			rc = e.getMessage() ;
		}
		return rc ;
	}
}
