package com.rc;

import java.nio.charset.Charset;
import java.text.ParseException;
import java.util.Calendar;
import java.util.Map;

public class ProcessorOptions {
	Charset cs ;
	boolean square ;
	boolean log ;
	boolean reciprocal ;
	boolean discrete ;
	boolean keepOriginal ;
	boolean normalize ;
	String  method ;
	DateParserFunction dateParser ;
	int dateBaseline ;
	double testRatio = 0.2 ;


	public ProcessorOptions( Map<String, String[]> params ) throws ParseException {
		square   	= get( params, "square-values") != null  ;
		log		 	= get( params, "log-values") != null  ;
		discrete 	= get( params, "discrete-to-col") != null  ;
		keepOriginal= get( params, "keep-original" ) != null  ;
		reciprocal  = get( params, "reciprocal" ) != null  ;
		normalize   = get( params, "normalize" ) != null  ;
		
		method   = get( params, "method" ) ;
		
		String baseline = get( params, "date-baseline" ) ;
		Calendar today = Calendar.getInstance() ;
		dateBaseline = 0 ;

		if( get( params, "to-years" ) != null ) {
			dateParser = ProcessorOptions::toYears ;
			if( baseline.equals( "1900") ) {
				dateBaseline = 1900 ;
			} else if( baseline.equals( "2000") ) {
				dateBaseline = 2000 ;
			} else {
				dateBaseline = today.get( Calendar.YEAR ) ;
			}
		}
		if( get( params, "to-months" ) != null ) {
			dateParser = ProcessorOptions::toMonths ;
			if( baseline.equals( "1900") ) {
				dateBaseline = 190000 ;
			} else if( baseline.equals( "2000") ) {
				dateBaseline = 200000 ;
			} else {
				dateBaseline = today.get( Calendar.YEAR ) * 100 + today.get( Calendar.MONTH) ;
			}
		}
		if( get( params, "to-days" ) != null ) {
			dateParser = ProcessorOptions::toDays ;
			if( baseline.equals( "1900") ) {
				dateBaseline = 19000000 ;
			} else if( baseline.equals( "2000") ) {
				dateBaseline = 20000000 ;
			} else {
				dateBaseline = today.get( Calendar.YEAR ) * 10000 
				+ today.get( Calendar.MONTH) * 100 
				+ today.get( Calendar.DAY_OF_MONTH ) ;}
		}
		//String dateText = get( params, "date-baseline" ) ;
	}

	public static int toYears( String years ) {
		return 0 ;
	}
	public static int toMonths( String years ) {
		return 0 ;
	}
	public static int toDays( String years ) {
		return 0 ;
	}

	public String get( Map<String, String[]> params, String key ) {
		String rc = null ;
		String s[] = params.get( key ) ;
		if( s != null ) {
			rc = s[0] ;
		}
		return rc ;
	}

	@FunctionalInterface
	public interface DateParserFunction {
		public int parse( String in ) ;
	}
}

