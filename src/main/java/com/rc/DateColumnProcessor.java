package com.rc;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DateColumnProcessor {
	final static Logger log = LoggerFactory.getLogger( DateColumnProcessor.class ) ;

	
	private DateColumnProcessor() {
		throw new UnsupportedOperationException( "DateColumnProcessor is a singleton" ) ;
	}
	
	
	public static Date parse( String col ) {

		Date rc = null ;
		
		final Pattern datePattern0 = Pattern.compile("([\\d]{4})[\\W]?([\\d]{2})[\\W]?([\\d]{2}).*([\\d]{2})[\\:]?([\\d]{2})[\\:]?([\\d]{2})") ;
		final Pattern datePattern1 = Pattern.compile("([\\d]{4})[\\W]?([\\d]{2})[\\W]?([\\d]{2})") ;
		final Pattern datePattern2 = Pattern.compile("([\\d]{2})[\\W]?([\\d]{2})[\\W]?([\\d]{4})") ;
		final Pattern datePattern3 = Pattern.compile("([\\d]{2})[\\W]?([\\d]{2})[\\W]?([\\d]{4}).*([\\d]{2}):([\\d]{2}):([\\d]{2})") ;

		DateFormat df0 = new SimpleDateFormat("") ;
		Matcher m = datePattern0.matcher( col ) ;
		if( m.find() ) {
			SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss" ) ;
			try {
				rc = sdf.parse(col) ;
			} catch( ParseException pex ) {
				log.warn( "Invlid date {} - {}", col, pex.getMessage() ) ;
			}
		} else {
		}
		return rc ;
	}
	
}
