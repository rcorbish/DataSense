package com.rc;

import java.io.File;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import spark.Request;
import spark.Response;

/**
 * This handles the web pages. 
 * 
 * We use spark to serve pages. It's simple and easy to configure. 
 * 
 * @author richard
 *
 */
public class Monitor implements AutoCloseable {
	
	final static Logger logger = LoggerFactory.getLogger( Monitor.class ) ;

	final Random random ;
	final Gson gson ;
	final DataProcessor processor ;

	public Monitor() {
		this.random = new Random() ;
		gson = new GsonBuilder()
				.registerTypeAdapter( Matrix.class, new Matrix.Deserializer() )
				.setPrettyPrinting()
				.create() ;
		
		this.processor = new DataProcessor() ;
	}
	
	
	public void start( int port ) {
		try {			
			spark.Spark.port( port ) ;
			URL mainPage = getClass().getClassLoader().getResource( "Client.html" ) ;
			File path = new File( mainPage.getPath() ) ;
			spark.Spark.staticFiles.externalLocation( path.getParent() ) ;
			spark.Spark.post( "/upload-data", this::postData, gson::toJson ) ;
			spark.Spark.awaitInitialization() ;
		} catch( Exception ohohChongo ) {
			logger.error( "Server start failure.", ohohChongo );
		}
	}

	private final static String RFC7230CHARS = "[\\w\\!\\#\\$\\%\\&\\'\\*\\+\\-\\.\\^\\`\\|\\~]" ;
	
	public Object postData(Request req, Response rsp) {
		Object rc = null ;
		// Pattern to match content type header incl. optional charset
		Pattern ctPattern = Pattern.compile( "(" + RFC7230CHARS + "+/" + RFC7230CHARS + "+)" +
											 "([\\s]*;[\\s]*charset[\\s]*=[\\s]*("+RFC7230CHARS+"+))?" ) ;
		try {
			rsp.type( "application/json" );	
			rsp.header("expires", "0" ) ;
			rsp.header("cache-control", "no-cache" ) ;
			String contentType = req.contentType() ;
			Matcher matcher = ctPattern.matcher( contentType ) ;
			String charset = "ISO-8859-1" ;
			String mime = null ; //contentType.trim() ;
			
			if( matcher.matches() ) {
				mime = matcher.group(1) ;
				if( matcher.groupCount()>2 && matcher.group(3) != null ) {
					charset = matcher.group(3) ;
				}
			}
			if( !"text/csv".equals(mime) ) {
				throw new RuntimeException( "Invalid content type - only text/csv supported" ) ;
			}
			Charset cs = Charset.forName( charset ) ;			
			InputStream is = req.raw().getInputStream() ;
			rc = processor.process( is, cs ) ;
		} catch ( Throwable t ) {
			logger.warn( "Error processing getItem request", t ) ;
			rsp.status( 400 ) ;	
		}
		return rc ;
	}


	@Override
	public void close() {
		spark.Spark.stop() ;
	}
}