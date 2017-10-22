package com.rc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class Compute implements AutoCloseable {
	final static Logger log = LoggerFactory.getLogger( Compute.class ) ;

	public final static String ComputeProperty = "compute_library" ;
	
	public abstract double dot( Matrix A, Matrix B ) ;
	public abstract double norm( Matrix A ) ;
	public abstract Matrix mmul( Matrix A, Matrix B ) ;
	public abstract Matrix solve( Matrix A, Matrix B ) ;
	public abstract Matrix solve2( Matrix A, Matrix B ) ;
	public abstract String getVersion() ;
	
	private static Class<? extends Compute> CLASS = null ;
	
	private static void InitLibrary() {	
		log.info( "Initializing compute library" ) ;
		
		String preferredLibrary = System.getProperty( ComputeProperty ) ;
		if( preferredLibrary == null ) {
			preferredLibrary = System.getenv( "compute_library" ) ;
		}
		if( preferredLibrary == null ) {
			log.info( "No preferred library specified - searching for best choice");				
			log.info( "Use -Dcompute_library=xxx or set global environment {}=xxx", ComputeProperty );
			log.info( "xxx is openblas or cuda");
		} else {
			if( preferredLibrary.equals( "openblas")  || preferredLibrary.equals( "cuda" ) ) {
				log.info( "preferred library {} specfied", preferredLibrary ) ;
			} else {
				log.info( "Invalid preferred library specified {} - searching for best choice", preferredLibrary ) ;				
			}
		}
		
		Class<? extends Compute> tmpClass = null ;
		if( preferredLibrary == null || "cuda".equalsIgnoreCase( preferredLibrary ) ) {
			log.info( "Looking for cuda library" ) ;
			try( Compute tmp = new Cuda() ) {
				tmpClass = tmp.getClass() ;	
				log.info( "Found cuda library" ) ; 
			} catch (Throwable ignore) {
				tmpClass = null ;
				if( "cuda".equalsIgnoreCase( preferredLibrary ) ) {
					log.warn( "Cannot find cuda library - trying openblas" ) ;
				}
			}
		}
		
		if( tmpClass == null ) { 
			try( Compute tmp = new Blas() ) {
				tmpClass = tmp.getClass() ;	
				log.warn( "Found openblas library" ) ; 
			} catch (Exception e) {
				tmpClass = null ;
				log.error( "Failed to find valid linear algebra library" ) ; 
				log.info( "Searching in: {}", System.getProperty("java.library.path") ) ;
				System.err.println( "**************************************************");
				System.err.println( "*  F A I L E D   T O   F I N D   L I B R A R Y   *");
				System.err.println( "*                                                *");				
				System.err.println( "*                 P A N I C                      *");
				System.err.println( "**************************************************");
				throw new Error( "Cannot load linear algebra library!" ) ; 
			}
		}
		CLASS = tmpClass ;
	}
	
	protected Compute() {		
	} 

	public static Compute getInstance() {
		synchronized( Compute.class ) { 
			if( CLASS == null ) {
				InitLibrary(); 
			}
		}
		try {
			return CLASS.newInstance() ;
		} catch (Exception e) {
			log.error( "Cannot create instance of linear algebra", e );
		} 
		return null ;
	}
}
