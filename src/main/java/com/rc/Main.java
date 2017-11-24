package com.rc;

import joptsimple.OptionParser;
import joptsimple.OptionSet;

public class Main {

	public static void main(String[] args) {
		Options options = new Options( args ) ;
		try {
			@SuppressWarnings("resource")
			Monitor m = new Monitor() ;
			m.start( options.port ) ;
			
		} catch( Throwable t ) {
			t.printStackTrace( ); 
			System.exit( 2 ); 
		}	
	}
}



class Options {
	
	int port = 8111 ;
	String platform = null ;
	
	public Options( String args[] ) {
		OptionParser parser = new OptionParser( );
		parser.accepts( "port", "Port number for website - defaults to 8111")
				.withRequiredArg().ofType( Integer.class ); 
		
		parser.accepts( "platform", "Preferred BLAs platform, cuda or openblas")
				.withRequiredArg().ofType( String.class ); ;
		
		OptionSet os = parser.parse( args ) ;
		if( os.has( "port" ) ) {
			port = (Integer)os.valueOf( "port" ) ;
		}
		if( os.has( "platform" ) ) {
			System.getProperties().setProperty( Compute.ComputeProperty, (String)os.valueOf( "platform" ) ) ;
		}
	}

}