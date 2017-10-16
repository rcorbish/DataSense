package com.rc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Loader {
	final static Logger log = LoggerFactory.getLogger( Blas.class ) ;

	private static final Random rng = new Random(7) ;

	public static Matrix loadFromCsv( int M, Path file ) throws IOException {

		log.info( "Loading up to {} rows from {}", M, file ) ;
		InputStream is = Files.newInputStream(file) ;
		return load( M, is, Charset.defaultCharset() ) ;
	}

	public static Matrix load( int M, InputStream is, Charset cs ) throws IOException {

		Matrix rc = null ;
		String SEPARATOR_CHARS = ",;\t" ; 

		try( Reader rdr = new InputStreamReader(is,  cs ) ;
				BufferedReader br = new BufferedReader(rdr) ; ) {
			String line = br.readLine() ;
			String regex = "\\," ;
			for( int i=0 ; i<line.length() ; i++ ) {
				if( SEPARATOR_CHARS.indexOf( line.charAt(i) ) > 0 ) {
					regex=String.valueOf( line.charAt(i) ) ;
					break ;
				}
			}
			
			String headers[] = line.split( regex ) ;
			int N = headers.length ;
			 
			for( int i=0 ; i<headers.length ; i++ ) {
				String col = headers[i].trim() ;
				if( col.charAt(0) == col.charAt(col.length()-1) && (col.charAt(0)=='"' || col.charAt(0)=='\'') ) {
					col = col.substring(1,col.length()-1 ) ;					
				}
				headers[i] = col ;
			}
			
			int m = 0 ;
			double reservoir[] = new double[M*N] ;
			log.info( "Found {} columns", N );
			
			@SuppressWarnings("unchecked")
			List<String> maps[] = new ArrayList[N] ;
			
			line = br.readLine() ;
			while( line != null ) {
				if( line.trim().length() == 0 ) continue ;
				String cols[] = line.split( regex ) ;
				double row[] = parse( cols, maps ) ;
				if( m==M ) { log.info( "Switching to reservoir mode. Keeping {} samples", M ) ; }
				if( m<M ) {
					for( int c=0 ; c<N ; c++ ) {
						reservoir[m + c*M] = row[c] ;
					}
				} else {
					int r = rng.nextInt(M) ;
					for( int c=0 ; c<N ; c++ ) {
						reservoir[r + c*M] = row[c] ;
					}
				}
				line=br.readLine() ;
				m++ ;
			}
			log.info( "Parsed {} lines", m ) ;
			rc = new Matrix( M, N, reservoir ) ;
			rc.reshape( Math.min(M, m),  N ) ;
			rc.labels = headers ;
		}
		return rc ;
	}

	public static void saveToCsv( int M, double data[], Path file ) throws IOException {
		DecimalFormat df = new DecimalFormat( "#.000000" ) ;

		try ( BufferedWriter bw = Files.newBufferedWriter( file ) ) {			
			int N = data.length / M ;
			for( int i=0 ; i<M ; i++ ) {
				StringJoiner line = new StringJoiner(",") ;
				for( int j=0 ; j<N ; j++ ) {
					int ix = j * M + i ;
					line.add( df.format(data[ix]) ) ;
				}
				bw.write( line.toString() );
				bw.newLine() ;
			}			
		}		
	}

	static String REGEX_NUMERIC =  "[+-]?[\\d\\.\\,]+" ;
	private static double buildMap( String col, List<String> map ) {
		String icol =
		( col.charAt(0) == col.charAt(col.length()-1) && (col.charAt(0)=='"' || col.charAt(0)=='\'') ) ?
				col.substring(1,col.length()-1 ).intern() :
				col.intern()
				;
		double rc = map.indexOf( icol ) ;
		if( rc < 0 ) {
			rc = map.size() ;
			map.add( icol ) ;
		}
		return rc ;
	}
	
	private static double[] parse( String [] cols, List<String>[] maps ) {
		double rc[] = new double[ cols.length ] ;
		for( int i=0 ; i<rc.length ; i++ ) {
			try {
				rc[i] = Double.parseDouble( cols[i] ) ;
			} catch ( Throwable t ) {
				List<String> map = maps[i] ;
				if( map == null ) {
					map = new ArrayList<>() ;
					maps[i] = map ;
				}
				rc[i] = buildMap(cols[i], map ) ;
			}
		}
		return rc ;
	}
}
