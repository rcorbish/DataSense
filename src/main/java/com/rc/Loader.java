package com.rc;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import au.com.bytecode.opencsv.CSVReader;

public class Loader {
	final static Logger log = LoggerFactory.getLogger( Loader.class ) ;

	private static final Random rng = new Random(7) ;
	
	
	private Loader() {
		throw new UnsupportedOperationException( "Loader is a singleton" ) ;
	}
	

	/**
	 * If we're asked to map ordinal columns to individual discrete columns
	 * this does that work.
	 * 
	 * e.g. instead of a column having values 1,2,3,4 or 5
	 * we create 5 columns of IS_1, IS_2, IS_3, IS_4 & IS_5 
	 * which can take values 0 or 1
	 * 
	 * @param in will be altered by this process
	 * @param maps 
	 * @param feature do NOT remap feature column to multi cols
	 * 
	 * @return a new Matrix containing remapped columns 
	 * @throws IOException
	 */
	public static Matrix makeDiscreteColumns( Matrix in, List<String> maps[], int feature ) throws IOException {

		Matrix rc = in ;

		for( int c=0 ; c<maps.length ; c++ ) {
			List<String> map = maps[c] ;
			if( map == null || c==feature ) {
				continue ;
			}

			log.info( "Creating {} buckets for {}", map.size(), rc.labels[c] ) ;
			
			String labels[] = new String[ map.size()] ;
			for( int i=0 ; i<labels.length ; i++ ) {
				labels[i] = rc.labels[c] + "-is-" + map.get(i) ;
			}
			Matrix B = Matrix.fill( rc.M, map.size(), 0.0, labels ) ;
			for( int i=0 ; i<rc.M ; i++ ) {
				int n = (int)rc.get(i,c) ;
				B.put( i, (int)n, 1.0 ) ;
			}
			log.info( "Adding {} columns to data", B.N ) ;
			rc = rc.appendColumns( B ) ;
		}
		
		return rc ;
	}
	
	public static void verifyMappedColumns( Matrix m, List<String> maps[] ) throws IOException {

		for( int c=0 ; c<maps.length ; c++ ) {
			List<String> map = maps[c] ;
			if( map == null ) {
				continue ;
			}

			double mappedDataLimit = -1e14 + map.size()  ;
			
			// Scan the rows for a real numeric value, which could be mixed in
			// with the mapped ordinals. e.g. if we had values 1,2,YES,NO the 1 & 2 
			// would not be recognized as non numerics
			//
			// we address that here
			for( int i=0 ; i<m.M ; i++ ) {
				double n = m.get(i,c) ;
				if( n>mappedDataLimit ) {			// we found a real number mixed in - let's map it now
					String col = String.valueOf( m.get( i, c) ) ;
					m.put( i, c, mapToDouble( col, map) + 1e14 ) ;
				} else {
					m.put( i, c, Math.round(n + 1e14) ) ;
				}
			}
		}
	}
	
	public static Dataset load( int dataM, InputStream is, ProcessorOptions options ) throws IOException {

		Dataset rc = null ;
		
		try( Reader rdr = new InputStreamReader(is, options.cs ) ;
				CSVReader reader = new CSVReader(rdr) ; ) {
			
			String headers[] = reader.readNext() ;
			int N = headers.length ;
			 
			int m = 0 ;
			int M = (int)( (1.0+options.testRatio)*dataM ) ;
			double reservoir[] = new double[M*N] ;
			log.info( "Found {} columns", N );
			
			@SuppressWarnings("unchecked")
			List<String> maps[] = new ArrayList[N] ;
			
			String cols[] = reader.readNext() ;
			while( cols != null ) {
				if( cols.length > 1 ) {
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
					m++ ;
				}
				cols = reader.readNext() ;
			}
			log.info( "Parsed {} lines", m ) ;
			Matrix all = new Matrix( M, N, reservoir ) ;
			all.reshape( Math.min(M, m),  N ) ;
			all.labels = headers ;

			verifyMappedColumns( all, maps ) ;

			rc = new Dataset( all, new Matrix(0, N, headers ) ) ;
			int testRows = (int)Math.floor( all.M * options.testRatio ) ;
			
			log.info( "Dataset has {} total and {} test data rows", m, testRows ) ;
			int feature = rc.getFeatureColumnIndex() ;
			
			if( options!=null && options.discrete ) {
				rc.train = makeDiscreteColumns( rc.train, maps, feature ) ;
				rc.test.labels = rc.train.labels.clone() ;
				rc.test.N = rc.train.N ;
		   	}

			for( int i=0 ; i<testRows ; i++ ) {
				int testRow = rng.nextInt( rc.train.M ) ;
				Matrix test = rc.train.extractRows( testRow ) ;
				rc.test = rc.test.appendRows( test ) ;
			}
			  
		} // end try()

		return rc ;
	}

	public void saveToCsv( int M, double data[], Path file ) throws IOException {
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
	private static double mapToDouble( String col, List<String> map ) {
		if( col.length()==0 ) {
			return - 1e14 ;
		}
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
		return rc - 1e14 ;
	}
	
	private static double[] parse( String [] cols, List<String> maps[] ) {
		double rc[] = new double[ cols.length ] ;
		// for each column
		for( int i=0 ; i<rc.length ; i++ ) {
			// if we've seen non numeric - let's map it to an ordinal
			if(  maps[i] != null ) {
				rc[i] = mapToDouble(cols[i], maps[i] ) ;
			} else { // else we've only seen valid numbers				
				try {
					rc[i] = Double.parseDouble( cols[i] ) ;
				} catch ( Throwable t ) {
					Date d = DateColumnProcessor.parse(cols[i]) ;
					if( d != null ) {
						rc[i] = d.getTime()/1000 ;
					} else {
						// we just found a non-numeric, start mapping... 
						List<String> map = new ArrayList<>() ;
						maps[i] = map ;
						// and get an ordinal
						rc[i] = mapToDouble(cols[i], map ) ;
					}
				}
			}
		}
		return rc ;
	}
	
}
