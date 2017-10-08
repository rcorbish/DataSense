package com.rc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.StringJoiner;

public class Loader {

	private static final Random rng = new Random() ;
	
	public static double[] loadFromCsv( int M, Path file ) throws IOException {
		
		double reservoir[] = null ;
		
		int m = 0 ;
		
		try ( BufferedReader br = Files.newBufferedReader( file ) ) {
			String line = br.readLine() ;
			String headers[] = line.split( ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)" ) ;
			int N = headers.length ;
			
			reservoir = new double[M*N] ;
			
			while( (line=br.readLine()) != null ) {
				if( line.trim().length() == 0 ) continue ;
				String cols[] = line.split( ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)" ) ;
				double row[] = parse( cols ) ;
				if( m<M ) {
					for( int c=0 ; c<N ; c++ ) {
						reservoir[m + c*M] = row[c] ;
					}
					m++ ;
				} else {
					int r = rng.nextInt(M) ;
					for( int c=0 ; c<N ; c++ ) {
						reservoir[r + c*M] = row[c] ;
					}
				}				
			}
			
			return reservoir ;
		}
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
	
	private static double[] parse( String [] cols) {
		double rc[] = new double[ cols.length ] ;
		for( int i=0 ; i<rc.length ; i++ ) {
			rc[i] = Double.parseDouble( cols[i] ) ;
		}
		return rc ;
	}
}
