package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogisticDataProcessor extends DataProcessor implements com.rc.Cgd.CostFunction, com.rc.Cgd.GradientsFunction {
	final static Logger log = LoggerFactory.getLogger( LogisticDataProcessor.class ) ;

	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		Dataset dataset = Loader.load( DataProcessor.ROWS_TO_KEEP, data, options ) ;

		if( options.square ) {
			dataset.square( options.keepOriginal );
		}		
		
		if( options.log ) {
			dataset.log( options.keepOriginal );
		}
		
		if( options.reciprocal ) {
			dataset.reciprocal( options.keepOriginal );
		}
				
		dataset.addBias() ;

		return dataset ;
	}

	
	public Object process( Dataset dataset ) {

		Matrix A  = dataset.train ;
		Matrix T  = dataset.test ;
		int feature = dataset.getFeatureColumnIndex() ;

		// value in dataset -> zero based index
		Map<Integer,Integer> featureKeys = dataset.getFeatureKeys() ;
		Map<Integer,Integer> inverseFeatureKeys = new HashMap<>() ;
		for( Entry<Integer, Integer> e :featureKeys.entrySet() ) {
			inverseFeatureKeys.put( e.getValue(), e.getKey() ) ;
		}

		Matrix F = A.extractColumns( feature ) ;

		int numBuckets = featureKeys.size() ; // (int) Math.floor( F.countBuckets( 1 ).get( 0 ) ) ;
		
		Matrix theta = new Matrix( A.N, numBuckets ) ;
		
		double lambda = 0.01 ;
		int maxIterations = 500 ;
		Cgd cgd = new Cgd() ;
		for( int i=0 ; i<numBuckets ; i++ ) {
			int f = inverseFeatureKeys.get(i) ;
			Matrix y = F.oneIfEquals( f, 1e-2 ) ;
			Matrix t = cgd.solve(this::cost, this::grad, A, y, lambda, maxIterations ) ;
			theta.putColumn( i, t.data ) ;
		}
				
		Matrix YR = T.extractColumns( feature ) ; 

		Matrix Y = T.mmul(theta) ;
		Y.map( v -> sigmoid(v) ) ;
		Y = Y.maxIndexOfRows() ;
		
		return score( YR, Y, inverseFeatureKeys )  ;
	}


	@Override
	public Matrix grad(Matrix X, Matrix y, Matrix theta, double lambda) {
		
		Matrix ht = X.mmul( theta ) ;
		ht.map( v -> sigmoid(v) ) ;

		Matrix G1 = X.hmul( ht.subi( y ) ).sum().muli( 1.0/y.length() ) ;
		Matrix G2 = theta.mul( lambda/y.length() ) ;
		G2.put(0,  0.0 ); 
		
		return G2.addi( G1.transpose() ) ;
	}
	
	@Override
	public double cost(Matrix X, Matrix y, Matrix theta, double lambda) {
		Matrix ht = X.mmul( theta ) ;
		ht.map( v -> sigmoid(v) ) ;
		Matrix loght = ht.dup().map( v -> Math.log(v) ) ;
		Matrix loght2 = ht.dup().map( v -> Math.log(1.0-v) ) ;

		double ts = theta.total() - theta.get(0) ;

		double J = loght.hmuli( y.mul(-1) ).subi( loght2.hmuli( y.mul(-1).add(1) ) ).total() / y.length() ;
		J += lambda * ts * ts / (2 * y.length() ) ;
		
		return J;
	}
	
		
	protected double sigmoid( double z ) {
		return  1.0 / ( Math.exp(-z) + 1 ) ;
	}
}

