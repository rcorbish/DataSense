package com.rc;

import java.io.IOException;
import java.io.InputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogisticDataProcessor extends DataProcessor implements com.rc.Cgd.CostFunction, com.rc.Cgd.GradientsFunction {
	final static Logger log = LoggerFactory.getLogger( LogisticDataProcessor.class ) ;

	final static String FEATURE_LABELS[] = { "Score", "Feature", "Result" } ;

	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		Dataset dataset = Loader.load( 1000, data, options ) ;

		if( options.square ) {
			Matrix A = dataset.train.dup() ;
			A.map( (value, context, r, c) ->  value * value ) ;			
			A.prefixLabels( "sqr " ) ;
			dataset.train = dataset.train.appendColumns(A) ;

			Matrix T = dataset.test.dup() ;
			T.hmuli( T ) ;			
			T.prefixLabels( "sqr " ) ;
			dataset.test = dataset.test.appendColumns(T) ;
		}

		Matrix A = Matrix.fill( dataset.train.M, 1,  1.0, "bias" ) ;
		dataset.train = A.appendColumns( dataset.train ) ;

		Matrix T = Matrix.fill( dataset.test.M, 1,  1.0, "bias" ) ;
		dataset.test = T.appendColumns(dataset.test) ;

		return dataset ;
	}

	
	public Object process( Dataset dataset ) {
		Object rc = null ;

		Matrix A  = dataset.train ;
		Matrix T  = dataset.test ;

		int feature = 0 ;

		featureSearch:
			for( int i=0 ; i<FEATURE_LABELS.length ; i++ ) {
				for( int j=0 ; j<A.N; j++ ) {
					if( FEATURE_LABELS[i].equalsIgnoreCase( A.labels[j] ) ) {
						feature = j ;
						break featureSearch ;
					}
				}
			}
		Matrix F = A.extractColumns( feature ) ;
		int numBuckets = (int) Math.floor( F.countBuckets( 1 ).get( 0 ) ) ;
		
		Matrix theta = new Matrix( A.N, numBuckets ) ;
		
		double lambda = 0.00001 ;
		int maxIterations = 500 ;
		Cgd cgd = new Cgd() ;
		for( int i=0 ; i<numBuckets ; i++ ) {
			Matrix y = F.oneIfEquals( i, 1e-2 ) ;
			Matrix t = cgd.solve(this::cost, this::grad, A, y, lambda, maxIterations ) ;
			theta.putColumn( i, t.data ) ;
		}
		
		
		Matrix YR = T.extractColumns( feature ) ; 

		Matrix Y = T.mmul(theta) ;
		Y.map( v -> sigmoid(v) ) ;
		Y = Y.maxIndexOfRows();
		Y.labels = new String[] { "Predicted" } ;
		
		Matrix YE = Y.sub(YR).map( (value, context, r, c) ->  value * value  ) ;
		YE.labels = new String[] { "MSE" } ;
		rc = new Dataset( A, Y.appendColumns( YR ).appendColumns(YE) ) ;

		return rc ;
	}


	@Override
	public Matrix grad(Matrix X, Matrix y, Matrix theta, double lambda) {
		
		Matrix ht = X.mmul( theta ) ;
		ht.map( v -> sigmoid(v) ) ;

		Matrix G1 = X.hmul( ht.subi( y ) ).sum().muli( 1.0/y.length() ) ;
		Matrix G2 = theta.mul( lambda/y.length() ) ;
		G2.put(0,  0.0 ); 
		
		return G2.addi( G1 ) ;
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
