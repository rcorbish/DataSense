package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MlpDataProcessor extends DataProcessor {
	final static Logger log = LoggerFactory.getLogger( MlpDataProcessor.class ) ;

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
		
		if( options.normalize ) {
			dataset.normalize();
		}

		return dataset ;
	}

	
	public Object process( Dataset dataset ) {
		
		Matrix A  = dataset.train ;
		Matrix T  = dataset.test ;

		// value in dataset -> zero based index
		Map<Integer,Integer> featureKeys = dataset.getFeatureKeys() ;
		Map<Integer,Integer> inverseFeatureKeys = new HashMap<>() ;
		for( Entry<Integer, Integer> e :featureKeys.entrySet() ) {
			inverseFeatureKeys.put( e.getValue(), e.getKey() ) ;
		}
		int feature = dataset.getFeatureColumnIndex() ;

		Matrix F = A.extractColumns( feature ) ;
		Matrix YR = T.extractColumns( feature ) ; 

		int numInputs = A.N ;
		int numOutputs = featureKeys.size() ;
		
		MultiLayerPerceptron nn = new MultiLayerPerceptron(TransferFunctionType.TANH, numInputs, numInputs, numInputs, numOutputs );
		// learn the training set
		DataSet trainingSet = new DataSet(numInputs, numOutputs);
		for( int i=0 ; i<A.M ; i++ ) {
			double outputs[] = new double[numOutputs] ;
			int f = (int)Math.floor( F.get(i)+0.5 ) ;
			outputs[ inverseFeatureKeys.getOrDefault(f, 0) ] = 1 ;
			trainingSet.addRow(new DataSetRow(A.copyRows(i).data, outputs ) );
		}
		BackPropagation lr = new MomentumBackpropagation() {
			int epoch = 0 ;
			@Override
			protected void beforeEpoch() {
				epoch++ ;
				if( epoch > 300 ) {
					setMaxError( 1000 ) ;
				} else {
					setMaxError( 0.0003 ) ;
				}
				super.beforeEpoch() ;
				setMinErrorChange( 0.001 ) ;
				setMinErrorChangeIterationsLimit( 3 ) ;
				setMaxIterations( 300 ) ;
				if( epoch > 200 ) {
					setLearningRate( 0.01 ) ;
				} else if( epoch > 100 ) {
					setLearningRate( 0.02 ) ;
				} else  {
					setLearningRate( 0.03 ) ;
				}
			
				if( !Double.isFinite( previousEpochError ) ) {
					previousEpochError = 0 ; 
				}
				log.info( "Epoch {} ; error {} ", epoch, previousEpochError ) ;
			}
		} ;
		nn.learn(trainingSet, lr );

		Matrix Y = new Matrix( T.M, 1 ) ;
		for( int i=0 ; i<T.M ; i++ ) {
			nn.setInput( T.copyRows(i).data ) ;
			nn.calculate();
			double d[] = nn.getOutput() ;
			double mx = d[0] ;
			int f = 0 ;
			for( int j=1 ; j<d.length ; j++ ) {
				if( d[j] > mx ) {
					mx = d[j] ;
					f = j ;
				}
			}
			Y.put( i, f ) ;
		}
		return score(YR, Y, inverseFeatureKeys ) ;
	}	
}


