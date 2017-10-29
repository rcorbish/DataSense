package com.rc;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

public class SvmcDataProcessor extends DataProcessor implements svm_print_interface {
	final static Logger log = LoggerFactory.getLogger( SvmcDataProcessor.class ) ;

	public Dataset load( InputStream data, ProcessorOptions options ) throws IOException {
		Dataset dataset = Loader.load( 1000, data, options ) ;

		if( options.square ) {
			dataset.square(); 
		}		
		
		if( options.log ) {
			dataset.log(); 
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
//		F.map( v -> featureKeys.get((int)v) ) ;
		Matrix YR = T.extractColumns( feature ) ; 
		
		Matrix means = A.mean() ;
		Matrix stddev = A.stddev(means) ;
		
		for( int i=0 ; i<A.M ; i++ ) {
			for( int j=0 ; j<A.N ; j++ ) {
				A.put( i, j, ( A.get( i, j ) - means.get(j) ) / stddev.get(j) ) ; 
			}			
		}
		for( int i=0 ; i<T.M ; i++ ) {
			for( int j=0 ; j<T.N ; j++ ) {
				T.put( i, j, ( T.get( i, j ) - means.get(j) ) / stddev.get(j) ) ; 
			}			
		}
		
		svm_node data[][] = new svm_node[A.M][] ;
		for( int i=0 ; i<A.M ; i++ ) {
			data[i] = new svm_node[A.N] ;
			for( int j=0 ; j<A.N ; j++ ) {
				data[i][j] = new svm_node() ;
				data[i][j].index = j + 1 ;
				data[i][j].value = A.get(i, j) ;
			}
		}
		
		svm_problem problem = new svm_problem() ;
		problem.x = data ;
		problem.y = F.data.clone() ;
		problem.l = F.length() ; 
		svm_parameter parameter = new svm_parameter() ;
		parameter.svm_type = svm_parameter.C_SVC ;
		parameter.kernel_type = svm_parameter.RBF ;
		parameter.degree = 2 ;	// poly only
		parameter.gamma = 0.001 ;	// exp( -gamma * (u-v)^2 )
		parameter.coef0 = 0 ; 	// sigmoid / poly only
		parameter.cache_size = 100 ;	// MB
		parameter.C = 10 ; // normal C parameter in svm
		parameter.eps = 0.0001 ; // error to stop
		parameter.nr_weight = 0 ; // all weights equal
		parameter.weight_label = null ; // all weights equal
		parameter.weight = null ; // all weights equal
		parameter.nu = 0 ; // NU models only
		parameter.p = 0 ; // EPSILON regression
		parameter.shrinking = 0 ; // no shrinking heuristics
		parameter.probability = 0 ; // no probability estimates
				
		String checkResult = svm.svm_check_parameter(  problem,  parameter ) ;
		if( checkResult != null ) {
			log.warn( "SVM-C check: {}", checkResult ) ;
		}
		
		svm.svm_set_print_string_function( this::print ) ;

		svm_model model = svm.svm_train( problem,  parameter ) ;

		Matrix Y = new Matrix( T.M, 1 ) ;

		svm_node test[] = new svm_node[T.N] ;
		for( int i=0 ; i<T.M ; i++ ) {
			for( int j=0 ; j<T.N ; j++ ) {
				test[j] = new svm_node() ;
				test[j].index = j + 1 ;
				test[j].value = T.get(i, j) ;
			}
			Y.put(i, svm.svm_predict( model, test )  ) ;
		}
		

		int numBuckets = model.nr_class ;		

		Y.labels = new String[] { "Predicted" } ;

		SvmcResults rc = new SvmcResults() ;

		Matrix precision = new Matrix( numBuckets, 1 ) ;
		Matrix recall = new Matrix( numBuckets, 1 ) ;
		
		for( int n=0 ; n<numBuckets ; n++ ) {
			int f = inverseFeatureKeys.get( n ) ;

			int nfp = 0 ;
			int ntp = 0 ;
			int nfn  = 0 ;
			for( int i=0 ; i<Y.length() ; i++ ) {
				int yn = (int)Y.get(i) ;
				int yrn = (int)YR.get(i) ;
				if( yrn == f && yn == f ) {	// true positives
					ntp++ ;
				} else if( yrn != f && yn == f ) {	// false positives
					nfp++ ;
				} else if( yrn == f && yn != f ) {	// false negatives
					nfn++ ;
				}
			}
			precision.put( n, (double)ntp / (double)(ntp + nfp) ) ;
			recall.put( n, (double)ntp / (double)(ntp + nfn) ) ;
		}
		rc.precision = precision ;
		rc.recall = recall ;
		rc.f1 = precision.hmul( recall ).muli(2.0).hdivi( precision.add( recall ) ) ;
		
		return rc ;
	}
	
	public void print(String s)	{
		if( s.length() > 1 ) {
			log.info( s.trim() ) ;
		}
	};
}

class SvmcResults {
	Matrix precision ;
	Matrix recall ;
	Matrix f1 ;
}
