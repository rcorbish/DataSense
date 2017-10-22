
package com.rc ;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Fmincg {

	final static Logger log = LoggerFactory.getLogger( Fmincg.class ) ;

	@FunctionalInterface
	static interface CostFunction {
		public double call( Matrix X, Matrix y, Matrix theta, double lambda ) ;
	}
	@FunctionalInterface
	static interface GradientsFunction {
		public Matrix call( Matrix X, Matrix y, Matrix theta, double lambda ) ;
	}


	public Matrix solve( CostFunction cost, GradientsFunction gradients, 
			Matrix Xin, Matrix yin,
			double lambda, int maxIters ) {

		log.info( "Start conjugate gradient descent" ) ;

// 		Add bias to X
//		DoubleMatrix Xcopy = new DoubleMatrix( X.rows, X.columns + 1 ) ;
//		for( int i=0 ; i<X.rows ; i++ ) {
//			for( int j=0 ; j<X.columns ; j++ ) {
//				Xcopy.put( i,  j+1, X.get( i,j ) ) ;
//			}
//			Xcopy.put( i,0, 1.0 ) ;
//		}
		//X = Xcopy ;
		// Init theta to 0
		
		Matrix theta = new Matrix( Xin.N, yin.N ) ;
		
		double RHO = 0.01;                           
		double SIG = 0.5;       //% RHO and SIG are the constants in the Wolfe-Powell conditions
		double INT = 0.1;    //% don't reevaluate within 0.1 of the limit of the current bracket
		double EXT = 3.0;                    //% extrapolate maximum 3 times the current bracket
		double MAX = 20;                         //% max 20 function evaluations per line search
		double RATIO = 100;                                      //% maximum allowed slope ratio

		double red = 1.0 ;

		// i = 0;                                            % zero the run length counter
		int iterations = 0;                  //                          % zero the run length counter

		boolean ls_failed = false ; //                % no previous line search has failed
		// fX = [];

		//[f1 df1] = eval(argstr);                      % get function value and gradient

		Matrix df1 = gradients.call(Xin, yin, theta, lambda) ;
		double f1 = cost.call(Xin, yin, theta, lambda) ;

		//s = -df1;                                        % search direction is steepest
		Matrix s = df1.mul(-1) ;

		//d1 = -s'*s;                                                 % this is the slope
		double d1 = -s.dot(s) ;

		//z1 = red/(1-d1);                                  % initial step is red/(|s|+1)
		double z1 = red / ( 1.0 - d1 ) ;

		while( iterations < maxIters ) { //                                     % while not finished
			iterations++ ;

			Matrix 		theta0 = theta.dup() ; 
			double 		f0 = f1; 
			Matrix	 	df0 = df1 ;


			theta.addi( s.mul(z1) ) ; 

			Matrix 		df2 = gradients.call(Xin, yin, theta, lambda) ;
			double 		f2 = cost.call(Xin, yin, theta, lambda) ;

			double d2 = df2.dot(s);

			double f3 = f1 ;
			double d3 = d1 ;
			double z3 = -z1 ;

			double M = MAX ;

			boolean success = false ;
			double limit = -1 ;

			double z2 ;

			while( true ) {
				while( ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) ) { 

					limit = z1;                                     	// tighten the bracket
					if( f2 > f1 ) {
						z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);   		// quadratic fit
					} else {
						double A = 6*(f2-f3)/z3+3*(d2+d3);             	// cubic fit
						double B = 3*(f3-f2)-z3*(d3+2*d2);
						z2 = (Math.sqrt(B*B-A*d2*z3*z3)-B)/A;      		// numerical error possible - ok!
					}
					if( !Double.isFinite(z2) ) {
						z2 = z3 / 2.0 ;
					}

					z2 = Math.max( Math.min( z2, INT*z3),(1-INT)*z3 ) ;
					z1 += z2 ;

					theta.addi( s.mul(z2) ) ;

					df2 	= gradients.call(Xin, yin, theta, lambda)  ;
					f2 		= cost.call(Xin, yin, theta, lambda)  ;

					d2 		= df2.dot( s ) ;
					z3 	   -= z2 ;
					M-- ;
				}

				if( d2>SIG*d1 ) {
					success = true ;
					break ;
				} else if( f2 > f1+z1*RHO*d1  || d2>-SIG*d1 || M == 0 ) {
					break ;
				}
				
				double A = 6*(f2-f3)/z3+3*(d2+d3);  //                     % make cubic extrapolation
				double B = 3*(f3-f2)-z3*(d3+2*d2);
				z2 = -d2*z3*z3/(B+Math.sqrt(B*B-A*d2*z3*z3));  //      % num. error possible - ok!
				
				if( !Double.isFinite(z2) || z2<0 ) {
					if( limit < -0.5 ) {
						z2 = z1 * (EXT-1);     //            % the extrapolate the maximum amount
					} else {
						z2 = (limit-z1)/2;   //                                % otherwise bisect
					}
				} else if( (limit > -0.5) && (z2+z1 > limit) ) { //         % extraplation beyond max?
					z2 = (limit-z1)/2;   //                                             % bisect
				} else if( (limit < -0.5) && (z2+z1 > z1*EXT) ) { //       % extrapolation beyond limit
					z2 = z1*(EXT-1.0);      //                     % set to extrapolation limit
				} else if( z2 < -z3*INT ) { //
					z2 = -z3*INT;
				} else if( (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT)) ) { //  % too close to limit?
					z2 = (limit-z1)*(1.0-INT);
				}

				f3 = f2 ;
				d3 = d2 ;
				z3 = -z2 ;
				
				z1 += z2 ;
				theta.addi( s.mul(z2) ) ;
				
				df2 = gradients.call(Xin, yin, theta, lambda)  ;
				f2 = cost.call(Xin, yin, theta, lambda)  ;

				M-- ;
				d2 = df2.dot(s) ;
			} // end of line search

			
			if( success ) { 
				// f1 = f2; fX = [fX' f1]';
				f1 = f2 ;

				// s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
//				Matrix df1t = df1.transpose() ;
//				Matrix df2t = df2.transpose() ;
				Matrix dn = s.mul( df1.dot( df1 ) ) ;
				Matrix up = new Matrix( 1, dn.M, new double[dn.M*dn.M] ) ;
				up.put( 0, 0, df2.dot( df2 ) - df1.dot( df2 ) );
				
				Matrix div = dn.divLeft( up ).transpose() ;   
				//Solve.solveLeastSquares( dn.transpose(), up.transpose() ) ; // ax = b  i.e. a\b  or
				s = div.subi( df2 ) ;
				
				Matrix tmp = df1 ;
				df1 = df2 ;
				df2 = tmp ;


				d2 = df1.dot(s) ;
				if( d2>0 ) {
					s = df1.mul( -1 ) ;
					d2 = -s.dot(s) ;
				}

				z1 *= Math.min( RATIO, d1/(d2-Double.MIN_VALUE ) ) ;

				d1 = d2;
				ls_failed = false ;   // this line search did not fail
			} else {
				theta 	= theta0 ;
				f1 		= f0 ;
				df1 	= df0 ;

				if( ls_failed || iterations>maxIters ) {
					break ;
				}

				Matrix tmp = df1 ;
				df1 = df2 ;
				df2 = tmp ;
				
				s = df1.mul( -1 ) ;
				d1 = s.dot( s ) ;
				z1 = 1.0 / (1.0 - d1 ) ;                     
				ls_failed = true ;
			}
		}
		log.info( "Conjugate gradient descent completed in {} iterations", iterations ) ;

		return theta ;
	}
}