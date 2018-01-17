
package com.rc ;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Cgd {

	final static Logger log = LoggerFactory.getLogger( Cgd.class ) ;

	@FunctionalInterface
	static interface CostFunction {
		public double cost( Matrix X, Matrix y, Matrix theta, double lambda ) ;
	}
	@FunctionalInterface
	static interface GradientsFunction {
		public Matrix grad( Matrix X, Matrix y, Matrix theta, double lambda ) ;
	}
	@FunctionalInterface
	static interface ProgressFunction {
		public void progress( double score, int iteration ) ;
	}

	private ProgressFunction progress ;
	public Cgd() {
		this( null ) ;
	}
	public Cgd( ProgressFunction progress ) {
		this.progress = progress ;
	}
	
	public Matrix solve( 
			CostFunction cost, 
			GradientsFunction gradients, 
			Matrix Xin, Matrix yin,
			double lambda, int maxIters ) {

		log.info( "Start conjugate gradient descent" ) ;

		// theta starts at 0,0,0...0
		Matrix theta = Matrix.rand( Xin.N, yin.N ) ;
		
		double RHO = 0.01;                           
		double SIG = 0.25;	// RHO and SIG are the constants in the Wolfe-Powell conditions
		double INT = 0.1;   // don't reevaluate within 0.1 of the limit of the current bracket
		double EXT = 3.0;   // extrapolate maximum ext times the current bracket
		double MAX = 30;    // max evaluations per line search
		double RATIO = 50;  // maximum allowed slope ratio

		double red = 1.0 ;	

		int numConsecutiveSearchFails = 0 ; //  no previous line search has failed

		Matrix df1 = gradients.grad(Xin, yin, theta, lambda) ;
		double f1 = cost.cost(Xin, yin, theta, lambda) ;
		Matrix s = df1.mul(-1) ;

		double d1 = -s.dot() ; 
		double z1 = red / ( 1.0 - d1 ) ;

		int iterations = 0;   
		while( iterations < maxIters ) { 
			iterations++ ;

			Matrix theta0 = theta.dup() ; 
			double f0 = f1; 
			Matrix df0 = df1 ;

			theta.addi( s.mul(z1) ) ; 

			Matrix df2 = gradients.grad(Xin, yin, theta, lambda) ;
			double f2 = cost.cost(Xin, yin, theta, lambda) ;

			double d2 = df2.dot(s);

			double f3 = f1 ;
			double d3 = d1 ;
			double z3 = -z1 ;

			double M = MAX ;

			boolean success = false ;
			double limit = -1 ;

			double z2 ;

			// Line search - find descent  
			while( true ) {
				
				// Strong Wolfe condition test - find longest line in best direction
				while( ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) ) { 
					limit = z1;                                     	// tighten the bracket
					
					if( f2 > f1 ) {
						z2 = z3 - ( 0.5*d3*z3*z3 )/( d3*z3 + f2 - f3 ); // quadratic fit
					} else {
						double A = 6*(f2-f3)/z3 + 3*(d2+d3);           	// cubic fit
						double B = 3*(f3-f2) - z3*( d3 + 2*d2 );
						z2 = ( Math.sqrt( B*B - A*d2*z3*z3 ) - B ) / A;      
					}
					if( !Double.isFinite(z2) ) {
						z2 = z3 / 2.0 ;
					}

					z2 = Math.max( Math.min( z2, INT*z3),(1-INT)*z3 ) ;
					z1 += z2 ;

					theta.addi( s.mul(z2) ) ;

					df2 = gradients.grad(Xin, yin, theta, lambda)  ;
					f2 = cost.cost(Xin, yin, theta, lambda)  ;

					d2 = df2.dot( s ) ;
					z3 -= z2 ;

					M-- ;
				}

				// exit line search ?
				if( f2 > f1+z1*RHO*d1 || d2>-SIG*d1) {
					break ;
				} else if( d2>SIG*d1 ) {
					success = true ;
					break ;
				} else if( M == 0 ) {
					break ;
				}
				
				double A = 6*(f2-f3)/z3+3*(d2+d3);  // cubic extrapolation
				double B = 3*(f3-f2)-z3*(d3+2*d2);
				z2 = -d2*z3*z3/(B+Math.sqrt(B*B-A*d2*z3*z3));  
				
				if( !Double.isFinite(z2) || z2<0 ) {
					if( limit < -0.5 ) {
						z2 = z1 * (EXT-1);		// maximum extrapolation
					} else {
						z2 = (limit-z1)/2;		// otherwise bisect
					}
				} else if( (limit > -0.5) && (z2+z1 > limit) ) { 	// extrapolated past max?
					z2 = (limit-z1)/2;   							// bisect
				} else if( (limit < -0.5) && (z2+z1 > z1*EXT) ) { 	// extrapolation past max? 
					z2 = z1*(EXT-1.0);      						// set to max
				} else if( z2 < -z3*INT ) { 
					z2 = -z3*INT;
				} else if( (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT)) ) {   // too close to limit?
					z2 = (limit-z1)*(1.0-INT);
				}

				f3 = f2 ;
				d3 = d2 ;
				z3 = -z2 ;
				
				z1 += z2 ;
				theta.addi( s.mul(z2) ) ;
				
				df2 = gradients.grad(Xin, yin, theta, lambda)  ;
				f2 = cost.cost(Xin, yin, theta, lambda)  ;

				M-- ;
				d2 = df2.dot(s) ;
			} // end of line search

			if( success ) { 
				numConsecutiveSearchFails = 0 ; 

				f1 = f2 ;

				double factor = ( df2.dot() - df2.dot( df1 ) ) / df1.dot() ;				
				s.muli( factor ).subi( df2 ) ;
				
				Matrix tmp = df1 ;
				df1 = df2 ;
				df2 = tmp ;

				d2 = df1.dot(s) ;
				if( d2>0 ) {
					s = df1.mul( -1 ) ;
					d2 = -s.dot() ;
				}

				z1 *= Math.min( RATIO, d1/(d2-Double.MIN_VALUE ) ) ;
				d1 = d2;
			} else {
				numConsecutiveSearchFails ++ ;
				// reset to previous guess
				theta 	= theta0 ;
				f1 		= f0 ;
				df1 	= df0 ;

				if( numConsecutiveSearchFails>1 || iterations>maxIters ) {
					log.debug( "CGD stopped converging - early exit" ) ;
					break ;
				}

				Matrix tmp = df1 ;
				df1 = df2 ;
				df2 = tmp ;
				
				s = df1.mul( -1 ) ;
				d1 = -s.dot() ;
				z1 = 1.0 / (1.0 - d1 ) ;                     
			}
			
			if( progress != null ) {
				progress.progress( f1, iterations) ;
			}
		} // main iteration loop
		log.info( "Conjugate gradient descent completed in {} iterations", iterations ) ;

		return theta ;
	}
	
	public double checkGrad( CostFunction cost, GradientsFunction gradients ) {

		Matrix X = Matrix.rand(1, 10) ;
		Matrix y = Matrix.rand(1, 1 ) ;
		Matrix theta = Matrix.rand(10, 1) ;
		double lambda = 0 ;
		double e = 1e-5 ;
		Matrix origGrad = gradients.grad(X, y, theta, lambda) ;
		Matrix finiteGrad = new Matrix( origGrad.M, origGrad.N ) ;
		
		for( int i=0 ; i<X.N ; i++ ) {
			Matrix dx = Matrix.fill( 10, 1, 0) ;
			dx.put( i, 1e-5 ) ;
			double y2 = cost.cost(X, y, theta.add(dx), lambda) ;
			double y1 = cost.cost(X, y, theta.sub(dx), lambda) ;
			finiteGrad.put( i, (y2-y1) / (2*e) ) ;
		}
		
		double diff = finiteGrad.sub( origGrad ).norm() /  origGrad.add( finiteGrad ).norm();
		log.debug( "Check grad diff is {}", diff ) ;
		return diff ;
	}
}