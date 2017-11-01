
abstract public class Fmincg {

    abstract public double cost( DoubleMatrix X ) ;
    abstract public double gradient( DoubleMatrix X ) ;
    
public solve( DoubleMatrix X, int maxIters, int maxEpochs ) {
// % Read options
// if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
//     length = options.MaxIter;
// else
//     length = 100;
// end


// RHO = 0.01;                            % a bunch of constants for line searches
// SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
// INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
// EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
// MAX = 20;                         % max 20 function evaluations per line search
// RATIO = 100;                                      % maximum allowed slope ratio

double RHO = 0.01;                            //% a bunch of constants for line searches
double SIG = 0.5;       //% RHO and SIG are the constants in the Wolfe-Powell conditions
double INT = 0.1;    //% don't reevaluate within 0.1 of the limit of the current bracket
double EXT = 3.0;                    //% extrapolate maximum 3 times the current bracket
double MAX = 20;                         //% max 20 function evaluations per line search
double RATIO = 100;                                      //% maximum allowed slope ratio

// argstr = ['feval(f, X'];                      % compose string used to call function
// for i = 1:(nargin - 3)
//   argstr = [argstr, ',P', int2str(i)];
// end
// argstr = [argstr, ')'];

// will ignore len reduction - set = 1.0
//if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
double red = 1.0 ;
//S=['Iteration '];

// i = 0;                                            % zero the run length counter
int iterations = 0;                  //                          % zero the run length counter
int epochs = 0 ;
// ls_failed = 0;                             % no previous line search has failed
boolean ls_failed = false ; //                % no previous line search has failed
// fX = [];

//[f1 df1] = eval(argstr);                      % get function value and gradient

DoubleMatrix df1 = gradient(X) ;
double f1 = cost(X) ;

epochs++; //                                           % count epochs?!
//s = -df1;                                        % search direction is steepest
DoubleMatrix s = df1.mul(-1) ;

//d1 = -s'*s;                                                 % this is the slope
double d1 = -s.dot(s) ;

//z1 = red/(1-d1);                                  % initial step is red/(|s|+1)
double z1 = red / ( 1.0 - d1 ) ;

//while i < abs(length)                                      % while not finished
while( iterations < maxIters ) { //                                     % while not finished
    // i = i + (length>0);                                      % count iterations?!
    iterations++ ;

//   X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
  DoubleMatrix X0 = X.dup() ; 
  double f0 = f1; 
  DoubleMatrix df0 = df1;

//   X = X + z1*s;                                             % begin line search
  X.addi( z1*s ) ; 

//   [f2 df2] = eval(argstr);
    DoubleMatrix df2 = gradient(X) ;
    double f2 = cost(X) ;
  
//   i = i + (length<0);                                          % count epochs?!
    epochs++ ;

// d2 = df2'*s;
    DoubleMatrix d2 = df2.transpose().mmul(s);

// f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
    double f3 = f1 ;
    double d3 = d1 ;
    double z3 = -z1 ;

// if length>0, M = MAX; else M = min(MAX, -length-i); end
    double M = Math.min( MAX, -maxIters-i ) ;

    // success = 0; limit = -1;                     % initialize quanteties
    boolean success = false ;
    double limit = -1 ;

    double z2 ;
    // while 1
    while( true ) {
    // while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
        while( ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) ) { 
            // limit = z1;                                         % tighten the bracket
            limit = z1;                                     //    % tighten the bracket
            if( f2 > f1 ) {
                z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);   //              % quadratic fit
            } else {
                double A = 6*(f2-f3)/z3+3*(d2+d3);             //            % cubic fit
                double B = 3*(f3-f2)-z3*(d3+2*d2);
                z2 = (Math.sqrt(B*B-A*d2*z3*z3)-B)/A;      // % numerical error possible - ok!
            }
        
        //   if isnan(z2) || isinf(z2)
        //     z2 = z3/2;                  % if we had a numerical problem then bisect
        //   end
            if( !Double.isFinite(z2) ) {
                z2 = z3 / 2.0 ;
            }

        //   z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
            z2 = Math.max( Math.min( z2, INT*z3),(1-INT*z3) ) ;
        // z1 = z1 + z2;                                           % update the step
            z1 += z2 ;

        // X = X + z2*s;
        X.addi( s.mul(z2) ) ;

        //   [f2 df2] = eval(argstr);
        df2 = gradient(X) ;
        f2 = cost(X) ;
    
        // M = M - 1; i = i + (length<0);                           % count epochs?!
            M-- ;
            epochs++ ;

        // d2 = df2'*s;
            d2 = df2.transpose().mmul( s ) ;

        // z3 = z3-z2;                    % z3 is now relative to the location of z2
            z3 -= z2 ;
        // end
        }

    // if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
    //   break;                                                % this is a failure
    // elseif d2 > SIG*d1
    //   success = 1; break;                                             % success
    // elseif M == 0
    //   break;                                                          % failure
    // end
        if( f2 > f1+z1*RHO*d1  || d2>-SIG*d1 ) {
            break ;
        } else if( d2>SIG*d1 ) {
            success = true ;
            break ;
        } else if ( M == 0 ) {
            break ;
        }
        double A = 6*(f2-f3)/z3+3*(d2+d3);  //                     % make cubic extrapolation
        double B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = -d2*z3*z3/(B+Math.sqrt(B*B-A*d2*z3*z3));  //      % num. error possible - ok!
        // if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 // % num prob or wrong sign?
        if( !Double.isFinite(z2) || z2<0 ) {
        // if limit < -0.5                               % if we have no upper limit
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
    // f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
        f3 = f2 ;
        d3 = d2 ;
        z3 = -z2 ;
    // z1 = z1 + z2; X = X + z2*s;                      % update current estimates
        z1 += z2 ;
        X.addi( s.mul(z2) ) ;
        // [f2 df2] = eval(argstr);
        df2 = gradient(X) ;
        f2 = cost(X) ;

    // M = M - 1; i = i + (length<0);                             % count epochs?!
        M-- ;
        epochs++ ;
    // d2 = df2'*s;
        d2 = df2.transpose().mmuli(s) ;
    } // end                                                    % end of line search

//   if success                                         % if line search succeeded
    jf( success ) { 
// f1 = f2; fX = [fX' f1]';
        f1 = f2 ;
    // fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
    // s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
        s = ( df2.transpose() * df2.sub( df1.transpose() ).mmul( df2 ) /
            df1.transpose().mmul( df1 ).mmuli( s ) ).subi( df2 ) ;
    // tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
        DoubleMatrix tmp = df1 ;
        df1 = df2 ;
        df2 = tmp ;

        // d2 = df1'*s;
        d2 = df1.dot(s) ;
    // if d2 > 0                                      % new slope must be negative
    //   s = -df1;                              % otherwise use steepest direction
    //   d2 = -s'*s;    
    // end
        if( d2>0 ) {
            s = df1.mul( -1 ) ;
            d2 = -s.dot(s) ;
        }
    // z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
        z1 *= min( RATIO, d1/(d2-???) ) ;

    d1 = d2;
    // ls_failed = 0 ;                              % this line search did not fail
    ls_failed = false ;   //                           % this line search did not fail
    } else {
    // X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
        X = X0 ;
        f1 = f0 ;
        df1 = df0 ;
    // if ls_failed || i > abs(length)          % line search failed twice in a row
    //   break;                             % or we ran out of time, so we give up
    // end
    if( ls_failed || iterations>maxIters ) {
        break ;
    }
    // tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    DoubleMatrix tmp = df1 ;
    df1 = df2 ;
    df2 = tmp ;
    // s = -df1;                                                    % try steepest
    s = df1.mul( -1 ) ;
    // d1 = -s'*s;
    d1 = s.dot( s ) ;
    // z1 = 1/(1-d1);
    z1 = 1.0 / (1.0 - d1 ) ;                     
    // ls_failed = 1;                                    % this line search failed
    ls_failed = true ;
}
    }
}