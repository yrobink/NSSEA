
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

## matrix_sqrt {{{

#' matrix_sqrt 
#'
#' Return the square root of a matrix M, i.e. the solution S of S %*% S = M
#'
#' @usage matrix_sqrt(M)
#'
#' @param M [matrix] a matrix
#'
#' @return Mroot [matrix] Square root of M
#'
#' @examples
#'
#' M  = matrix( base::c(4,0,0,2) , nrow = 2 )
#' Mroot = matrix_sqrt(M) ## = [2,0,0,sqrt(2)]
#' 
#' @export
matrix_sqrt = function(M)
{
	Md = base::eigen(M)
	return( base::Re(Md$vectors) %*% base::diag( base::sqrt( base::pmax( base::Re(Md$values) , 0 ) ) ) %*% base::t( base::Re(Md$vectors) ) )
}
##}}}

## matrix_positive_part {{{

#' matrix_positive_part 
#'
#' Return the positive part of a matrix
#'
#' @usage matrix_positive_part(M)
#'
#' @param M [matrix] a matrix
#'
#' @return Mp [matrix] Positive part of M
#'
#' @examples
#'
#' M  = matrix( base::c(1,0,0,-1) , nrow = 2 ) ## Not positive
#' Mp = matrix_positive_part(M) ## = [1,0,0,0] just the positive part
#' 
#' @export
matrix_positive_part = function(M)
{
	eigen = base::eigen(M)
	return( eigen$vectors %*% base::diag( base::pmax(eigen$values,0) ) %*% base::t(eigen$vectors) )
}
##}}}

## ginv {{{

#' ginv (Moore-Penrose generalized inverse of a matrix)
#'
#' Compute the Moore-Penrose generalized inverse of a matrix, code from MASS package
#'
#' @usage ginv(X,tol)
#'
#' @param X [matrix] a matrix
#' @param tol [double] Numerical tolerance
#'
#' @return Xinv [matrix] Generalized inverse
#'
#' @examples
#'
#' X = matrix( base::c(0,0,1,1) , nrow = 2 ) ## Not invertible (det==0)
#' Xinv = ginv(X) ## But generalized inverse exist
#' 
#' @export
ginv = function( X , tol = base::sqrt(.Machine$double.eps) )
{
	if( length(dim(X)) > 2L || !(is.numeric(X) || is.complex(X)) )
		stop("'X' must be a numeric or complex matrix")
	
	if( !is.matrix(X) )
		X = as.matrix(X)
	
	Xsvd = svd(X)
	
	if( is.complex(X) )
		Xsvd$u = base::Conj(Xsvd$u)
	
	Positive = ( Xsvd$d > base::max( tol * Xsvd$d[1L] , 0 ) )
	
	out = NULL
	if( base::all(Positive) )
	{
		out = Xsvd$v %*% ( 1. / Xsvd$d * base::t(Xsvd$u) )
	}
	else if( !base::any(Positive) )
	{
		out = array( 0 , dim(X)[2L:1L] )
	}
	else
	{
		out = Xsvd$v[, Positive, drop = FALSE] %*% ( ( 1. / Xsvd$d[Positive] ) * base::t( Xsvd$u[, Positive, drop = FALSE] ) )
	}
	return(out)
}
##}}}

## Barycenter_covariance {{{

#' barycenter_covariance
#'
#' Compute the covariance matrix of the barycenter of Gaussian law (in optimal transport), the process is iterative and stop when criteria is lower than tol or after maxit iterations.
#'
#' @usage barycenter_covariance( lcov , weights , maxit , tol , verbose )
#'
#' @param lcov [array] Array of covariance matrix with dimension number of cov * dim * dim (lcov[i,,] is a cov matrix)
#' @param weights [vector] Vector of weights. If NULL, all covariance matrix have the same weight
#' @param maxit [integer] Maximum number of iteration, default 50.
#' @param tol [double] Tolerance to stop iteration. Default is 1e-3
#' @param verbose [bool] print or not state of execution
#'
#' @return cov [matrix] The covariance matrix barycenter of lcov weighted by weights
#'
#' @examples
#'
#' lcov = array( 0 , dim = base::c(2,3,3) )
#' base::diag(lcov[1,,]) = 1
#' base::diag(lcov[2,,]) = 5
#' covB = barycenter_covariance(lcov)
#' 
#' @export
barycenter_covariance = function( lcov , weights = NULL , maxit = 50 , tol = 1e-3 , verbose = FALSE )
{
	n_cov = dim(lcov)[1]
	
	if( is.null(weights) )
		weights = base::rep( 1. , n_cov )
	weights = weights / base::sum(weights)
	
	brower = function( S , lcov , weights )
	{
		root_S = NSSEA::matrix_sqrt(S)
		Sn = matrix( 0 , nrow = dim(S)[1] , ncol = dim(S)[2] )
		
		for( i in 1:n_cov )
		{
			Sn = Sn + weights[i] * NSSEA::matrix_sqrt( root_S %*% lcov[i,,] %*% root_S )
		}
		
		return(Sn)
	}
	
	cov  = base::diag( base::rep( 1. , dim(lcov)[2] ) )
	diff = 1. + tol
	nit  = 0
	
	while( diff > tol && nit < maxit )
	{
		if(verbose) cat( "Optimal barycenter, nit:" , as.character(nit) , "( < maxit =" , as.character(maxit) , "), diff:" ,  as.character(base::round(diff,4)) , "( > tol =" , as.character(base::round(tol,4)) , ")         " , sep = " " , end = "\r" )
		covn = brower( cov , lcov , weights )
		diff = base::sqrt( base::sum( (cov - covn)^2 ) )
		cov  = covn
		nit = nit + 1
	}
	if(verbose) cat( "Optimal barycenter (Done)             \n" )
	
	return(cov)
}
##}}}

