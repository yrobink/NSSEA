
##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, Aurelien Ribes, 2020                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
## aurelien.ribes@meteo.fr                                                      ##
##                                                                              ##
## This software is a computer program that is part of the NSSEA                ##
## (Non-Stationary Statistics for Extreme Attribution) This library makes it    ##
## possible to infer the probability of an (extreme) event in the factual /     ##
## counter-factual world (without anthropic forcing) to attribute it to climate ##
## change.                                                                      ##
##                                                                              ##
## This software is governed by the CeCILL-C license under French law and       ##
## abiding by the rules of distribution of free software.  You can  use,        ##
## modify and/ or redistribute the software under the terms of the CeCILL-C     ##
## license as circulated by CEA, CNRS and INRIA at the following URL            ##
## "http://www.cecill.info".                                                    ##
##                                                                              ##
## As a counterpart to the access to the source code and  rights to copy,       ##
## modify and redistribute granted by the license, users are provided only      ##
## with a limited warranty  and the software's author,  the holder of the       ##
## economic rights,  and the successive licensors  have only  limited           ##
## liability.                                                                   ##
##                                                                              ##
## In this respect, the user's attention is drawn to the risks associated       ##
## with loading,  using,  modifying and/or developing or reproducing the        ##
## software by the user in light of its specific status of free software,       ##
## that may mean  that it is complicated to manipulate,  and  that  also        ##
## therefore means  that it is reserved for developers  and  experienced        ##
## professionals having in-depth computer knowledge. Users are therefore        ##
## encouraged to load and test the software's suitability as regards their      ##
## requirements in conditions enabling the security of their systems and/or     ##
## data to be ensured and,  more generally, to use and operate it in the        ##
## same conditions as regards security.                                         ##
##                                                                              ##
## The fact that you are presently reading this means that you have had         ##
## knowledge of the CeCILL-C license and that you accept its terms.             ##
##                                                                              ##
##################################################################################
##################################################################################

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, Aurelien Ribes, 2020                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
## aurelien.ribes@meteo.fr                                                      ##
##                                                                              ##
## Ce logiciel est un programme informatique faisant partie de la librairie     ##
## NSSEA (Non-Stationary Statistics for Extreme Attribution). Cette librairie   ##
## permet d'estimer la probabilité d'un evenement (extreme) dans le monde       ##
## factuel / contre factuel (sans forcage anthropogenique) et de l'attribuer au ##
## changement climatique.                                                       ##
##                                                                              ##
## Ce logiciel est régi par la licence CeCILL-C soumise au droit français et    ##
## respectant les principes de diffusion des logiciels libres. Vous pouvez      ##
## utiliser, modifier et/ou redistribuer ce programme sous les conditions       ##
## de la licence CeCILL-C telle que diffusée par le CEA, le CNRS et l'INRIA     ##
## sur le site "http://www.cecill.info".                                        ##
##                                                                              ##
## En contrepartie de l'accessibilité au code source et des droits de copie,    ##
## de modification et de redistribution accordés par cette licence, il n'est    ##
## offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,    ##
## seule une responsabilité restreinte pèse sur l'auteur du programme, le       ##
## titulaire des droits patrimoniaux et les concédants successifs.              ##
##                                                                              ##
## A cet égard  l'attention de l'utilisateur est attirée sur les risques        ##
## associés au chargement,  à l'utilisation,  à la modification et/ou au        ##
## développement et à la reproduction du logiciel par l'utilisateur étant       ##
## donné sa spécificité de logiciel libre, qui peut le rendre complexe à        ##
## manipuler et qui le réserve donc à des développeurs et des professionnels    ##
## avertis possédant  des  connaissances  informatiques approfondies.  Les      ##
## utilisateurs sont donc invités à charger  et  tester  l'adéquation  du       ##
## logiciel à leurs besoins dans des conditions permettant d'assurer la         ##
## sécurité de leurs systèmes et ou de leurs données et, plus généralement,     ##
## à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.           ##
##                                                                              ##
## Le fait que vous puissiez accéder à cet en-tête signifie que vous avez       ##
## pris connaissance de la licence CeCILL-C, et que vous en avez accepté les    ##
## termes.                                                                      ##
##                                                                              ##
##################################################################################
##################################################################################

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

