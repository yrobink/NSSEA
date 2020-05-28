 
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

#library(R6)

## eval_str {{{

#' eval_str
#'
#' Execute a string
#' 
#' @usage eval_str(str)
#'
#' @param str [str] string of a function to execute
#'
#' @return results of execution
#'
#' @examples
#' str = "rnorm(3)" ## execution return 3 random gaussian numbers
#' eval_str(str) ## Good
#'
#' str = base::c("rnorm(" , "3" , ")") ## execution return 3 random gaussian numbers
#' eval_str(str) ## Good
#'
#' @export
eval_str = function(str)
{
	return( base::eval( base::parse( text = str ) ) )
}
##}}}

## ones {{{

#' ones
#'
#' Creates an array (/vector/matrix) with only 1 [and appropriate names / dimnames]
#' 
#' @usage ones(X)
#'
#' @param X [array] input array
#'
#' @return Xones [array]
#'
#' @examples
#' dim = base::c( 10 , 3 )
#' dimnames = list( dim1 = 1:10 , dim2 = base::c( -1,-2,-3 ) )
#' X  = array( stats::rnorm( 10 * 3 ) , dim = dim  , dimnames = dimnames )
#' Xo = ones(X)
#'
#' @export
ones = function(X) 
{
	Xones = X - X + 1
	Xones[is.na(Xones)] = 1
	return(Xones)
}
##}}}

## dropd {{{

#' dropd
#'
#' Drop requested dimensions of length 1 of an array
#' 
#' @usage dropd(X,drop)
#'
#' @param X [array] input array, with dim and dimnames
#'
#' @param drop [vector] dimensions to be droped
#'
#' @return Xd [array]
#'
#' @examples
#' X = array( NA , dim = base::c(3,1,4) )
#' Xd = dropd( X , drop = base::c(2) )
#'
#' @export
dropd = function( X , drop = base::c() ) ## 
{
	if( is.null( dim(X) ) )
	{
		base::warning("NSSEA::dropd: input array has no dimensions, input returned")
		return(X)
	}
	
	d = dim(X)
	
	# Check that dimensions to be dropped are all 1
	if( !base::all( d[drop] == 1 ) )
	{
		base::warning("NSSEA::dropd: dimensions to drop are > 1, input returned")
		return(X)
	}
	
	d_new = d[-drop]
	dn = dimnames(X)
	
	if( !is.null(dn) )
	{
		dn = dn[-drop]
	}
	
	return( array( X , dim = d_new , dimnames = dn ) )
}
##}}}

## str_list_int {{{

#' Formatted string vector of integers
#'
#' Transform a vector of integer to string of same size, i.e [0,1] is transformed to ["0","1"] but [0,1,3,10] is transformed to ["00","01","03","10"]
#' 
#' @usage str_list_int(l)
#' @param l [vector] vector of integers
#'
#' @return l_str [vector] vector of string
#'
#' @examples
#' l = 7:9 ## l = [7,8,9]
#' l_str = str_list_int(l) ## l_str = ["7","8","9"]
#' l = base::c(0,1,3,10)
#' l_str = str_list_int(l) ## l_str = ["00","01","03","10"]
#'
#' @export
str_list_int = function( l )
{
	l = base::floor(l)
	max_exp = base::floor( base::log10( base::max(l) ) )
	l_str = as.character(l)
	for( i in 1:max_exp )
	{
		zeros = base::paste0( base::rep(0,max_exp-i+1) , collapse = "" )
		idx = base::which( (base::log10(l) < i) & (base::log10(l) >= i - 1 ) )
		for( j in idx )
			l_str[j] = base::paste( zeros , l_str[j] , sep = "" )
	}
	
	return(l_str)
}
##}}}

## ProgressBar {{{

#' Progress Bar
#'
#' Tools to print a progress bar during a loop
#'
#' @docType class
#' @importFrom R6 R6Class
#'
#' @param message [str] message printed before the percentage
#'
#' @param size [integer] length of the loop
#'
#' @param digits [integer] number of digits printed for percentage, default 2.
#'
#' @return Object of \code{\link{R6Class}}
#' @format \code{\link{R6Class}} object.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new(message,size,digits)}}{This method is used to create object of this class with \code{ProgressBar}}
#'   \item{\code{init()}}{Re-initialize the percentage to 0}
#'   \item{\code{print()}}{print the current state}.
#'   \item{\code{end()}}{print the final state}.
#' }
#' @examples
#' size = 10
#' pb = ProgressBar$new( "Progress" , size )
#' for( i in 1:size )
#' 	pb$print()
#' pb$end()
#' 
#' @export
ProgressBar = R6::R6Class( "ProgressBar" ,
	public = list(
	
	message = NULL,
	size    = NULL,
	iter    = NULL,
	digits  = NULL,
	
	initialize = function( message , size , digits = 2 )
	{
		self$message = message
		self$size    = size
		self$digits  = digits
		self$iter    = 0
	},
	
	init = function()
	{
		self$iter = 0
	},
	
	print = function()
	{
		base::cat( self$message , " (", base::round( 100 * self$iter / self$size , digits = self$digits ) , "%...)                \r" , sep = "" )
		self$iter = self$iter + 1
	},
	
	end  = function()
	{
		base::cat( self$message , "(Done)                  \n" )
	}
	
	)
)
##}}}


## Not used, so commented

#max.which = function(X) ##{{{
#{
#	return( base::max( base::which(X) ) )
#}
##}}}

#min.which = function(X) ##{{{
#{
#	return( base::min( base::which(X) ) )
#}
##}}}

#nnames = function(X) ##{{{
#{
#	return(names(dimnames(X)))
#}
##}}}
