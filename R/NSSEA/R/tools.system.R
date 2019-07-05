 
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

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
