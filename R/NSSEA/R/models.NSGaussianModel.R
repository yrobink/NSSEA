
###############
## Libraries ##
###############

####################################################################################################
##
## Non-Stationnary Gaussian Model class
##
####################################################################################################

#' Non-stationary Gaussian Model
#'
#' Model a non stationary Gaussian law where non-stationarity is given by a co-variable.
#'
#' @docType class
#' @importFrom R6 R6Class
#' @importFrom SDFC NormalLaw
#' @importFrom SDFC IdLinkFct
#' @importFrom SDFC ExpLinkFct
#' 
#' @param Y [vector]
#'        Dataset inferred by the model
#' @param X [vector]
#'        Covariable of the dataset Y
#' @param t [vector]
#'        Time axis
#' @param q [vector]
#'        Quantiles in [0,1]
#' @param init [NULL or vector]
#'        Initialization for the optimization of the likelihood, if NULL, estimated during the fit.
#'
#' @return Object of \code{\link{R6Class}}
#' @format \code{\link{R6Class}} object.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new()}}{This method is used to create object of this class with \code{NSGaussianModel}}
#'   \item{\code{fit(Y,X,init)}}{Fit the model from Y and X}
#'   \item{\code{set_covariable(X,t)}}{Set the covariable of the model}
#'   \item{\code{mut(t)}}{Mean of the Gaussian law at time t}
#'   \item{\code{sigmat(t)}}{Standard devian of the Gaussian law at time t}
#'   \item{\code{rvs(t)}}{Random values generator at each time t}
#'   \item{\code{cdf(Y,t)}}{Cumulative Distribution Function of values Y at time t}
#'   \item{\code{icdf(q,t)}}{Inverse of CDF of quantiles values q at time t}
#'   \item{\code{sf(Y,t)}}{Survival Function (1-CDF) of values Y at time t}
#'   \item{\code{isf(q,t)}}{Inverse of sf of quantiles values q at time t}
#' }
#' @examples
#' ## Define a Gaussian dataset
#'
#' @export
NSGaussianModel = R6::R6Class( "NSGaussianModel" ,
	
	inherit = NSAbstractModel,
	
	
	public = list(
	
	###############
	## Arguments ##
	###############
	
	mu0     = NULL,
	mu1     = NULL,
	scale0  = NULL,
	scale1  = NULL,
	
	
	#################
	## Constructor ##
	#################
	
	initialize = function( link_fct_loc = IdLinkFct$new() , link_fct_scale = ExpLinkFct$new() , method = "MLE" ) ##{{{
	{
		self$norm = SDFC::NormalLaw( link_fct_loc = link_fct_loc , link_fct_scale = link_fct_scale , method = method )
	},
	##}}}
	
	default_arg = function( args = NULL ) ##{{{
	{
		da = list(  link_fct_loc = IdLinkFct$new() , link_fct_scale = ExpLinkFct$new() , method = "MLE" )
		if( !is.null(args) )
		{
			for( m in base::names(args) )
			{
				if( m %in% base::names(da) )
					da[[m]] = args[[m]]
			}
		}
		return( da )
	},
	##}}}
	
	params_info = function( args = NULL )##{{{
	{
		return( list( n = 4 , names = base::c( "mu0" , "mu1" , "sig0" , "sig1" ) ) )
	},
	##}}}
	
	
	###############
	## Accessors ##
	###############
	
	mut = function(t) ##{{{
	{ return(private$mut_(t)) },
	##}}}
	
	scalet = function(t) ##{{{
	{ return(private$scalet_(t)) },
	##}}}
	
	get_params = function() ##{{{
	{
		params = base::c( self$mu0 , self$mu1 , self$scale0 , self$scale1 )
		names(params) = base::c( "mu0" , "mu1" , "scale0" , "scale1" )
		return( params )
	},
	##}}}
	
	set_params = function( params ) ##{{{
	{
		self$mu0    = params[1]
		self$mu1    = params[2]
		self$scale0 = params[3]
		self$scale1 = params[4]
	},
	##}}}
	
	
	#############
	## Methods ##
	#############
	
	fit = function( Y , X ) ##{{{
	{
		norm$fit( Y , loc_cov = X , scale_cov = X )
		self$mu0    = norm$coef_[1]
		self$mu1    = norm$coef_[2]
		self$scale0 = norm$coef_[3]
		self$scale1 = norm$coef_[4]
	},
	##}}}
	
	set_covariable = function( X , t = NULL ) ##{{{
	{
		if( is.null(t) )
		{
			t = base::seq( 1 , length(X) )
		}
		
		private$mut_    = stats::approxfun( t , self$mu0 + X * self$mu1 )
		private$scalet_ = stats::approxfun( t , self$scale0 + X * self$scale1 )
	},
	##}}}
	
	rvs = function(t) ##{{{
	{
		return( stats::rnorm( length(t) , mean = private$mutfn(t) , sd = private$scaletfn(t) ) )
	},
	##}}}
	
	cdf = function( Y , t ) ##{{{
	{
		return( stats::pnorm( Y , mean = private$mutfn(t) , sd = private$scaletfn(t) ) )
	},
	##}}}
	
	icdf = function( q , t ) ##{{{
	{
		q[ !(q>0) ] = .Machine$double.eps
		q[ !(q<1) ] = 1. - .Machine$double.eps
		return( stats::qnorm( q , mean = private$mutfn(t) , sd = private$scaletfn(t) ) )
	},
	##}}}
	
	sf = function( Y , t ) ##{{{
	{
		return( stats::pnorm( Y , mean = private$mutfn(t) , sd = private$scaletfn(t) , lower.tail = FALSE ) )
	},
	##}}}
	
	isf = function( q , t ) ##{{{
	{
		return( stats::qnorm( q , mean = private$mutfn(t) , sd = private$scaletfn(t) , lower.tail = FALSE ) )
	},
	##}}}
	
	mean = function(t) ##{{{
	{
		return( self$mut(t) )
	}
	##}}}
	
	), ## Public list
	
	
	
	#############
	## Private ##
	#############
	
	private = list(
	
	
	#######################
	## Private arguments ##
	#######################
	
	mutfn    = NULL,
	scaletfn = NULL
	
	
	) ##Private list
) ## NSGaussianModel



