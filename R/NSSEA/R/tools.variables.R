
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

## make_coffee {{{

#' make_coffee
#'
#' Build the coffee variable containing results of execution of NSSEA
#' 
#' @importFrom NSModel NSGaussianModel
#'
#' @usage make_coffee( time , n_sample , models , ns_law , ns_law_arg )
#'
#' @param time [vector] Vector of time values
#' @param n_sample [integer] Number of sample drawn for confidence interval
#' @param models [vector of str] Names of models
#' @param ns_law [NSModel::NS**Model] Statistical non stationary model from library NSModel
#' @param ns_law_arg [NULL or list] Parameters of NSModel, see library NSModel
#'
#' @return coffee [list] The coffee variable, see example
#'
#' @examples
#'
#' coffee = make_coffee( 1:10 , 20 , base::c( "mod0" , "mod1" ) , NSModel::NSGaussianModel )
#' coffee$X           ## Will contain covariables after GAM decomposition
#' coffee$time        ## time,
#' coffee$n_time      ## length(time),
#' coffee$n_sample    ## n_sample,
#' coffee$models      ## models,
#' coffee$n_models    ## length(models),
#' coffee$
#' coffee$ns_law      ## ns_law (here gaussian)
#' coffee$ns_law_arg  ## arguments for ns_law
#' coffee$ns_params   ## will be non stationary parameters estimated after ns_fit
#' coffee$n_ns_params ## how many ns params by sample and model
#' coffee$
#' coffee$stats       ## will be statistics estimated (as pall, pnat, iall, RR, FAR etc.)
#' coffee$n_stats     ## will be the numbers of statistics fitted, 6 in general
#' coffee$
#' coffee$mm_params   ## will be the numbers of multimodel parameters
#' 
#' @export
make_coffee = function( time , n_sample , models , ns_law , ns_law_arg = NULL )
{
	coffee = list(
		X           = NULL,
		time        = time,
		n_time      = length(time),
		n_sample    = n_sample,
		models      = models,
		n_models    = length(models),
		
		ns_law      = ns_law,
		ns_law_arg  = ns_law_arg,
		ns_params   = NULL,
		n_ns_params = NULL,
		
		stats       = NULL,
		n_stats     = NULL,
		
		mm_params   = NULL
	)
	
	return(coffee)
}
##}}}

## make_event {{{

#' make_event
#'
#' Build the event variable containing information about event considered
#' 
#' @usage make_event( name , dir_name , time , anom , ref_anom , var , side )
#'
#' @param name [str] Name of event
#' @param dir_name [str] directory of output
#' @param time [time] time of event
#' @param anom [double] anomaly of event
#' @param ref_anom [vector] time period to considered as reference for anomaly
#' @param var [str] name of variable
#' @param side [str] "upper" or "lower" extremes event
#'
#' @return event [list] The event variable, see example
#'
#' @examples
#'
#' event = make_event( "EHW03" , "output" , 2003 , 5. , 1961:1990 , "tas" , "high" )
#' 
#' event$name		# Name of event
#' event$dir_name	# directory of output
#' event$time		# time of event
#' event$anom		# anomaly of event
#' event$ref_anom	# time period to considered as reference for anomaly
#' event$var		# name of variable
#' event$side		# "upper" or "lower" extremes event
#' 
#' @export
make_event = function( name , dir_name , time , anom , ref_anom , var , side )
{
	event = list(
				name     = name,
				dir_name = dir_name,
				def_type = "threshold",
				time     = time,
				anom     = anom,
				ref_anom = ref_anom,
				var      = var,
				side     = side
				)
	return(event)
}
##}}}

## make_CXParams {{{

#' make_CXParams
#'
#' Build the CXParams variable containing parameters for CX constraints
#' 
#' @usage make_CXParams( centering , ref , trust )
#'
#' @param centering [bool] if we need to center with respect to ref
#' @param ref [time] reference period for constraints CX
#' @param trust [bool] if the covariance matrix of observed covariate is assumed TRUE or FALSE
#'
#' @return cx_params [list] The CX params variable, see example
#'
#' @examples
#'
#' cx_params = make_CXParams( TRUE , 1961:1990 , FALSE )
#' print( cx_params$centering )
#' print( cx_params$ref )
#' print( cx_params$trust )
#' 
#' @export
make_CXParams = function( centering , ref , trust )
{
	return( list( centering = centering , ref = ref , trust = trust ) )
}
##}}}


