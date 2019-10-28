
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

## nslaw_fit {{{

#' nslaw_fit
#'
#' Fit non stationary parameters.
#'
#' @usage nslaw_fit( lY , coffee , verbose , code )
#'
#' @param lY [list] list of models
#' @param coffee [coffee variable]
#' @param verbose [bool] print or not state of execution
#' @param code [str] code of "yrobin" or "aribes"
#'
#' @return coffee [coffee] same coffee with ns_params
#'
#' @examples
#' ##
#' 
#' @export
nslaw_fit = function( lY , coffee , verbose = FALSE , code = "yrobin" )
{
	if( code == "yrobin" )
	{
		return( NSSEA::nslaw_fit_yrobin( lY , coffee , verbose ) )
	}
	else
	{
		return( NSSEA::nslaw_fit_aribes( lY , coffee , verbose ) )
	}
}
##}}}

## nslaw_fit_yrobin {{{

#' nslaw_fit_yrobin
#'
#' Fit non stationary parameters
#'
#' @usage nslaw_fit_yrobin( lY , coffee , verbose )
#'
#' @param lY [list] list of models
#' @param coffee [coffee variable]
#' @param verbose [bool] print or not state of execution
#'
#' @return coffee [coffee] same coffee with ns_params
#'
#' @examples
#' ##
#' 
#' @export
nslaw_fit_yrobin = function( lY , coffee , verbose = FALSE )
{
	## Parameters
	ns_law       = coffee$ns_law
	ns_law_arg   = ns_law$public_methods$default_arg(coffee$ns_law_arg)
	info_params  = ns_law$public_methods$params_info(ns_law_arg)
	params_names = info_params$names
	n_ns_params  = info_params$n
	n_sample     = coffee$n_sample
	n_models     = coffee$n_models
	tX = coffee$time
	
	## Initialize NS_param
	coffee$ns_params = array( NA ,
					dim = base::c( n_ns_params , n_sample + 1 , n_models ),
					dimnames = list( params = params_names , sample = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) ) , models = coffee$models )
					)
	
	## Main loop
	pb = NSSEA::ProgressBar$new( "NS fit" , n_models * n_sample )
	for( i in 1:n_models )
	{
		## Extract
		Y    = lY[[i]]
		tY   = as.numeric(names(Y))
		X    = coffee$X[ match(tY,tX) ,"be","all",i]
		n_tY = length(tY)
		Y    = as.vector(Y)
		
		## Fit itself
		law = base::do.call( ns_law$new , ns_law_arg )
		law$fit( Y , X )
		coffee$ns_params[,"be",i] = law$get_params()
		
		## Loop on sample
		for( j in 1:n_sample )
		{
			if(verbose) pb$print()
			
			idx = base::sample( 1:n_tY , n_tY , replace = TRUE )
			Ys  = Y[idx]
			Xs  = coffee$X[match(tY[idx],tX),j+1,"all",i]
			
			law = base::do.call( ns_law$new , ns_law_arg )
			law$fit( Ys , Xs )
			coffee$ns_params[,j+1,i] = law$get_params()
		}
	}
	
	coffee$n_ns_params = n_ns_params
	
	if(verbose) pb$end()
	
	return(coffee)
}
##}}}

## nslaw_fit_aribes {{{

#' nslaw_fit_aribes
#'
#' Fit non stationary parameters
#'
#' @usage nslaw_fit_aribes( lY , coffee , verbose )
#'
#' @param lY [list] list of models
#' @param coffee [coffee variable]
#' @param verbose [bool] print or not state of execution
#'
#' @return coffee [coffee] same coffee with ns_params
#'
#' @examples
#' ##
#' 
#' @export
nslaw_fit_aribes = function( lY , coffee , verbose = FALSE )
{
	NS_param = NSSEA::gauss_fit_full( lY , coffee$X , verbose )
	coffee$ns_params   = NS_param
	coffee$n_ns_params = 4
	return(coffee)
}
##}}}



