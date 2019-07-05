
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

#' ebm_response
#'
#' Compute n_sample EBM along time 
#'
#' @importFrom utils data
#'
#' @usage ebm_response( time , n_sample )
#'
#' @param time [vector] time
#' @param n_sample [integer]
#'
#' @return Enat [matrix] Natural forcing
#'
#' @examples
#' ##
#' 
#' @export
ebm_response = function( time , n_sample )
{
	hmodel = function( FF , c , c0 , lamb , gamm )
	{
		N     = length(FF)
		dt    = 1
		T     = numeric(N+1);
		To    = numeric(N+1);
		T[1]  = 0
		To[1] = 0
		
		for( n in 2:(N+1) )
		{
		    T[n]  =  ( T[n-1] + dt / c * ( FF[n-1] - lamb * T[n-1] - gamm * ( T[n-1] - To[n-1] ) ) );
		    To[n] =  ( To[n-1] + dt / c0 * ( gamm * ( T[n-1]-To[n-1] ) ) );
		}
		invisible(T[-1])
	}
	
	ebm_params = NULL ## Just here for devtools
	FF         = NULL ## Just here for devtools
	
	utils::data( "ebm_params" , envir = environment() )
	utils::data( "FF" , envir = environment() )
	
	n_params_ebm = length(ebm_params[[1]])
	n_time = length(time)
	
	# Calculate nat response for each set of params
	Enat_multi = matrix( NA , nrow = n_time , ncol = n_params_ebm )
	
	for( i in 1:n_params_ebm )
	{
		Enat_multi_1750 = hmodel( FF$FF$nat , ebm_params$c[i] , ebm_params$c0[i] , ebm_params$lamb[i] , ebm_params$gamm[i] )
		Enat_multi[,i] = Enat_multi_1750[FF$FF$year %in% time ]
	}
	
	# Enat -- Best-estimate and random resampling of responses
	Enat = array( NA , dim = base::c( n_time , n_sample + 1 ) , dimnames = list( time = time , sample = base::c( "be" , base::paste0( "sample" , 1:n_sample ) ) ) )
	iparam = base::sample( 1:n_params_ebm , n_sample , replace = TRUE )
	Enat[,1]  = base::apply( Enat_multi , 1 , base::mean )
	Enat[,-1] = Enat_multi[,iparam]
	
	return(Enat)
}



