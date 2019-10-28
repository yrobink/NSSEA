
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

## gauss_fit_full {{{

#' gauss_fit_full
#'
#' Fit many Gaussian law
#'
#' @usage gauss_fit_full( Y , X , verbose )
#'
#' @param Y [array] Variables to fit
#' @param X [array] covariables
#' @param verbose [bool] print or not state of execution
#'
#' @return NSparam [coffee$ns_params]
#'
#' @examples
#' ##
#' 
#' @export
gauss_fit_full = function( Y , X , verbose = FALSE )
{

	# Get dimensions
	time     = as.numeric(dimnames(X)$time)
	n_time   = length(time)
	n_sample = dim(X)[2]-1
	models   = dimnames(X)$models
	n_models = length(models)
	
	# Initialize NS_param
	params_names = base::c("mu0","mu1","sig0","sig1")
	n_params     = length(params_names)
	
	NS_param = array( NA ,
					dim = base::c( n_params , n_sample + 1 , n_models ),
					dimnames = list( params = params_names , sample = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) ) , model = models )
					)
	
	 # Main loop
	pb = NSSEA::ProgressBar$new( "NS fit" , n_models * n_sample )
	for( imod in 1:n_models )
	{
		name_imod = models[imod]
		
		year_y          = as.numeric( names( Y[[imod]] ) )
		y               = as.numeric(Y[[imod]])
		nyf             = length(year_y)
		x_all_resampled = X[,-1,"all",imod]
		
		# NS-fit itself
		# Best-estimate
		x_nsfit = X[ match(year_y,time) ,"be","all",imod ]
		NS_param[,"be",imod] = gauss_fit_center( y , x_nsfit )
		
		# Resampling
		# Bootstrap of raw data Y (and year_Y)
		ipts        = base::sample( 1:nyf , n_sample * nyf , replace = TRUE )
		year_y_boot = matrix( as.numeric(year_y[ipts]) , nyf , n_sample )
		y_boot      = matrix( as.numeric(y[ipts]) , nyf , n_sample )
		for( ires in 1:n_sample )
		{
			pb$print()
			year_y_tmp = year_y_boot[,ires]
			x_all_tmp = x_all_resampled[match(year_y_tmp,time),ires]
			NS_param[,ires+1,imod] = gauss_fit_center(y_boot[,ires],x_all_tmp)
		}
	}
	pb$end()
	
	return(NS_param)
}
##}}}

## gauss_fit_center {{{

#' gauss_fit_center
#'
#' Center and fit a Gaussian law
#'
#' @usage gauss_fit_center( y , x )
#'
#' @param y [vector] Variables to fit
#' @param x [vector] covariables
#'
#' @return NSparam [coffee$ns_params[,"sample","model"]]
#'
#' @examples
#' ##
#' 
#' @export
gauss_fit_center = function(y, x)
{
	xm = base::mean(x)
	xc = x - xm
	ym = base::mean(y)
	yc = y - ym
	ns_fit = gauss_fit(y = yc, x = xc)    # The fit itself
	paramc = ns_fit    # = [mu_0,mu_1,sig_0,sig_1] for centered data
	param  = paramc
	
	# Correct mu_0 from centering effect
	#         E(y-ym) = mu_0 + mu_1 * (x-xm)
	# means     E(y) = mu_0 + ym - mu_1 * xm
	param[1] = paramc[1] + ym - paramc[2] * xm
	
	# Correct sig_0 from centering effect
	#        Sig(y-ym) = sig_0 + sig_1 * (x-xm)
	# means    Sig(y) = sig_0 - sig_1*xm + sig_1 * x
	param[3] = paramc[3] - paramc[4] * xm
	
	return(param)
}
##}}}

## gauss_fit {{{

#' gauss_fit
#'
#' Fit a Gaussian law
#' @importFrom stats coefficients
#' @importFrom stats lm.fit
#' @importFrom stats optim
#' @importFrom stats residuals
#'
#' @usage gauss_fit( y , x , init )
#'
#' @param y [vector] Variables to fit
#' @param x [vector] covariables
#' @param init [vector] starting point to optim
#'
#' @return NSparam [coffee$ns_params[,"sample","model"]]
#'
#' @examples
#' ##
#' 
#' @export
gauss_fit = function( y , x , init = NULL )
{
	
	# Design matrix
	design = cbind(1, x)
	
	# Gaussian log-likelihood for given this design
	gauss_lik = function(ns_param)
	{
		mu   = design %*% ns_param[1:2]
		sig  = design %*% ns_param[3:4]
		gauss_negll( y , mu , sig ) 
	}
	# debug(gauss_lik)
	
	# Set initial value for NS-parameters
	if(is.null(init))
	{
		
		y_fit    = lm.fit( x = design , y = y ) # Linear fit y(x)
		fit_res2 = residuals(y_fit)^2 # Linear fit of squared residuals: var(x)
		var_fit  = lm.fit(x = design, y = fit_res2) #fit_res2 <- sqrt(residuals(y_fit)^2)
		init     = c(coefficients(y_fit), coefficients(var_fit))
		
		# In case of "strong slope" on sigma (eg leading to negative values), correct sigma slope
		# The condition for "strong slope" is min(sig) < msig/2 (so we take a huge margin wrt 0)
		sig  = design %*% init[3:4]
		dsig = max(sig) - min(sig)
		msig = mean(range(sig))
		if( dsig > msig )
		{
			
			new_slope = init[4] / dsig * msig # Claculate new slope (decrease its abs value), Note: this preserves the sign of the slope...
			init[3] = init[3] + (new_slope - init[4]) * mean(range(x)) # Correct mean for change in slope
			init[4] = new_slope # Correct slope
		}
	}
	
	# Optimization loop
	y_fit = optim( par = init , gauss_lik )
	
	# Return result
	ns_param               = y_fit$par
	names(ns_param)        = c("mu0", "mu1", "sig0", "sig1")
	attr(ns_param, "npar") = 4
	
	return(ns_param)
}
##}}}

## gauss_negll {{{

#' gauss_negll
#'
#' Negative log-likelihood of a Gaussian law
#'
#' @usage gauss_negll( y , mu , sig )
#'
#' @param y [vector] Variables to fit
#' @param mu [vector] mean param
#' @param sig [vector] sigma param
#'
#' @return negll [double] negloglikelihood value
#'
#' @examples
#' ##
#' 
#' @export
gauss_negll = function( y , mu , sig )
{
	n = length(y)
	stopifnot(length(mu) == n & length(sig) == n)
	if( base::any( sig < 0 ) )
	{
		negll = Inf 
	}
	else
	{
		negll = n * log(2*pi) + sum( log( sig^2 ) ) + sum( ( y - mu )^2 / sig^2 )
	}
	return(negll)
}
##}}}


