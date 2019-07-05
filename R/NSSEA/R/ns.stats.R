
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

## extremes_stats {{{

#' extremes_stats
#'
#' Add to coffee the statistics as pall, pnat, RR/FAR, iall, inat and di
#'
#' @usage extremes_stats( coffee , event , threshold_by_world , verbose , code )
#'
#' @param coffee [coffee variable]
#' @param event [event_variable] 
#' @param threshold_by_world [bool] If FALSE, the treshold is the same for the factual and counter-factual world, otherwise different threshold is used 
#' @param verbose [bool] print or not state of execution
#' @param code [str] code of "yrobin" or "aribes"
#'
#' @return coffee [coffee] same coffee with stats
#'
#' @examples
#' ##
#' 
#' @export
extremes_stats = function( coffee , event , threshold_by_world = FALSE , verbose = FALSE , code = "yrobin" )
{
	if( code == "yrobin" )
	{
		return( extremes_stats_yrobin( coffee , event , threshold_by_world , verbose ) )
	}
	else
	{
		return( extremes_stats_aribes( coffee , event ) )
	}
}
##}}}

## extremes_stats_yrobin {{{

#' extremes_stats_yrobin
#'
#' Add to coffee the statistics as pall, pnat, RR/FAR, iall, inat and di
#'
#' @usage extremes_stats_yrobin( coffee , event , threshold_by_world , verbose )
#'
#' @param coffee [coffee variable]
#' @param event [event_variable]
#' @param threshold_by_world [bool] If FALSE, the treshold is the same for the factual and counter-factual world, otherwise different threshold is used 
#' @param verbose [bool] print or not state of execution
#'
#' @return coffee [coffee] same coffee with stats
#'
#' @examples
#' ##
#' 
#' @export
extremes_stats_yrobin = function( coffee , event , threshold_by_world = FALSE , verbose = FALSE )
{
	## Usefull variables
	time           = coffee$time
	n_time         = coffee$n_time
	models         = coffee$models
	n_models       = coffee$n_models
	n_sample       = coffee$n_sample
	n_stats        = 6
	upper_side     = event$side == "high"
	coffee$n_stats = n_stats
	event_time     = event$time
	
	
	## Names of statistics
	stats_names = base::c( "pnat" , "pall" , "rr" , "inat" , "iall" , "di" )
	
	## Output
	stats = array( NA,
				  dim = base::c( n_time , n_sample + 1 , n_stats , n_models ),
				  dimnames = list( time = time , sample = dimnames(coffee$X)$sample , stats = stats_names , models = models )
				  ) 
	
	
	## 
	ns_law_arg = coffee$ns_law$public_methods$default_arg(coffee$ns_law_arg)
	law        = base::do.call( coffee$ns_law$new , ns_law_arg )
	
	pb = NSSEA::ProgressBar$new( "Statistics" , n_models * (n_sample + 1) )
	for( i in 1:n_models )
	{
		
		for( j in 1:(n_sample+1) )
		{
			if(verbose) pb$print()
			## X        = ( n_time * sample * ("nat",all") * models )
			## NS_param = (n_ns_params * sample * models )
			
			law$set_params( coffee$ns_params[,j,i] )
			
			## Find threshold
			law$set_covariable( coffee$X[,j,"all",i] , time )
			threshold = numeric(n_time) + base::mean( law$mean(event$ref_anom) ) + event$anom
			
			## Find pall
			stats[,j,"pall",i] = if( upper_side ) law$sf( threshold , time ) else law$cdf( threshold , time )
			
			## Probability of the event
			pf = numeric(n_time) + ( if( upper_side ) law$sf( threshold[1] , event_time ) else law$cdf( threshold[1] , event_time ) )
			
			## I1
			stats[,j,"iall",i] = if( upper_side ) law$isf( pf , time ) else law$icdf( pf , time )
			
			## Find pnat
			law$set_covariable( coffee$X[,j,"nat",i] , time )
			if( threshold_by_world )
				threshold = numeric(n_time) + base::mean( law$mean(event$ref_anom) ) + event$anom
			
			stats[,j,"pnat",i] = if( upper_side ) law$sf( threshold , time ) else law$cdf( threshold , time )
			
			## I0
			stats[,j,"inat",i] = if( upper_side ) law$isf( pf , time ) else law$icdf( pf , time )
			
		}
		
	}
	
	pnat = stats[,,"pnat",]
	tol = base::min(pnat[ pnat > 0 ])
	pnat[ !(pnat>0) ] = tol
	
	## RR
	stats[,,"rr",] = stats[,,"pall",] / pnat#stats[,,"pnat",]
	
	## deltaI
	stats[,,"di",] = stats[,,"iall",] - stats[,,"inat",]
	
	coffee$stats = stats
	if(verbose) pb$end()
	
	return(coffee)
}
##}}}

## extremes_stats_aribes {{{

#' extremes_stats_aribes
#'
#' Add to coffee the statistics as pall, pnat, RR/FAR, iall, inat and di
#'
#' @usage extremes_stats_aribes( coffee , event )
#'
#' @param coffee [coffee variable]
#' @param event [event_variable] 
#'
#' @return coffee [coffee] same coffee with stats
#'
#' @examples
#' ##
#' 
#' @export
extremes_stats_aribes = function( coffee , event )
{
	diags = ns_param_to_diag( coffee$X , coffee$ns_params , event )
	coffee$stats = diags
	return(coffee)
}
##}}}

## ns_param_to_diag {{{

#' ns_param_to_diag
#'
#' Find statistics as pall, pnat, RR/FAR, iall, inat and di
#'
#' @importFrom stats setNames
#'
#' @usage ns_param_to_diag( X , NS_param , event , year_rel )
#'
#' @param X [array] coffee$X variable
#' @param NS_param [array] coffee$ns_params variable
#' @param event [event_variable] 
#' @param year_rel [vector] vector of time or NULL, do not use...
#'
#' @return d_of_t [array] coffee$stats variable
#'
#' @examples
#' ##
#' 
#' @export
ns_param_to_diag = function( X , NS_param , event , year_rel = NULL )
{
	
	# A few checks on X and NS_param
	if( base::any( dimnames(X)$models != dimnames(NS_param)$models ) )
		stop("Number of models does not match")
	
	n_models = dim(X)[4]
	if( n_models == 1 )
		models = dimnames(X)$models
	
	# Diagnoses
	#-----------
	nd = 6 #+ 2*length(year_rel)
	diag_names = base::c( "pnat" , "pall" , "rr" , "inat" , "iall" , "di" )
	d_of_t = array( NA , dim = base::c( dim(X)[1:2] , nd , dim(X)[4] ) ,
					dimnames = base::c( dimnames(X)[1:2] , list( stats = diag_names ) , dimnames(X)[4] )
					)
	
	years = as.numeric(dimnames(X)$time)
	
	# Gaussian case: parameters are (mu, sig)
	
	# Moving parameters
	#-------------------
	# Adapt NS_param (dim=np,Nres+1,Nmod) to the dimensions of X (dim=ny,Nres+1,2,Nmod)
	unit_tmp = NSSEA::ones(X[,1,,1])
	NS_param_conform = base::aperm( NS_param %o% unit_tmp , base::c(1,4,2,5,3) )
	
	mu_of_t  = NS_param_conform["mu0",,,,]  + NS_param_conform["mu1",,,,]  * X[,,,]
	sig_of_t = NS_param_conform["sig0",,,,] + NS_param_conform["sig1",,,,] * X[,,,]
	
	
	# Event definition
	#------------------
	# Adapt event def to models' world
	### Related question: 
	###	The calculation made is based on smoothed quantities mu_of_t[all].
	###	Is it correct? Should we only use mu_of_t_all_be (no uncertainty related to resampling)? Should we add some uncertainty related to noise around the (smooth) forced response?
	
	ref_Y = mu_of_t[as.character(event$ref_anom),,"all",]
	
	if( n_models == 1 )
		ref_Y = ref_Y %o% setNames( base::c(1) , models)	# Restore the "model" dimension
	
	absolute_thresh_model = base::apply( ref_Y , base::c(2,3) , base::mean ) + event$anom		# dim=(Nres+1,Nmod)
	lower.tail = !(event$side == "high")
	# Note: According to event$def_type, should we calculate proba or intensity first? if[event$def_type=="threshold"]{		??
	
	
	# Adapt dimensions of absolute_thresh_model
	abs_thresh_conform = base::aperm( NSSEA::ones(X[,1,,1]) %o% absolute_thresh_model , base::c(1,3,2,4) )		# dim=(ny,Nres+1,2,Nmod)
	
	## Compute p0 and p1
	# Note: lower.tail=F enables accurate calculations when the result is close to 0. Otherwise pnorm gets close to 1, and in fact indistinguishable from 1, so the estimated probability is exactly 0.
	d_of_t[,,c("pnat","pall"),] = stats::pnorm( abs_thresh_conform , mean = mu_of_t , sd = sig_of_t , lower.tail = lower.tail ) 
	d_of_t[,,"rr",] = d_of_t[,,"pall",] / d_of_t[,,"pnat",]
	# Intensity
	# Reference probability
	# Note : we do not calculate 1-p to avoid values = 1
	p_ref_model = d_of_t[as.character(event$time),,"pall",]	# dim=(Nres+1,Nmod)  # Check for other classes...
	if( n_models == 1 )
		p_ref_model = p_ref_model %o% setNames( base::c(1) , models )
	
	p_ref_conform = base::aperm( NSSEA::ones(X[,"be",,1] ) %o% p_ref_model , base::c(1,3,2,4) )	# dim=(ny,Nres+1,2,Nmod)		(ny,2)*(Nres+1,Nmod)
	
	# Note: this formula is related to the use of p instead of 1-p
	d_of_t[,,c("inat","iall"),] = 2 * mu_of_t - stats::qnorm( p_ref_conform , mean = mu_of_t , sd = sig_of_t )
	d_of_t[,,"di",] = d_of_t[,,"iall",] - d_of_t[,,"inat",]
	
	# Relative proba/intensity
	if (!is.null(year_rel))
	{
		for( i in 1:length(year_rel) )
		{
			name_diag_p = paste0("rel_p_",year_rel[i])
			name_diag_i = paste0("rel_i_",year_rel[i])
			d_of_t_relp = d_of_t[,,"pall",] / ( NSSEA::ones(years) %o% d_of_t[as.character(year_rel[i]),,"pall",] )
			
			if( n_models == 1 )
				d_of_t_relp = d_of_t_relp %o% setNames( base::c(1) , models )	# Restore model dimension
			
			d_of_t_reli = d_of_t[,,"iall",] - ( NSSEA::ones(years) %o% d_of_t[as.character(year_rel[i]),,"iall",] )
			
			if( n_models == 1 )
				d_of_t_reli = d_of_t_reli %o% setNames( base::c(1) , models )	# Restore model dimension
			
			d_of_t = abind::abind( d_of_t , d_of_t_relp , d_of_t_reli , along = 3 , use.dnns = TRUE )
			dimnames(d_of_t)$diag[6+2*i+(-1:0)] = base::c( name_diag_p , name_diag_i )
		}
	}
	
	return(d_of_t)
}
## }}}


