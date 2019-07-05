
#############################
## Aurelien Ribes          ##
## aurelien.ribes@meteo.fr ##
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

## gam_decomposition {{{

#' gam_decomposition
#'
#' Perform the GAM decomposition
#'
#' @usage gam_decomposition( Xd , Enat , Sigma , time_center , gam_dof , verbose , code )
#'
#' @param Xd [array] covariate
#' @param Enat [array] natural signal
#' @param Sigma [array] auto covariance matrix, if NULL, identity is used
#' @param time_center [time] centering covariate at time_center. If NULL, not used
#' @param gam_dof [double] degree of freedom of GAM
#' @param verbose [bool] print or not state of execution
#' @param code [str] code of "yrobin" or "aribes"
#'
#' @return X_splitted [list] $X is the covariate, $X_center the centered variable and $X_event the value removed to center
#'
#' @examples
#' ##
#' 
#' @export
gam_decomposition = function( Xd , Enat , Sigma = NULL , time_center = NULL , gam_dof = 7 , verbose = FALSE , code = "yrobin" )
{
	if( code == "yrobin" )
	{
		return( gam_decomposition_yrobin( Xd , Enat , Sigma , time_center , gam_dof , verbose ) )
	}
	else
	{
		return( gam_decomposition_aribes( Xd , Enat , Sigma , time_center , gam_dof , verbose ) )
	}
}
##}}}

## gam_decomposition_aribes {{{

#' gam_decomposition_aribes
#'
#' Perform the GAM decomposition
#'
#' @importFrom mgcv gam
#' @importFrom mgcv s
#'
#' @usage gam_decomposition_aribes( Xd , Enat , Sigma , time_center , gam_dof , verbose )
#'
#' @param Xd [array] covariate
#' @param Enat [array] natural signal
#' @param Sigma [array] auto covariance matrix, if NULL, identity is used
#' @param time_center [time] centering covariate at time_center. If NULL, not used
#' @param gam_dof [double] degree of freedom of GAM
#' @param verbose [bool] print or not state of execution
#'
#' @return X_splitted [list] $X is the covariate, $X_center the centered variable and $X_event the value removed to center
#'
#' @examples
#' ##
#' 
#' @export
gam_decomposition_aribes = function( Xd , Enat , Sigma = NULL , time_center = NULL , gam_dof = 7 , verbose = FALSE )
{
	Sigma = if( is.null(Sigma) ) base::diag(base::rep(1,dim(Xd)[1])) else Sigma
	
	# Preliminary spline calculation
	#--------------------------------
	spl  = NSSEA::SplinesModel$new( dim(Xd)[1] , dof = gam_dof - 2 )
	HatM = spl$projection_matrix
	Cov_spline = t(HatM) %*% Sigma %*% HatM
	V = eigen(Cov_spline)
	Cov_spline_12 = V$vectors %*% diag(sqrt(pmax(V$values,0))) %*% t(V$vectors)
	# For extention: compute a Hat matrix (possibly with different methods), then apply to Xd...
	
	# Design X
	#----------
	time_str    = dimnames(Xd)$time
	time        = as.numeric(time_str)
	n_time      = length(time)
	models      = dimnames(Xd)$models
	n_models    = length(models)
	n_sample    = dim(Enat)[2] - 1
	sample_str  = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) )
	
	X        = array( NA , dim = base::c( n_time , n_sample + 1 , 3 , n_models ) , dimnames = list( time  = time_str , sample = sample_str , forcing = c("nat","ant","all") , models = models ) )
	beta_nat = array( NA , dim = base::c( n_sample + 1 , n_models ) , dimnames = list( sample = sample_str , models  = models ) )
	
	
#	s = gam::s
	# Fit gam, then apply smoothing splines
	#---------------------------------------
	# x = 1 + Enat + f(t) + e
	# By default the intercept "1" is put together with Enat, so x_nat = 1 + Enat
	pb = NSSEA::ProgressBar$new( "Covariate decomposition" , n_models * n_sample )
	for( i in 1:n_models )
	{
		x = Xd[,i]
		
		# Best estimate
		## Aurelien: j'ai change gam pour mgcv car on ne peut pas mettre gam et mgcv ensemble dans le meme package, ton implementation est laisse en commentaire
#		gam_be = gam::gam( x ~ s( time , df = gam_dof - 1 ) + Enat[,1] )
#		beta_nat["be",i] = gam_be$coefficients[3]
		
		gam_be = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,1] , t = time ) )
		beta_nat["be",i] = gam_be$coefficients[2]
		
		
		x_nat_brut = beta_nat["be",i] * Enat[,1]				# Enat
		x_nat_removed_be = x - x_nat_brut						# x - Enat
		x_ant_uncentered_be = HatM %*% x_nat_removed_be			# mu + f(t)
		x_ant_be = x_ant_uncentered_be - x_ant_uncentered_be[1]	# f(t)
		
		X[,"be","ant",i] = x_ant_be
		X[,"be","all",i] = base::mean(x) + x_nat_brut - base::mean(x_nat_brut) + x_ant_be - base::mean(x_ant_be)	# = 1 + Enat + f(t)
		
		# Rough estimate of beta_nat variance (Gaussian distribution is assumed in resampling) -- OLS formula
		sd_beta_nat = base::sqrt( base::t( Enat[,1] ) %*% Sigma %*% Enat[,1] / ( base::t(Enat[,1]) %*% Enat[,1] )^2 ) ## Yoann : ce serait pas plutot Enat[,i] ???
		
		# Resampling
		for( j in 1:n_sample )
		{
			pb$print()
#			gam_re = gam::gam(x ~ s( time , df = gam_dof - 1 ) + Enat[,i+1] )
#			beta_nat[j+1,i] = gam_re$coefficients[3] + sd_beta_nat * stats::rnorm(1)	# beta_nat; Possibly sd_beta_nat should be calculated for each Enat_resampled...
			
			gam_re = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,i+1] , t = time ) )
			beta_nat[j+1,i] = gam_re$coefficients[2] + sd_beta_nat * stats::rnorm(1)
			
			x_nat_brut_re = beta_nat[j+1,i] * Enat[,j+1]												# Enat
			x_nat_removed_re = x - x_nat_brut_re														# x - Enat
			x_ant_uncentered_re = HatM %*% x_nat_removed_re + Cov_spline_12 %*% stats::rnorm(n_time)	# mu + f(t)
			X[,j+1,"ant",i] = x_ant_uncentered_re - x_ant_uncentered_re[1]								# f(t)
			X[,j+1,"all",i] = base::mean(x) + x_nat_brut_re - base::mean(x_nat_brut_re) + X[,j+1,"ant",i] - base::mean(X[,j+1,"ant",i])	# = 1 + Enat + f(t)
		}
		X[,,"nat",i] = X[,,"all",i] - X[,,"ant",i]		# X_nat = 1 + Enat
	}
	pb$end()
	
	## Centering
	X_center = NULL
	X_event  = NULL
	if( !is.null(time_center) )
	{
		X_event  = X[as.character(time_center),,"all",]
		X_center = NSSEA::ones(X[,,base::c("nat","all"),])
		X_center[,,"all",] = base::t( base::apply( X[,,"all",] , "time" , function(x) { return(x - X_event) } ) )
		X_center[,,"nat",] = base::t( base::apply( X[,,"nat",] , "time" , function(x) { return(x - X_event) } ) )
	}
	
	return( list( X = X , X_center = X_center , X_event = X_event ) )
}
#}}}

## gam_decomposition_yrobin {{{

#' gam_decomposition_yrobin
#'
#' Perform the GAM decomposition
#'
#' @importFrom mgcv gam
#' @importFrom mgcv s
#'
#' @usage gam_decomposition_yrobin( Xd , Enat , Sigma , time_center , gam_dof , verbose )
#'
#' @param Xd [array] covariate
#' @param Enat [array] natural signal
#' @param Sigma [array] auto covariance matrix, if NULL, identity is used
#' @param time_center [time] centering covariate at time_center. If NULL, not used
#' @param gam_dof [double] degree of freedom of GAM
#' @param verbose [bool] print or not state of execution
#'
#' @return X_splitted [list] $X is the covariate, $X_center the centered variable and $X_event the value removed to center
#'
#' @examples
#' ##
#' 
#' @export
gam_decomposition_yrobin = function( Xd , Enat , Sigma = NULL , time_center = NULL , gam_dof = 7 , verbose = FALSE )
{
	## Usefull variables
	time     = as.numeric(base::unique( names(Xd[[1]]) ))
	n_time   = length(time)
	time_l   = base::rep( time[1] , n_time ) ## Stationary time axis, blocked at first time point
	models   = names(Xd)
	n_models = length(models)
	n_sample = dim(Enat)[2] - 1
	Eant     = base::rep( 0. , n_time )       ## Enat == 0 along time axis, for "ant"
#	Sigma    = if( is.null(Sigma) ) base::diag(base::rep(1,dim(Xd)[1])) else Sigma
	
	## Output
	sample_str  = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) )
	X = array( NA , dim = base::c( n_time , n_sample + 1 , 3 , n_models ) , dimnames = list( time  = time , sample = sample_str , forcing = c("nat","ant","all") , models = models ) )
	
	pb = NSSEA::ProgressBar$new( "Covariate decomposition" , n_models * n_sample )
	for( i in 1:n_models )
	{
		## Model i
		x      = Xd[[i]]
		time_x = as.numeric(names(x))
		idx_Xl = match( time_x , as.numeric(names(Enat[,1])) )
		Xl = Enat[idx_Xl,1]
		
		## Best estimate
		gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = FALSE ) + Xl , data = data.frame( X = x , Xl = Xl , t = time_x ) )
		X[,"be","all",i] = mgcv::predict.gam( gam_model , newdata = data.frame(  t = time   , Xl = Enat[,1] ) )
		X[,"be","nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame(  t = time_l , Xl = Enat[,1] ) )
		X[,"be","ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame(  t = time   , Xl = Eant     ) )
		
		
		## Covariance matrix of GAM coefficients
		mean_gam = gam_model$coefficients
		cov_gam  = gam_model$Vp
		std_gam  = NSSEA::matrix_sqrt( cov_gam )
		
		## Bootstrap
		for( j in 1:n_sample )
		{
			
			if(verbose) pb$print()
			gam_model$coefficients = mean_gam + std_gam %*% stats::rnorm( length(mean_gam) , mean = 0 , sd = 1 )
			
			X[,j+1,"all",i] = mgcv::predict.gam( gam_model , newdata = data.frame( t = time   , Xl = Enat[,j+1] ) )
			X[,j+1,"nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame( t = time_l , Xl = Enat[,j+1] ) )
			X[,j+1,"ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame( t = time   , Xl = Eant       ) )
		}
	}
	
	
	## Centering
	X_center = NULL
	X_event  = NULL
	if( !is.null(time_center) )
	{
		X_event  = X[as.character(time_center),,"all",]
		X_center = NSSEA::ones(X[,,base::c("nat","all"),])
		X_center[,,"all",] = base::t( base::apply( X[,,"all",] , "time" , function(x) { return(x - X_event) } ) )
		X_center[,,"nat",] = base::t( base::apply( X[,,"nat",] , "time" , function(x) { return(x - X_event) } ) )
	}
	
	
	if(verbose) pb$end()
	
#	X[,,"ant",] = X[,,"all",] - X[,,"nat",] ==> Not good: noise_spl - noise_spl
	
	return( list( X = X , X_center = X_center , X_event = X_event ) )
}
##}}}


## gam_decomposition_aribes_save {{{

#' gam_decomposition_aribes_save
#'
#' Perform the GAM decomposition
#'
#' @importFrom mgcv gam
#' @importFrom mgcv s
#'
#' @usage gam_decomposition_aribes_save( Xd , Enat , Sigma , time_center , gam_dof , verbose )
#'
#' @param Xd [array] covariate
#' @param Enat [array] natural signal
#' @param Sigma [array] auto covariance matrix, if NULL, identity is used
#' @param time_center [time] centering covariate at time_center. If NULL, not used
#' @param gam_dof [double] degree of freedom of GAM
#' @param verbose [bool] print or not state of execution
#'
#' @return X_splitted [list] $X is the covariate, $X_center the centered variable and $X_event the value removed to center
#'
#' @examples
#' ##
#' 
#' @export
gam_decomposition_aribes_save = function( Xd , Enat , Sigma = NULL , time_center = NULL , gam_dof = 7 , verbose = FALSE )
{
	Sigma = if( is.null(Sigma) ) base::diag(base::rep(1,dim(Xd)[1])) else Sigma
	
	# Preliminary spline calculation
	#--------------------------------
	spl  = NSSEA::SplinesModel$new( dim(Xd)[1] , dof = gam_dof - 1 )
	HatM = spl$projection_matrix
	Cov_spline = t(HatM) %*% Sigma %*% HatM
	V = eigen(Cov_spline)
	Cov_spline_12 = V$vectors %*% diag(sqrt(pmax(V$values,0))) %*% t(V$vectors)
	# For extention: compute a Hat matrix (possibly with different methods), then apply to Xd...
	
	# Design X
	#----------
	time_str    = dimnames(Xd)$time
	time        = as.numeric(time_str)
	n_time      = length(time)
	models      = dimnames(Xd)$models
	n_models    = length(models)
	n_sample    = dim(Enat)[2] - 1
	sample_str  = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) )
	
	X        = array( NA , dim = base::c( n_time , n_sample + 1 , 3 , n_models ) , dimnames = list( time  = time_str , sample = sample_str , forcing = c("nat","ant","all") , models = models ) )
	beta_nat = array( NA , dim = base::c( n_sample + 1 , n_models ) , dimnames = list( sample = sample_str , models  = models ) )
	
	
	# Fit gam, then apply smoothing splines
	#---------------------------------------
	# x = 1 + Enat + f(t) + e
	# By default the intercept "1" is put together with Enat, so x_nat = 1 + Enat
	pb = NSSEA::ProgressBar$new( "Covariate decomposition" , n_models * n_sample )
	for( i in 1:n_models )
	{
		x = Xd[,i]
		
		# Best estimate
		## Aurelien: j'ai change gam pour mgcv car on ne peut pas mettre gam et mgcv ensemble dans le meme package, ton implementation est laisse en commentaire
#		gam_be = gam::gam( x ~ s( time , df = gam_dof ) + Enat[,1] )
#		beta_nat["be",i] = gam_be$coefficients[3]
		
		gam_be = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,1] , t = time ) )
		beta_nat["be",i] = gam_be$coefficients[2]
		
		
		x_nat_brut = beta_nat["be",i] * Enat[,1]				# Enat
		x_nat_removed_be = x - x_nat_brut						# x - Enat
		x_ant_uncentered_be = HatM %*% x_nat_removed_be			# mu + f(t)
		x_ant_be = x_ant_uncentered_be - x_ant_uncentered_be[1]	# f(t)
		
		X[,"be","ant",i] = x_ant_be
		X[,"be","all",i] = base::mean(x) + x_nat_brut - base::mean(x_nat_brut) + x_ant_be - base::mean(x_ant_be)	# = 1 + Enat + f(t)
		
		# Rough estimate of beta_nat variance (Gaussian distribution is assumed in resampling) -- OLS formula
		sd_beta_nat = base::sqrt( base::t( Enat[,1] ) %*% Sigma %*% Enat[,1] / ( base::t(Enat[,1]) %*% Enat[,1] )^2 ) ## Yoann : ce serait pas plutot Enat[,i] ???
		
		# Resampling
		for( j in 1:n_sample )
		{
			pb$print()
#			gam_re = gam::gam(x ~ gam::s( time , df = gam_dof ) + Enat[,i+1] )
#			beta_nat[j+1,i] = gam_re$coefficients[3] + sd_beta_nat * stats::rnorm(1)	# beta_nat; Possibly sd_beta_nat should be calculated for each Enat_resampled...
			
			gam_re = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,i+1] , t = time ) )
			beta_nat[j+1,i] = gam_re$coefficients[2] + sd_beta_nat * stats::rnorm(1)
			
			x_nat_brut_re = beta_nat[j+1,i] * Enat[,j+1]												# Enat
			x_nat_removed_re = x - x_nat_brut_re														# x - Enat
			x_ant_uncentered_re = HatM %*% x_nat_removed_re + Cov_spline_12 %*% stats::rnorm(n_time)	# mu + f(t)
			X[,j+1,"ant",i] = x_ant_uncentered_re - x_ant_uncentered_re[1]								# f(t)
			X[,j+1,"all",i] = base::mean(x) + x_nat_brut_re - base::mean(x_nat_brut_re) + X[,j+1,"ant",i] - base::mean(X[,j+1,"ant",i])	# = 1 + Enat + f(t)
		}
		X[,,"nat",i] = X[,,"all",i] - X[,,"ant",i]		# X_nat = 1 + Enat
	}
	pb$end()
	
	## Centering
	X_center = NULL
	X_event  = NULL
	if( !is.null(time_center) )
	{
		X_event  = X[as.character(time_center),,"all",]
		X_center = NSSEA::ones(X[,,base::c("nat","all"),])
		X_center[,,"all",] = base::t( base::apply( X[,,"all",] , "time" , function(x) { return(x - X_event) } ) )
		X_center[,,"nat",] = base::t( base::apply( X[,,"nat",] , "time" , function(x) { return(x - X_event) } ) )
	}
	
	return( list( X = X , X_center = X_center , X_event = X_event ) )
}
#}}}

## gam_decomposition_yrobin_save {{{

#' gam_decomposition_yrobin_save
#'
#' Perform the GAM decomposition
#'
#' @importFrom mgcv gam
#' @importFrom mgcv s
#'
#' @usage gam_decomposition_yrobin_save( Xd , Enat , Sigma , time_center , gam_dof , verbose )
#'
#' @param Xd [array] covariate
#' @param Enat [array] natural signal
#' @param Sigma [array] auto covariance matrix, if NULL, identity is used
#' @param time_center [time] centering covariate at time_center. If NULL, not used
#' @param gam_dof [double] degree of freedom of GAM
#' @param verbose [bool] print or not state of execution
#'
#' @return X_splitted [list] $X is the covariate, $X_center the centered variable and $X_event the value removed to center
#'
#' @examples
#' ##
#' 
#' @export
gam_decomposition_yrobin_save = function( Xd , Enat , Sigma = NULL , time_center = NULL , gam_dof = 7 , verbose = FALSE )
{
	## Usefull variables
	time     = as.numeric(dimnames(Xd)$time)
	n_time   = length(time)
	time_l   = base::rep( time[1] , n_time ) ## Stationary time axis, blocked at first time point
	models   = dimnames(Xd)$models
	n_models = length(models)
	n_sample = dim(Enat)[2] - 1
	Eant     = base::rep( 0. , n_time )       ## Enat == 0 along time axis, for "ant"
	Sigma = if( is.null(Sigma) ) base::diag(base::rep(1,dim(Xd)[1])) else Sigma
	
	## Output
	sample_str  = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) )
	X = array( NA , dim = base::c( n_time , n_sample + 1 , 3 , n_models ) , dimnames = list( time  = time , sample = sample_str , forcing = c("nat","ant","all") , models = models ) )
	
	pb = NSSEA::ProgressBar$new( "Covariate decomposition" , n_models * n_sample )
	for( i in 1:n_models )
	{
		## Model i
		x = Xd[,i]
		
		## Best estimate
		gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,1] , t = time ) )
		X[,"be","all",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Enat[,1] ) )
		X[,"be","nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time_l , Xl = Enat[,1] ) )
		X[,"be","ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Eant     ) )
		
		## Bootstrap
		for( j in 1:n_sample )
		{
			
			if(verbose) pb$print()
			gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,i+1] , t = time ) )
			
			## Coefficients of decomposition
			int_coef  = gam_model$coefficients[1]
			lin_coef  = gam_model$coefficients[2]
			spl_coef  = gam_model$coefficients[3:gam_dof]
			
			model_mat = mgcv::predict.gam( gam_model , type = "lpmatrix" )
			spl_mat   = model_mat[,3:gam_dof]
			proj_mat  = spl_mat %*% base::solve( base::t(spl_mat) %*% spl_mat ) %*% base::t(spl_mat)
			
			## Noise of linear term
			sigma_lin = base::sqrt( base::t(Enat[,i+1]) %*% Sigma %*% Enat[,i+1] / (base::t(Enat[,i+1]) %*% Enat[,i+1])^2  )
			noise_lin = stats::rnorm( 1 , mean = 0 , sd = sigma_lin )
			
			## Noise of spline term
			cov_spl   = base::t(proj_mat) %*% Sigma %*% proj_mat
			eigen     = base::eigen(cov_spl)
			cov_spl   = eigen$vectors %*% base::diag( base::sqrt( base::pmax( eigen$values , 0 ) ) ) %*% base::t(eigen$vectors)
			noise_spl = as.vector(cov_spl %*% stats::rnorm( n_time ))
			noise_spl = noise_spl - noise_spl[1]
			
			## Final decomposition
			gam_model$coefficients[2] = lin_coef + noise_lin
			X[,j+1,"all",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Enat[,i+1] ) ) + noise_spl
			X[,j+1,"nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time_l , Xl = Enat[,i+1] ) )
			X[,j+1,"ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Eant       ) ) + noise_spl
		}
	}
	
	## Centering
	X_center = NULL
	X_event  = NULL
	if( !is.null(time_center) )
	{
		X_event  = X[as.character(time_center),,"all",]
		X_center = NSSEA::ones(X[,,base::c("nat","all"),])
		X_center[,,"all",] = base::t( base::apply( X[,,"all",] , "time" , function(x) { return(x - X_event) } ) )
		X_center[,,"nat",] = base::t( base::apply( X[,,"nat",] , "time" , function(x) { return(x - X_event) } ) )
	}
	
	
	if(verbose) pb$end()
	
#	X[,,"ant",] = X[,,"all",] - X[,,"nat",] ==> Not good: noise_spl - noise_spl
	
	return( list( X = X , X_center = X_center , X_event = X_event ) )
}
##}}}

## gam_decomposition_yrobin_save2 {{{

#' gam_decomposition_yrobin_save2
#'
#' Perform the GAM decomposition
#'
#' @importFrom mgcv gam
#' @importFrom mgcv s
#'
#' @usage gam_decomposition_yrobin_save2( Xd , Enat , Sigma , time_center , gam_dof , verbose )
#'
#' @param Xd [array] covariate
#' @param Enat [array] natural signal
#' @param Sigma [array] auto covariance matrix, if NULL, identity is used
#' @param time_center [time] centering covariate at time_center. If NULL, not used
#' @param gam_dof [double] degree of freedom of GAM
#' @param verbose [bool] print or not state of execution
#'
#' @return X_splitted [list] $X is the covariate, $X_center the centered variable and $X_event the value removed to center
#'
#' @examples
#' ##
#' 
#' @export
gam_decomposition_yrobin_save2 = function( Xd , Enat , Sigma = NULL , time_center = NULL , gam_dof = 7 , verbose = FALSE )
{
	## Usefull variables
	time     = as.numeric(dimnames(Xd)$time)
	n_time   = length(time)
	time_l   = base::rep( time[1] , n_time ) ## Stationary time axis, blocked at first time point
	models   = dimnames(Xd)$models
	n_models = length(models)
	n_sample = dim(Enat)[2] - 1
	Eant     = base::rep( 0. , n_time )       ## Enat == 0 along time axis, for "ant"
	Sigma = if( is.null(Sigma) ) base::diag(base::rep(1,dim(Xd)[1])) else Sigma
	
	## Output
	sample_str  = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) )
	X = array( NA , dim = base::c( n_time , n_sample + 1 , 3 , n_models ) , dimnames = list( time  = time , sample = sample_str , forcing = c("nat","ant","all") , models = models ) )
	
	pb = NSSEA::ProgressBar$new( "Covariate decomposition" , n_models * n_sample )
	for( i in 1:n_models )
	{
		## Model i
		x = Xd[,i]
		
		## Best estimate
		gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,1] , t = time ) )
		X[,"be","all",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Enat[,1] ) )
		X[,"be","nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time_l , Xl = Enat[,1] ) )
		X[,"be","ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Eant     ) )
		
		## Coefficients of decomposition
		int_coef  = gam_model$coefficients[1]
		lin_coef  = gam_model$coefficients[2]
		spl_coef  = gam_model$coefficients[3:gam_dof]
		
		model_mat = mgcv::predict.gam( gam_model , type = "lpmatrix" )
		spl_mat   = model_mat[,3:gam_dof]
		proj_mat  = spl_mat %*% base::solve( base::t(spl_mat) %*% spl_mat ) %*% base::t(spl_mat)
		
		## Noise of linear term
		sigma_lin = base::sqrt( base::t(Enat[,1]) %*% Sigma %*% Enat[,1] / (base::t(Enat[,1]) %*% Enat[,1])^2  )
		
		## Noise of spline term
		cov_spl   = base::t(proj_mat) %*% Sigma %*% proj_mat
		eigen     = base::eigen(cov_spl)
		cov_spl   = eigen$vectors %*% base::diag( base::sqrt( base::pmax( eigen$values , 0 ) ) ) %*% base::t(eigen$vectors)
		
		
		## Bootstrap
		for( j in 1:n_sample )
		{
			
			if(verbose) pb$print()
			gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = TRUE ) + Xl , data = data.frame( X = x , Xl = Enat[,i+1] , t = time ) )
			
			## Noise of linear term
			noise_lin = sigma_lin * stats::rnorm(1)
			
			## Noise of spline term
			noise_spl = as.vector(cov_spl %*% stats::rnorm( n_time ))
			noise_spl = noise_spl - noise_spl[1]
			
			## Final decomposition
			gam_model$coefficients[2] = gam_model$coefficients[2] + noise_lin
			
			X[,j+1,"all",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Enat[,i+1] ) ) + noise_spl
			X[,j+1,"nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time_l , Xl = Enat[,i+1] ) )
			X[,j+1,"ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Eant       ) ) + noise_spl
		}
	}
	
	
	## Centering
	X_center = NULL
	X_event  = NULL
	if( !is.null(time_center) )
	{
		X_event  = X[as.character(time_center),,"all",]
		X_center = NSSEA::ones(X[,,base::c("nat","all"),])
		X_center[,,"all",] = base::t( base::apply( X[,,"all",] , "time" , function(x) { return(x - X_event) } ) )
		X_center[,,"nat",] = base::t( base::apply( X[,,"nat",] , "time" , function(x) { return(x - X_event) } ) )
	}
	
	
	if(verbose) pb$end()
	
#	X[,,"ant",] = X[,,"all",] - X[,,"nat",] ==> Not good: noise_spl - noise_spl
	
	return( list( X = X , X_center = X_center , X_event = X_event ) )
}
##}}}

## gam_decomposition_yrobin_save3 {{{

#' gam_decomposition_yrobin_save3
#'
#' Perform the GAM decomposition
#'
#' @importFrom mgcv gam
#' @importFrom mgcv s
#'
#' @usage gam_decomposition_yrobin_save3( Xd , Enat , Sigma , time_center , gam_dof , verbose )
#'
#' @param Xd [array] covariate
#' @param Enat [array] natural signal
#' @param Sigma [array] auto covariance matrix, if NULL, identity is used
#' @param time_center [time] centering covariate at time_center. If NULL, not used
#' @param gam_dof [double] degree of freedom of GAM
#' @param verbose [bool] print or not state of execution
#'
#' @return X_splitted [list] $X is the covariate, $X_center the centered variable and $X_event the value removed to center
#'
#' @examples
#' ##
#' 
#' @export
gam_decomposition_yrobin_save3 = function( Xd , Enat , Sigma = NULL , time_center = NULL , gam_dof = 7 , verbose = FALSE )
{
	## Usefull variables
	time     = as.numeric(dimnames(Xd)$time)
	n_time   = length(time)
	time_l   = base::rep( time[1] , n_time ) ## Stationary time axis, blocked at first time point
	models   = dimnames(Xd)$models
	n_models = length(models)
	n_sample = dim(Enat)[2] - 1
	Eant     = base::rep( 0. , n_time )       ## Enat == 0 along time axis, for "ant"
	Sigma    = if( is.null(Sigma) ) base::diag(base::rep(1,dim(Xd)[1])) else Sigma
	
	## Output
	sample_str  = base::c( "be" , base::paste0("S" , NSSEA::str_list_int(1:n_sample) ) )
	X = array( NA , dim = base::c( n_time , n_sample + 1 , 3 , n_models ) , dimnames = list( time  = time , sample = sample_str , forcing = c("nat","ant","all") , models = models ) )
	
	pb = NSSEA::ProgressBar$new( "Covariate decomposition" , n_models * n_sample )
	for( i in 1:n_models )
	{
		## Model i
		x = Xd[,i]
		
		## Best estimate
		gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = FALSE ) + Xl , data = data.frame( X = x , Xl = Enat[,1] , t = time ) )
		X[,"be","all",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Enat[,1] ) )
		X[,"be","nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time_l , Xl = Enat[,1] ) )
		X[,"be","ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Eant     ) )
		
		
		## Covariance matrix of GAM coefficients
		mean_gam = gam_model$coefficients
		cov_gam  = gam_model$Vp
		std_gam  = NSSEA::matrix_sqrt( cov_gam )
		
		## Bootstrap
		for( j in 1:n_sample )
		{
			
			if(verbose) pb$print()
#			gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = FALSE ) + Xl , data = data.frame( X = x , Xl = Enat[,i+1] , t = time ) )
			gam_model$coefficients = mean_gam + std_gam %*% stats::rnorm( length(mean_gam) , mean = 0 , sd = 1 )
			
			X[,j+1,"all",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Enat[,i+1] ) )
			X[,j+1,"nat",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time_l , Xl = Enat[,i+1] ) )
			X[,j+1,"ant",i] = mgcv::predict.gam( gam_model , newdata = data.frame( X = x , t = time   , Xl = Eant       ) )
		}
	}
	
	
	## Centering
	X_center = NULL
	X_event  = NULL
	if( !is.null(time_center) )
	{
		X_event  = X[as.character(time_center),,"all",]
		X_center = NSSEA::ones(X[,,base::c("nat","all"),])
		X_center[,,"all",] = base::t( base::apply( X[,,"all",] , "time" , function(x) { return(x - X_event) } ) )
		X_center[,,"nat",] = base::t( base::apply( X[,,"nat",] , "time" , function(x) { return(x - X_event) } ) )
	}
	
	
	if(verbose) pb$end()
	
#	X[,,"ant",] = X[,,"all",] - X[,,"nat",] ==> Not good: noise_spl - noise_spl
	
	return( list( X = X , X_center = X_center , X_event = X_event ) )
}
##}}}

