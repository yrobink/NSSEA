
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
gam_decomposition = function( Xd , Enat , time_center = NULL , gam_dof = 7 , Sigma = NULL , verbose = FALSE , code = "yrobin" )
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
#		gam_model = mgcv::gam( X ~ s( t , k = gam_dof - 1 , fx = FALSE ) + Xl , data = data.frame( X = x , Xl = Xl , t = time_x ) )
		gam_model = NSSEA::fit_gam_with_fix_dof( x , Xl , time_x , gam_dof )
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

fit_gam_with_fix_dof = function( x , Xl , time_x , dof ) ##{{{
{
		## Best estimate
		tol = 1e-2
		edof = dof + 1. + tol
		rup = 1e2
		rlo = 1e-2
		k   = dof + 2
		nit = 0
		while( abs( dof - edof ) > tol )
		{
			r = (rup + rlo) / 2
			gam_model = mgcv::gam( X ~ s( t , k = k , fx = FALSE , sp = r ) + Xl , data = data.frame( X = x , Xl = Xl , t = time_x ) )
			edof = sum( gam_model$edf )
			if( edof < dof )
				rup = r
			else
				rlo = r
			
			nit = nit + 1
			if( nit %% 100 == 0 )
			{
				rup = 1e2
				rlo = 1e-2
				k   = k + 1
			}
		}
	return(gam_model)
}
##}}}

