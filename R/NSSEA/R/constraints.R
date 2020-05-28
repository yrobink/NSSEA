
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


## generic_constraint {{{

#' generic_constraint
#'
#' Generic method to apply constraint
#'
#' @usage generic_constraint(x,SX,y,SY,H)
#'
#' @param x	[vector] a priori mean (background)
#' @param SX	[matrix] cov matrix of x (background error)
#' @param y	[vector] observation (obs)
#' @param SY	[matrix] cov matrix of y (observational error)
#' @param H	[matrix] observation operator (observation operator)
#'
#' @return out [list] mean, cov and std estimated
#'
#' @examples
#' ##
#' 
#' @export
generic_constraint = function( x , SX , y , SY , H )
{
	
	Sinv     = NSSEA::ginv( H %*% SX %*% base::t(H) + SY )
	K	     = SX %*% base::t(H) %*% Sinv
	x_post	 = x + K %*% ( y - H %*% x )
	SX_post	 = SX - SX %*% base::t(H) %*% Sinv %*% H %*% SX
	std_post = NSSEA::matrix_sqrt(SX_post)
	
	out = list( mean = as.numeric(x_post) , cov = SX_post , std = std_post )
	return(out)
}
##}}}

## constraints_CX {{{

#' constraints_CX
#'
#' Constraint on covariable by observed covariable
#'
#' @usage constraints_CX( coffee , Xo , cx_params , Sigma , verbose )
#'
#' @param coffee [coffee variable]
#' @param Xo [vector] Observed covariable
#' @param cx_params [list] A list build by NSSEA::make_CXParams
#' @param Sigma [matrix] autocovariance matrix
#' @param verbose [bool] print or not state of execution
#'
#' @return coffee [coffee variable] with constraint CX
#'
#' @examples
#' ##
#' 
#' @export
constraints_CX = function( coffee , Xo , cx_params , Sigma = NULL , verbose = FALSE )
{
	if(verbose) cat( "Constraints CX         \r" )
	
	## Parameters
	time        = coffee$time
	time_Xo     = as.numeric(names(Xo))
	n_time      = coffee$n_time
	n_time_Xo   = length(time_Xo)
	n_mm_params = coffee$n_mm_params
	n_sample    = coffee$n_sample
	if( is.null(Sigma) )
		Sigma = base::diag( base::rep( 1 , n_time ) )
	
	## Time of observed X
	idx_obs     = time %in% time_Xo
	ref      = if( is.null(cx_params$ref) ) time_Xo else cx_params$ref
	
	if( !base::all( ref %in% time_Xo ) )
		print("WARNING: Reference period for constrain C1 is not acceptable (inconsistency with obs)") ; return
	
	if( !base::all(ref %in% time) )
		print("WARNING: Reference period for constrain C1 is not acceptable (inconsistency with model outputs)") ; return
	
	
	## Projection matrix H
	centerX  = base::diag( base::rep( 1 , n_time    ) ) - ( if(cx_params$centering) base::rep( 1 , n_time )    %o% ( time    %in% ref ) / length(ref) else 0 )
	centerY  = base::diag( base::rep( 1 , n_time_Xo ) ) - ( if(cx_params$centering) base::rep( 1 , n_time_Xo ) %o% ( time_Xo %in% ref ) / length(ref) else 0 )
	extractX = base::cbind( base::diag( base::rep( 0 , n_time ) ) , base::diag( base::rep( 1 , n_time ) ) , array( 0 , dim = base::c( n_time , coffee$n_ns_params ) ) )
	H_full   = centerX %*% extractX			# Centering * extracting
	H        = H_full[idx_obs,]	# Restriction to the observed period
	
	
	# Other inputs : x, SX, y, SY
	X  = coffee$mm_params$mean
	SX = coffee$mm_params$cov
	Y  = centerY %*% Xo
	SY = centerY %*% Sigma[idx_obs,idx_obs] %*% base::t(centerY)
	
	
	##	Re-scale SY in order to be consistent with observations Xo, we search for lambda s.t. 
	##		res := Y - H %*% X ~ N( 0 , H %*% SX %*% t(H) + lambda *SY )
	##	Let iSig := ginv( H %*% SX %*% t(H) + lambda * SY ), the mle of lambda satisfies
	##		Tr( iSig %*% SY ) - t(res) %*% iSig %*% SY %*% iSig %*% res = 0
	if( !cx_params$trust )
	{
		res   = Y - H %*% X
		K = H %*% SX %*% base::t(H)
		fct_to_root = function(lambda)
		{
			SY_tmp = lambda * SY
			iSig = NSSEA::ginv( K + SY_tmp )
			return( base::sum( base::diag( iSig %*% SY_tmp ) ) - base::t(res) %*% iSig %*% SY_tmp %*% iSig %*% res )
		}
		
		interval = base::c(1e-2,1e2)
		while( fct_to_root(interval[1]) * fct_to_root(interval[2]) > 0 ) ## test opposite sign of extreme of interval (if true, interval contains a 0)
		{
			interval[1] = interval[1] / 2
			interval[2] = interval[2] * 2
		}
		lambda = stats::uniroot( fct_to_root , interval = interval )$root
		
		SY = lambda * SY
	}
	
	## Apply constraint
	post_dist = NSSEA::generic_constraint( X , SX , Y , SY , H )
	
	## Sample
	X_cons          = array( NA , dim = base::c( n_time , n_sample + 1 , 2 ) , dimnames = list( time = time , sample = dimnames(coffee$X)$sample , forcing = base::c("all","nat") ) )
	epsil_cons      = array( 0 , dim = base::c( n_mm_params , n_sample + 1 ) )
	epsil_cons[,-1] = stats::rnorm( n_mm_params * n_sample )
	X_cons_tmp      = post_dist$mean %o% base::rep( 1 , n_sample + 1 ) + post_dist$std %*% epsil_cons
	X_cons[,,"all"] = X_cons_tmp[1:n_time,]
	X_cons[,,"nat"] = X_cons_tmp[n_time + 1:n_time,]
	ns_params       = X_cons_tmp[(2*n_time+1):n_mm_params,]
	
	##TODO on peut récupérer des paramètres ici pour ns_param ???
	coffee$ns_params_CX = coffee$ns_params
	
	for( i in 1:coffee$n_models )
	{
		coffee$X[,,,i]           = X_cons
		coffee$ns_params_CX[,,i] = ns_params
	}
	
	if(verbose) cat( "Constraints CX (Done)        \n" )
	
	return(coffee)
}
##}}}

## constraints_C0 {{{

#' constraints_C0
#'
#' Constraint by observations
#'
#' @usage constraints_C0( coffee , Yo , event , verbose )
#'
#' @param coffee [coffee variable]
#' @param Yo [vector] Observations
#' @param event [event variable]
#' @param verbose [bool] print or not state of execution
#'
#' @return coffee [coffee variable] with constraint C0
#'
#' @examples
#' ##
#' 
#' @export
constraints_C0 = function( coffee , Yo , event , verbose = FALSE )
{
	if(verbose) cat( "Constraints C0         \r" )
	n_models = coffee$n_models
	n_sample = coffee$n_sample
	models   = coffee$models
	time     = coffee$time
	time_Yo  = as.numeric(names(Yo))
	n_time_Yo = length(time_Yo)
	
	
	# New NS_param
	ns_params_C0 = coffee$ns_params
	ns_params_C0["sig1",,] = ns_params_C0["sig1",,] / ns_params_C0["sig0",,]
	
	## Bootstrap on Y
	time_bs = array( base::sample( time_Yo , ( n_sample + 1 ) * n_models , replace = TRUE ) , dim = base::c( n_time_Yo , n_sample + 1 , n_models ) , dimnames = list( time = time_Yo , sample = dimnames(ns_params_C0)$sample , models = models ) )
	time_bs[,1,] = time_Yo
	Yo_bs = array( Yo[as.character( time_bs )] , dim = dim(time_bs) , dimnames = dimnames(time_bs) )
	
	## Bootstrap on X
	X_bs  = NSSEA::ones(Yo_bs) + NA
	for( i in 1:(n_sample + 1) )
	{
		for( j in 1:n_models )
		{
			X_bs[,i,j] = coffee$X[as.character(time_bs[,i,j]),i,"all",j]
		}
	}
	
	## Usefull function to apply ns_params
	
	
	## Correction of mu0
	mu1X = NSSEA::ones(X_bs)
	for( i in 1:n_models ) { mu1X[,,models[i]] = X_bs[,,models[i]] * ns_params_C0["mu1",,models[i]] }
	Yo_bs_mean_corrected  = Yo_bs - mu1X
	ns_params_C0["mu0",,] = base::apply( Yo_bs_mean_corrected , base::c(2,3) , base::mean )
	
	
	## Correction of sig0
	Yo_bs_mu0 = NSSEA::ones(Yo_bs)
	sig1X     = NSSEA::ones(X_bs)
	for( i in 1:n_models )
	{
		Yo_bs_mu0[,,models[i]] = Yo_bs[,,models[i]] - ns_params_C0["mu0",,models[i]]
		sig1X[,,models[i]]     = X_bs[,,models[i]] * ns_params_C0["sig1",,models[i]]
	}
	Yo_bs_full_corrected   = (Yo_bs_mu0 - mu1X) / ( 1. + sig1X )
	ns_params_C0["sig0",,] = base::apply( Yo_bs_full_corrected , base::c(2,3) , stats::sd )
	
	ns_params_C0["sig1",,] = ns_params_C0["sig1",,] * ns_params_C0["sig0",,]
	coffee$ns_params = ns_params_C0
	coffee$Y_C0 = Yo_bs_mean_corrected
	
	if(verbose) cat( "Constraints C0 (Done)        \n" )
	
	return(coffee)
}
##}}}



#CX = function( S_params , obs_x , X , Sigma , n_sample , centering_CX = TRUE , ref_CX = NULL , trust_SY = TRUE ) ##{{{
#{
#	# Time axis
#	years      = as.numeric(dimnames(X)$years)
#	year_obs_x = as.numeric(names(obs_x))
#	
#	# Dimensions
#	n_years     = length(years)
#	n_years_ox  = length(year_obs_x)
#	n_bootstrap = dim(X)[2] - 1
#	n_mm_params = S_params$n_mm_params
#	n_params    = S_params$n_params
#	n_models    = dim(X)[4]
#	
#	# Key to restrict to obs
#	iobs = years %in% year_obs_x
#	
#	# Matrix H: selecting "all", centering, extracting relevant years
#	if( centering_CX )
#	{
#		if( is.null(ref_CX) )
#			ref_CX = year_obs_x
#		if( !base::all( ref_CX %in% year_obs_x ) )
#		{
#			print("WARNING: Reference period for constrain C1 is not acceptable (inconsistency with obs)")
#			return
#		}
#		if( !base::all(ref_CX %in% years) )
#		{
#			print("WARNING: Reference period for constrain C1 is not acceptable (inconsistency with model outputs)")
#			return
#		}
#		Center_X = base::diag( base::rep( 1 , n_years    ) ) - base::rep( 1 , n_years )    %o% ( years      %in% ref_CX ) / length(ref_CX)
#		Center_Y = base::diag( base::rep( 1 , n_years_ox ) ) - base::rep( 1 , n_years_ox ) %o% ( year_obs_x %in% ref_CX ) / length(ref_CX)
#	}
#	else
#	{
#		Center_X  = base::diag( base::rep( 1 , n_years    ) )
#		Center_Y  = base::diag( base::rep( 1 , n_years_ox ) )
#	}
#	Extract_Xall = base::cbind( base::diag( base::rep( 0 , n_years ) ) , base::diag( base::rep( 1 , n_years ) ) , array( 0 , dim = base::c( n_years , n_params ) ) )
#	H_full       = Center_X %*% Extract_Xall		# Centering * extracting
#	H            = H_full[ years %in% year_obs_x,]		# Restriction to the observed period
#	
#	# Other inputs : x, SX, y, SY
#	x  = S_params$mean
#	SX = S_params$cov
#	y  = Center_Y %*% obs_x
#	SY = Center_Y %*% Sigma[iobs,iobs] %*% base::t(Center_Y)
#	
#	# Re-scale SY in order to be consistent with observations obs_x
#	#	We search for lambda s.t. 
#	#		res := y - H%*%x ~ N(0, H%*%SX%*%t(H) + lambda *SY)
#	#	Let iSig := ginv(H%*%SX%*%t(H) + lambda *SY)
#	#	The mle of lambda satisfies
#	#		Tr( iSig %*% SY ) - t(res) %*% iSig%*%SY%*%iSig %*% res = 0
#	#	It can be derived using a newton-Raphson algo...
#	lambda = 1
#	if( !trust_SY )
#	{
#		res   = y - H %*% x
#		vlam  = 10^(base::seq(-2,2,.1))	# Arbitrary choice!
#		nvlam = length(vlam)
#		score = NSSEA::ones(vlam) + NA
#		for( i in 1:nvlam )
#		{
#			SY_tmp = vlam[i]* SY
#			iSig = NSSEA::ginv( H %*% SX %*% base::t(H) + SY_tmp )
#			score[i] = base::sum( base::diag( iSig %*% SY_tmp ) ) - base::t(res) %*% iSig %*% SY_tmp %*% iSig %*% res
#		}
#		lambda = stats::approx( score , vlam , xout = 0 )$y
#	}
#	SYr = lambda * SY
#	
#	# Apply constraint
###	SYr = SYr + diag(c(rep(0,nyox-50),rep(0,50)))
#	post_dist       = NSSEA::generic_constrain( x , SX , y , SYr , H )
#	Sigma_post_sqrt = NSSEA::matrix_sqrt( post_dist$cov )
#	
#	# X_cons		dim=(ny,Nres+1,3)
#	X_cons          = X[,,,1] + NA
#	epsil_cons      = array( 0 , dim = base::c( n_mm_params , n_bootstrap + 1 ) )
#	epsil_cons[,-1] = stats::rnorm( n_mm_params * n_bootstrap )
#	X_cons_tmp      = post_dist$mean %o% base::rep( 1 , n_bootstrap + 1 ) + Sigma_post_sqrt %*% epsil_cons
#	X_cons[,,"nat"] = X_cons_tmp[1:n_years,]
#	X_cons[,,"all"] = X_cons_tmp[n_years + 1:n_years,]
#	
#	# X_Cons		dim=(ny,Nres+1,3,Nmod)
#	X_Cons = X_cons %o% array( base::rep( 1 , n_models ) , dimnames = dimnames(X)[4] )
#	return(X_Cons)
#}
##}}}

#C0 = function( X , NS_param , obs_y , y_distrib , event , return_obs_corr = FALSE ) ##{{{
#{
#	## I need to divide sig1 / sig0
#	# A few checks on X and NS_param
#	if( base::any( dimnames(X)$models != dimnames(NS_param)$models ) )
#		stop("Number of models does not match")
#	
#	n_models   = dim(X)[4]
#	n_runs     = dim(X)[2] ## Bootstrap + best estimate
#	models     = dimnames(X)$models
#	years      = as.numeric(dimnames(X)$years)
#	year_obs_y = as.numeric(names(obs_y))
#	
#	
# 	# Calculate revised NS_param: see scientific documentation
#	unit_time_obs = array( NSSEA::ones( year_obs_y ) , dimnames = list( years = as.character(year_obs_y) ) )
#	X_obs         = NSSEA::dropd( X[years %in% year_obs_y,,"all",,drop = FALSE] , drop = 3 )
#	
#	
#	# New NS_param
#	NS_param_C0 = NS_param
#	
#	
#	# Boostrap on observed years
#	year_obs_y_boot       = array( year_obs_y , dimnames = list( years = as.character(year_obs_y) ) ) %o% NSSEA::ones( NSSEA::dropd( NS_param[1,,,drop = FALSE] , drop = 1 ) )
#	year_obs_y_boot[,-1,] = base::apply( year_obs_y_boot[,-1,,drop=FALSE] , base::c(2,3) , base::sample , replace = TRUE )
#	obs_y_boot	          = array( obs_y[as.character( year_obs_y_boot )] , dim = dim(year_obs_y_boot) , dimnames = dimnames(year_obs_y_boot) )
#	X_boot		          = NSSEA::ones( obs_y_boot) + NA
#	
#	for( i in 1:n_runs )
#	{
#		for( j in 1:n_models ) {
#			X_boot[,i,j] = X[as.character(year_obs_y_boot[,i,j]),i,"all",j]
#		}
#	}
#	
#	# Correction and derive revised NS_param
#	# Correction of the mean
#	NS_param_mu1_conform       = unit_time_obs %o% NSSEA::dropd( NS_param["mu1",,,drop=FALSE] , drop = 1 )
#	obs_y_boot_mean_corrected  = obs_y_boot - NS_param_mu1_conform * X_boot				# y_t - mu_1 * X
#	NS_param_C0["mu0",,]       = base::apply( obs_y_boot_mean_corrected , base::c(2,3) , base::mean )
#	
#	# Correction of mean + variance
#	obs_y_boot_full_corrected  = ( obs_y_boot -	unit_time_obs %o% NSSEA::dropd( NS_param_C0["mu0",,,drop=FALSE] , 1 )	-			# ( y_t - mu_{t_1} -  
#									( unit_time_obs %o% NSSEA::dropd( NS_param_C0["mu1",,,drop=FALSE],1) ) * X_boot )	/			#  mu_1 * X ) /
#									( 1 + ( unit_time_obs %o% NSSEA::dropd(NS_param_C0["sig1",,,drop=FALSE],1) ) * X_boot )		#  ( sig_1 * X )
#	
#	NS_param_C0["sig0",,] = base::apply( obs_y_boot_full_corrected , base::c(2,3) , stats::sd )		
#	
#	
#	if( return_obs_corr )
#		return(obs_y_boot_mean_corrected)
#	
#	return(NS_param_C0)
#}
##}}}

