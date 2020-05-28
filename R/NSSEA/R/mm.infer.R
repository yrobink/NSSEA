 
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

## mm_stats {{{

#' mm_stats
#'
#' Multi model estimator
#'
#' @usage mm_stats( S , method , verbose )
#'
#' @param S [array] dim = (n_params,n_sample,n_model)
#' @param method [str] If "classic", use aribes methods, else use optimal transport barycenter. Default "classic"
#' @param verbose [bool] print or not state of execution
#'
#' @return mm [list] mm$mean is the mean, mm$cov the covariance and mm$std the standard deviation
#'
#' @examples
#' ##
#' 
#' @export
mm_stats = function( S , method , verbose = FALSE )
{
	n_params = dim(S)[1]
	n_sample = dim(S)[2]
	n_model  = dim(S)[3]
	
	## MM Mean
	Mean_MM = base::apply( S[,1,] , 1 , base::mean )
	
	if( method == "classic" )
	{
		## MM Var
		Var_BS = matrix( 0 , nrow = n_params , ncol = n_params )
		for( i in 1:n_model )
		{
			Var_BS = Var_BS + stats::var( base::t(S[,-1,i]) )
		}
		
		SSM = stats::var( base::t(S[,1,] ) ) * ( n_model - 1 )
		Var_CMU = NSSEA::matrix_positive_part( SSM / ( n_model - 1 ) - Var_BS / n_model )
		Var_MM = ( n_model + 1 ) / n_model * Var_CMU + Var_BS / n_model^2
	}
	else
	{
		lcov = array( NA , dim = base::c( n_model , n_params , n_params ) )
		for( i in 1:n_model )
		{
			lcov[i,,] = stats::var( base::t(S[,,i]) )
		}
		Var_MM = NSSEA::barycenter_covariance( lcov , verbose = verbose )
	}
	Std_MM = NSSEA::matrix_sqrt(Var_MM)
	
	
	return( list( mean = Mean_MM , cov = Var_MM , std = Std_MM ) )
}
##}}}

## infer_multi_model {{{

#' infer_multi_model
#'
#' Add multimodel to the coffee
#'
#' @importFrom abind abind
#'
#' @usage infer_multi_model( coffee , method , verbose )
#'
#' @param coffee [coffee variable]
#' @param method [str] If "classic", use aribes methods, else use optimal transport barycenter. Default "classic"
#' @param verbose [bool] print or not state of execution
#'
#' @return coffee [coffee variable] coffee with multi model
#'
#' @examples
#' ##
#' 
#' @export
infer_multi_model = function( coffee , method = "classic" , verbose = FALSE )
{
	if( verbose )
	{
		end_cat = if( method == "classic" ) "\r" else "\n"
		cat( "Multi model" , end = end_cat )
	}
	n_time      = coffee$n_time
	n_ns_params = coffee$n_ns_params
	n_sample    = coffee$n_sample
	n_mm_params = 2 * n_time + n_ns_params
	
	# Put Xnat, Xant and ns_params into one single big vector / matrix for multi-model calculation
	mm_matrix = abind::abind( coffee$X[,,"nat",] , coffee$X[,,"all",] , coffee$ns_params , along = 1 )
	mm_params = NSSEA::mm_stats( mm_matrix , method , verbose )
	
	# Generating realisations
#	mm_sample = matrix( 0 , n_mm_params , n_sample )
#	for( i in 1:n_sample )
#	{
#		mm_sample[,i] = mm_params$mean + mm_params$std %*% stats::rnorm(n_mm_params)
#	}
	
	mm_sample = base::apply( matrix( stats::rnorm( n_mm_params * n_sample ) , nrow = n_sample ) , 1 , function(x) { return( mm_params$mean + mm_params$std %*% x ) } )
	
	# Add multi-model to X
	M = array(	NA ,
					dim = base::c( n_time , n_sample + 1 , 2 , 1 ) ,
					dimnames = list( time = coffee$time , sample = dimnames(coffee$X)$sample , forcing = dimnames(coffee$X)$forcing , models = "multi" )
				)
	
	M[,"be","nat","multi"] = mm_params$mean[1:n_time]
	M[,"be","all","multi"] = mm_params$mean[ n_time + (1:n_time) ]
	M[,-1,"nat","multi"]   = mm_sample[1:n_time,]
	M[,-1,"all","multi"]   = mm_sample[n_time + (1:n_time),]
	coffee$X = abind::abind( coffee$X , M , along = 4 , use.dnns = TRUE )	# option 'use.dnns' allows use of names(dimnames())...
	
	# Add multi-model to ns_params
	N = array( NA , dim = base::c( n_ns_params , n_sample + 1 , 1 ) , dimnames = base::c( dimnames(coffee$ns_params)[1:2] , list( models = "multi" ) ) )
	N[,"be","multi"] = mm_params$mean[ 2 * n_time + (1:n_ns_params)]
	N[,-1,"multi"]   = mm_sample[ 2 * n_time + (1:n_ns_params),]
	coffee$ns_params = abind::abind( coffee$ns_params , N , along = 3 , use.dnns = TRUE )
	
	coffee$models      = base::c( coffee$models , "multi" )
	coffee$n_models    = coffee$n_models + 1
	coffee$mm_params   = mm_params
	coffee$n_mm_params = n_mm_params
	
	if( verbose ) cat("Multi model (Done)\n")
	
	return(coffee)
}
##}}}



