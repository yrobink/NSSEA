
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

## make_coffee {{{

#' make_coffee
#'
#' Build the coffee variable containing results of execution of NSSEA
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


