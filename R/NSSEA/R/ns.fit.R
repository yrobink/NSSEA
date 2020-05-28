
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



