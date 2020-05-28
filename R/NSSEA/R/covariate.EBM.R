
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



