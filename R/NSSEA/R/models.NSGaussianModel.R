
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

###############
## Libraries ##
###############

####################################################################################################
##
## Non-Stationnary Gaussian Model class
##
####################################################################################################

#' Non-stationary Gaussian Model
#'
#' Model a non stationary Gaussian law where non-stationarity is given by a co-variable.
#'
#' @docType class
#' @importFrom R6 R6Class
#' @importFrom SDFC NormalLaw
#' @importFrom SDFC IdLinkFct
#' @importFrom SDFC ExpLinkFct
#' 
#' @param Y [vector]
#'        Dataset inferred by the model
#' @param X [vector]
#'        Covariable of the dataset Y
#' @param t [vector]
#'        Time axis
#' @param q [vector]
#'        Quantiles in [0,1]
#' @param init [NULL or vector]
#'        Initialization for the optimization of the likelihood, if NULL, estimated during the fit.
#'
#' @return Object of \code{\link{R6Class}}
#' @format \code{\link{R6Class}} object.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new()}}{This method is used to create object of this class with \code{NSGaussianModel}}
#'   \item{\code{fit(Y,X,init)}}{Fit the model from Y and X}
#'   \item{\code{set_covariable(X,t)}}{Set the covariable of the model}
#'   \item{\code{mut(t)}}{Mean of the Gaussian law at time t}
#'   \item{\code{sigmat(t)}}{Standard devian of the Gaussian law at time t}
#'   \item{\code{rvs(t)}}{Random values generator at each time t}
#'   \item{\code{cdf(Y,t)}}{Cumulative Distribution Function of values Y at time t}
#'   \item{\code{icdf(q,t)}}{Inverse of CDF of quantiles values q at time t}
#'   \item{\code{sf(Y,t)}}{Survival Function (1-CDF) of values Y at time t}
#'   \item{\code{isf(q,t)}}{Inverse of sf of quantiles values q at time t}
#' }
#' @examples
#' ## Define a Gaussian dataset
#'
#' @export
NSGaussianModel = R6::R6Class( "NSGaussianModel" ,
	
	inherit = NSAbstractModel,
	
	
	public = list(
	
	###############
	## Arguments ##
	###############
	
	mu0     = NULL,
	mu1     = NULL,
	scale0  = NULL,
	scale1  = NULL,
	norm    = NULL,
	
	#################
	## Constructor ##
	#################
	
	initialize = function( link_fct_loc = IdLinkFct$new() , link_fct_scale = ExpLinkFct$new() , method = "MLE" ) ##{{{
	{
		self$norm = SDFC::NormalLaw$new( link_fct_loc = link_fct_loc , link_fct_scale = link_fct_scale , method = method )
	},
	##}}}
	
	default_arg = function( args = NULL ) ##{{{
	{
		da = list(  link_fct_loc = IdLinkFct$new() , link_fct_scale = ExpLinkFct$new() , method = "MLE" )
		if( !is.null(args) )
		{
			for( m in base::names(args) )
			{
				if( m %in% base::names(da) )
					da[[m]] = args[[m]]
			}
		}
		return( da )
	},
	##}}}
	
	params_info = function( args = NULL )##{{{
	{
		return( list( n = 4 , names = base::c( "mu0" , "mu1" , "sig0" , "sig1" ) ) )
	},
	##}}}
	
	
	###############
	## Accessors ##
	###############
	
	meant = function(t) ##{{{
	{
		return( private$mut_(t) )
	},
	##}}}
	
	mediant = function(t) ##{{{
	{
		return( private$mut_(t) )
	},
	##}}}
	
	mut = function(t) ##{{{
	{ return(private$mut_(t)) },
	##}}}
	
	scalet = function(t) ##{{{
	{ return(private$scalet_(t)) },
	##}}}
	
	get_params = function() ##{{{
	{
		params = base::c( self$mu0 , self$mu1 , self$scale0 , self$scale1 )
		names(params) = base::c( "mu0" , "mu1" , "scale0" , "scale1" )
		return( params )
	},
	##}}}
	
	set_params = function( params ) ##{{{
	{
		self$mu0    = params[1]
		self$mu1    = params[2]
		self$scale0 = params[3]
		self$scale1 = params[4]
	},
	##}}}
	
	
	#############
	## Methods ##
	#############
	
	fit = function( Y , X ) ##{{{
	{
		self$norm$fit( Y , loc_cov = X , scale_cov = X )
		self$mu0    = self$norm$loc_$coef_[1]
		self$mu1    = self$norm$loc_$coef_[2]
		self$scale0 = self$norm$scale_$coef_[1]
		self$scale1 = self$norm$scale_$coef_[2]
	},
	##}}}
	
	set_covariable = function( X , t = NULL ) ##{{{
	{
		if( is.null(t) )
		{
			t = base::seq( 1 , length(X) )
		}
		
		private$mut_    = stats::approxfun( t , self$norm$loc_$linkFct$eval( self$mu0 + X * self$mu1 ) )
		private$scalet_ = stats::approxfun( t , self$norm$scale_$linkFct$eval( self$scale0 + X * self$scale1 ) )
	},
	##}}}
	
	rvs = function(t) ##{{{
	{
		return( stats::rnorm( length(t) , mean = private$mut_(t) , sd = private$scalet_(t) ) )
	},
	##}}}
	
	cdf = function( Y , t ) ##{{{
	{
		return( stats::pnorm( Y , mean = private$mut_(t) , sd = private$scalet_(t) ) )
	},
	##}}}
	
	icdf = function( q , t ) ##{{{
	{
		q[ !(q>0) ] = .Machine$double.eps
		q[ !(q<1) ] = 1. - .Machine$double.eps
		return( stats::qnorm( q , mean = private$mut_(t) , sd = private$scalet_(t) ) )
	},
	##}}}
	
	sf = function( Y , t ) ##{{{
	{
		return( stats::pnorm( Y , mean = private$mut_(t) , sd = private$scalet_(t) , lower.tail = FALSE ) )
	},
	##}}}
	
	isf = function( q , t ) ##{{{
	{
		return( stats::qnorm( q , mean = private$mut_(t) , sd = private$scalet_(t) , lower.tail = FALSE ) )
	}
	##}}}
	
	), ## Public list
	
	
	
	#############
	## Private ##
	#############
	
	private = list(
	
	
	#######################
	## Private arguments ##
	#######################
	
	mut_    = NULL,
	scalet_ = NULL
	
	) ##Private list
) ## NSGaussianModel



