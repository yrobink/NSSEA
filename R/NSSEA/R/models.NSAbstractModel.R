

###############
## Libraries ##
###############

####################################################################################################
##
## Non-Stationnary Base Model class (DO NOT USE)
##
####################################################################################################

#' Abstract definition of a non-stationary model
#'
#' Empty model, it is a template for other non-stationary model. Can be used to test if a model is a NS model
#'
#' @docType class
#' @importFrom R6 R6Class
#' 
#'
#' @return Object of \code{\link{R6Class}}
#' @format \code{\link{R6Class}} object.
#'
#' @examples
#' ## Define models
#' #mod0 = NSSEA::DiracModel$new(0)
#' mod1 = NSSEA::NSGaussianModel$new()
#' #mod2 = NSSEA::NSGPDModel$new()
#' #mod3 = NSSEA::NSNPModel$new()
#' #mod4 = NSSEA::NSNPTailsModel$new()
#'
#' ## Test
#' "NSAbstractModel" %in% class(mod0) ## TRUE
#' "NSAbstractModel" %in% class(mod1) ## TRUE
#' "NSAbstractModel" %in% class(mod2) ## TRUE
#' "NSAbstractModel" %in% class(mod3) ## TRUE
#' "NSAbstractModel" %in% class(mod4) ## TRUE
#'
NSAbstractModel = R6::R6Class( "NSAbstractModel" ,
	
	public = list(
	
	###############
	## Arguments ##
	###############
	
	
	
	#################
	## Constructor ##
	#################
	
	initialize = function()
	{},
	
	
	###############
	## Accessors ##
	###############
	
	#############
	## Methods ##
	#############
	
	fit = function( Y , X , init = NULL )
	{},
	
	set_covariable = function( X , t = NULL )
	{},
	
	rvs = function(t)
	{},
	
	cdf = function( value , t )
	{},
	
	icdf = function( q , t )
	{},
	
	sf = function( value , t )
	{},
	
	isf = function( q , t )
	{}
	
	), ## Public list
	
	
	
	#############
	## Private ##
	#############
	
	private = list(
	
	
	#######################
	## Private arguments ##
	#######################
	
	#####################
	## Private methods ##
	#####################
	
	
	
	) ##Private list
) 




