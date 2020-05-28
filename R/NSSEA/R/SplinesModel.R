
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


########################################
## Re-implementation of base_spline.R ##
## and proj_dl.R from Aurelien Ribes  ##
########################################

###############
## Libraries ##
###############

#library(R6)


#############
## Classes ##
#############

#' SplinesModels
#'
#' Class to construct splines
#'
#' @docType class
#' @importFrom R6 R6Class
#'
#' @param n_knots [integer] Numbers of knots
#' @param weight [double] weights
#' @param dof [double] degree of freedom
#' @param tol [double] Numerical tolerance, default 1e-2
#' @param maxit [integer] max number of iteration during smoothing, default 1000
#'
#' @return Object of \code{\link{R6Class}}
#' @format \code{\link{R6Class}} object.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new(n_knots,weights,dof,tol,maxit)}}{This method is used to create object of this class with \code{SplinesModel}}
#'   \item{\code{projection(y)}}{Return the projection of y by the spline basis}
#' }
#' @examples
#' size = 10
#' pb = ProgressBar$new( "Progress" , size )
#' for( i in 1:size )
#' 	pb$print()
#' pb$end()
#'
#' @export
SplinesModel = R6::R6Class( "SplinesModel" ,
	
	############
	## Public ##
	############
	
	public = list(
	
	
	###############
	## Arguments ##
	###############
	
	basis_size = 0,
	n_knots = 0,
	rho = 0,
	dof = 0,
	weight = NULL,
	tol = NULL,
	maxit = NULL,
	projection_matrix = NULL,
	
	
	#################
	## Constructor ##
	#################
	
	initialize = function( n_knots , weight = 1. , dof = NULL , tol = 1e-2 , maxit = 1000 )
	{
		self$n_knots = n_knots
		self$weight  = weight
		self$dof     = dof
		self$tol     = tol
		self$maxit   = maxit
		
		if( length(self$weight) == 1 )
		{
			self$weight = base::rep( 1 , self$n_knots )
		}
		if( length(self$weight) != self$n_knots )
		{
			base::stop("SplinesModel::init : Length of weight is not correct")
		}
		
		private$build_base()
		
	},
	
	
	#############
	## Methods ##
	#############
	
	projection = function( y )
	{
		return( self$projection_matrix %*% y )
	}
	
	),
	
	private = list(
	
	###############
	## Arguments ##
	###############
	
	gram_schmidt_matrix = NULL,
	splines_basis = NULL,
	
	
	#############
	## Methods ##
	#############
	
	build_base = function()
	{
		## Base spline : 
		x = 1:self$n_knots	                # Les knots
		self$basis_size = self$n_knots + 2	# Dimension de la base spline
		
		X = matrix( data = 0 , nrow = self$basis_size , ncol = 1 )
		X[1,1] = x[1]
		X[2,1] = base::mean(x[1:2])
		X[3:( self$basis_size - 2 ),1] = x[2:(self$n_knots-1)]
		X[self$basis_size-1,1] = base::mean(x[(self$n_knots-1):self$n_knots])
		X[self$basis_size,1] = x[self$n_knots]
		
		Y0 = base::diag( x = 1 ,nrow = self$basis_size , ncol = self$basis_size )
		Y1 = matrix( data = 0 , nrow = self$basis_size , ncol = self$basis_size )
		Y2 = matrix( data = 0 , nrow = self$basis_size , ncol = self$basis_size )
		Y3 = matrix( data = 0 , nrow = self$basis_size , ncol = self$basis_size )
		for( i in 1:self$basis_size )
		{
			f = stats::splinefun( X , Y0[,i] , method = "fmm" ) 
			Y1[,i] = f( X , deriv = 1 )
			Y2[,i] = f( X , deriv = 2 )
			Y3[,i] = f( X , deriv = 3 )
		}
		
		
		# Construct the final weights ww
		w_interv = base::apply( base::cbind( self$weight[1:(self$n_knots-1)] , self$weight[2:self$n_knots] ) , 1 , base::mean )
		ww = w_interv[ base::c( 1 , 1:(self$n_knots-1) , self$n_knots-1 ) ]
		
		## 
		private$splines_basis = Y0[ -base::c(2,self$basis_size-1) ,]
		private$gram_schmidt_matrix = matrix( data = 0 , nrow = self$basis_size , ncol = self$basis_size ) 
		
		mat_one = matrix( data = 1 , nrow = 1 , ncol = length(Y2[1,]) )
		Interv = matrix( data = X[2:self$basis_size] - X[1:(self$basis_size-1)] , nrow = self$basis_size - 1 , ncol = 1 ) %*% mat_one
		
		for( i in 1:self$basis_size )
		{
			## nombre de noeuds copies en colonnes des derivee secondes et troisiemes en les noeuds sauf le dernier pour la i-ieme fct de base spline
			f2 = Y2[1:(self$basis_size-1),i] %*% mat_one
			f3 = Y3[1:(self$basis_size-1),i] %*% mat_one
			
			## Derivee des fonction de base arrangee en colonnes sur tout les noeud sauf le dernier
			g2 = Y2[1:(self$basis_size-1),]
			g3 = Y3[1:(self$basis_size-1),]
			
			## Pour comprendre ce calcul, il faut revenir a la definition de G et developper le produit des derivees secondes sur chaque intervalle
			Int = f2 * g2 *Interv + (f2 * g3 + g2 * f3) * Interv^2 / 2 + f3 * g3 * Interv^3/3
			
			## Sommation sur les colonnes de Int
			private$gram_schmidt_matrix[i,] = base::apply( ww * Int , 2 , base::sum )   
		}
		
		self$dof = if( is.null(self$dof) ) self$n_knots else self$dof
		
		private$dof_projection()
	},
	
	
	dof_projection = function()
	{
		
		# Initialise
		rho_up = 1e7
		rho_lo = 1e-2
		n_it = 0
		
		## Main loop
		while( ( rho_up / rho_lo > ( 1 + self$tol ) ) && (n_it < self$maxit) )
		{
			self$rho = ( rho_up + rho_lo ) / 2.
			
			Hn = base::t(private$splines_basis) %*% private$splines_basis + self$rho * private$gram_schmidt_matrix
			self$projection_matrix = private$splines_basis %*% base::solve(Hn) %*% base::t(private$splines_basis)
			
			dof_new = base::sum( base::diag( self$projection_matrix ) )
			if( dof_new > self$dof )
			{
				rho_lo = self$rho
			}
			else
			{ 
				rho_up = self$rho 
			}
			n_it = n_it + 1
		}
		
		self$dof = dof_new
		Hn = base::t(private$splines_basis) %*% private$splines_basis + self$rho * private$gram_schmidt_matrix
		self$projection_matrix = private$splines_basis %*% base::solve(Hn) %*% base::t(private$splines_basis)
	}
	
	)
	
)

