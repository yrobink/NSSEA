
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
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

