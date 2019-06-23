
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import numpy as np
import scipy.linalg as scl


class ProgressBar: ##{{{
	"""
	NSSEA.ProgressBar
	=================
	
	Class used to print a progress bar in screen during execution
	
	
	Example
	-------
	>> pb = ProgressBar( "My task" , 100 )
	>> for i in range(100):
	>>     pb.print()
	>> pb.end()
	"""
	
	def __init__( self , message , size , digits = 2 ):
		"""
		Constructor

		Arguments
		---------
		message: str
			Message printed
		size   :
			Lenght of loop where progress bar will be inserted
		digits :
			Number of digits for the percentage printed
		"""
		self.message = message
		self.size    = size
		self.digits  = digits
		self.iter    = 1
	
	def print(self):
		"""
		Method which print on the screen
		"""
		print( self.message + " ({}%)               ".format( round(self.iter / self.size * 100,self.digits) ) , end = "\r" )
		self.iter += 1
	
	def end(self):
		"""
		Method printed the final message afer loop
		"""
		print( self.message + " (Done)              " )
##}}}


###############
## Functions ##
###############

def matrix_squareroot( M , disp = False ):##{{{
	"""
	NSSEA.matrix_squareroot
	=======================
	Method which compute the square root of a matrix (in fact just call scipy.linalg.sqrtm), but if disp == False, never print warning
	
	Arguments
	---------
	M   : np.array
		A matrix
	disp: bool
		disp error (or not)
	
	Return
	------
	Mp : np.array
		The square root of M
	"""
	Mh = scl.sqrtm( M , disp = disp )
	if not disp:
		Mh = Mh[0]
	return np.real(Mh)
##}}}

def matrix_positive_part( M ):##{{{
	"""
	NSSEA.matrix_positive_part
	==========================
	Return the positive part of a matrix
	
	Arguments
	---------
	M  : np.array
		A matrix
	
	Return
	------
	Mp : np.array
		The positive part of M
	
	"""
	lbda,v = np.linalg.eig(M)
	lbda   = np.real(lbda)
	v      = np.real(v)
	lbda[lbda<0] = 0
	return v @ np.diag(lbda) @ v.T
##}}}

def barycenter_covariance( lcov , weights = None , maxit = 50 , tol = 1e-3 , verbose = False ):##{{{
	"""
	NSSEA.barycenter_covariance
	===========================
	Compute the barycenter (in optimal transport sense) of covariance matrices
	
	Arguments
	---------
	lcov   : list[np.array]
		A list of covariance matrix
	weights: array
		Weights of each covariance matrices, if None 1/size is used
	maxit  : integer
		Max number of iterate. Default is 50
	tol    : float
		Numerical tolerance, default is 1e-3
	verbose : bool
		Print (or not) state of execution
	Return
	------
	cov   : np.array
		A covariance matrix barycenter of lcov
	"""
	
	n_cov = len(lcov)
	
	if weights is None:
		weights = np.ones( n_cov )
	weights = np.array(weights) / np.sum(weights)
	
	def brower( S , lcov , weights ):
		root_S = matrix_squareroot(S)
		Sn = np.zeros( lcov[0].shape )
		
		for i in range(n_cov):
			Sn += weights[i] * matrix_squareroot( root_S @ lcov[i] @ root_S )
		
		return Sn
	
	cov  = np.identity(lcov[0].shape[0])
	diff = 1. + tol
	nit  = 0
	
	while diff > tol and nit < maxit:
		if verbose: print( "Optimal barycenter: {} (<{}), {} (>{})                                        ".format(nit,maxit,round(diff,4),tol) , end = "\r" )
		covn = brower( cov , lcov , weights )
		diff = np.linalg.norm(cov - covn)
		cov  = covn
		nit += 1 
	
	if verbose: print( "Optimal barycenter (Done)                                                    " )
	return cov
##}}}


