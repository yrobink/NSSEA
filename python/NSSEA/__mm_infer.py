
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import numpy        as np
import xarray       as xr

from NSSEA.__tools import matrix_positive_part
from NSSEA.__tools import matrix_squareroot
from NSSEA.__tools import barycenter_covariance


#############
## Classes ##
#############

class MMStats:
	"""
	NSSEA.MMStats
	=============
	Class infering multimodel parameters. Use NSSEA.infer_multi_model to build MMStats
	
	
	Attributes
	----------
	mean   : array
		Multi model mean
	cov    : array
		Multi model covariance matrix
	std    : array
		Square root of multimodel covariance matrix
	method : str
		Method used ("classic" or "optimal")
	"""
	
	def __init__( self , mm_matrix = None , method = "classic" , verbose = False ):
		"""
		Constructor of MMStats
		
		Parameters
		----------
		mm_matrix: array
			Big matrix containing sample to infer multi model parameters
		method   : str
			Method used, "classic" or "optimal"
		verbose  : bool
			Print (or not) state of execution
		"""
		
		self.mean = None
		self.cov  = None
		self.std  = None
		self.method = method
	
		if mm_matrix is not None:
			self._initialize( mm_matrix , verbose )
	
	
	def _initialize( self , mm_matrix , verbose ):
		n_params,n_sample,n_models = mm_matrix.shape
		
		self.mean = np.mean( mm_matrix[:,0,:] , axis = 1 )
		
		if self.method == "classic":
			cov_S = np.zeros( (n_params,n_params) )
			
			for i in range(n_models):
				cov_S += np.cov( mm_matrix[:,1:,i] )
			
			SSM     = np.cov( mm_matrix[:,0,:] ) * ( n_models - 1 )
			cov_CMU = matrix_positive_part( SSM / ( n_models - 1 ) - cov_S / n_models )
			self.cov  = ( n_models + 1 ) / n_models * cov_CMU + cov_S / n_models**2
		elif self.method == "empirical":
			self.cov = np.cov( mm_matrix.reshape( (n_params,n_sample*n_models) ) )
		else:
			lcov = [ np.cov( mm_matrix[:,:,i] ) for i in range(n_models) ]
			self.cov = barycenter_covariance( lcov , verbose = verbose )
		
		self.std = matrix_squareroot(self.cov)
	
	def copy(self):
		"""
		Return a copy
		"""
		mm = MMStats( None , self.method )
		mm.mean = self.mean.copy()
		mm.cov  = self.cov.copy()
		mm.std  = self.std.copy()
		return mm


###############
## Functions ##
###############

def infer_multi_model( climIn , mm_method = "classic" , verbose = False ):
	"""
	NSSEA.infer_multi_model
	=======================
	Infer multimodel mean and covariance
	
	Arguments
	---------
	climIn : NSSEA.Climatology
		clim variable
	mm_method: str
		Multi model method, currently "classic" (A. Ribes method) or "optimal" (Optimal transport)
	verbose  : bool
		Print (or not) state of execution
	
	Return
	------
	clim: NSSEA.Climatology
		A COPY of the input clim, where clim.mm_params is set, and clim.X contains multi model sample. The input clim IS NOT MODIFIED.
	
	Remark
	------
	clim.mm_params is a NSSEA.MMStats class
	
	"""
	if verbose: print( "Multi model" , end = "\r" if mm_method == "classic" else "\n" )
	## Parameters
	clim      = climIn.copy()
	n_time      = clim.n_time
	n_ns_params = clim.n_ns_params
	n_sample    = clim.n_sample
	n_models    = clim.n_models
	sample      = clim.X.sample.values.tolist()
	n_mm_params = 2 * n_time + n_ns_params
	
	## Big matrix
	S                        = np.zeros( (n_mm_params,n_sample + 1,n_models) )
	S[:n_time,:,:]           = clim.X.loc[:,:,"all",:].values
	S[n_time:(2*n_time),:,:] = clim.X.loc[:,:,"nat",:].values
	S[(2*n_time):,:,:]       = clim.ns_params.values
	
	## Multi model parameters inference
	clim.mm_params   = MMStats( S , mm_method , verbose )
	clim.n_mm_params = n_mm_params
	
	## Generate sample
	mm_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,3,1) )    , coords = [ clim.X.time , sample , clim.X.forcing , ["multi"] ] , dims = ["time","sample","forcing","models"] )
	mm_params = xr.DataArray( np.zeros( (n_ns_params,n_sample + 1,1) ) , coords = [ clim.ns_params.ns_params , sample , ["multi"] ]       , dims = ["ns_params","sample","models"] )
	
	mm_sample.loc[:,"be","all","multi"] = clim.mm_params.mean[:n_time]
	mm_sample.loc[:,"be","nat","multi"] = clim.mm_params.mean[n_time:(2*n_time)]
	mm_params.loc[:,"be","multi"]       = clim.mm_params.mean[(2*n_time):]
	
	for s in sample[1:]:
		draw = clim.mm_params.mean + clim.mm_params.std @ np.random.normal(size = n_mm_params)
		mm_sample.loc[:,s,"all","multi"] = draw[:n_time]
		mm_sample.loc[:,s,"nat","multi"] = draw[n_time:(2*n_time)]
		mm_params.loc[:,s,"multi"]       = draw[(2*n_time):]
	
	mm_sample.loc[:,:,"ant","multi"] = mm_sample.loc[:,:,"all","multi"] - mm_sample.loc[:,:,"nat","multi"]
	
	## Add multimodel to clim
	clim.X         = xr.concat( [clim.X , mm_sample] , "models" )
	clim.ns_params = xr.concat( [clim.ns_params,mm_params] , "models" )
	clim.n_models += 1
	clim.models.append( "multi" )
	
	if verbose: print( "Multi model (Done)" )
	
	return clim

