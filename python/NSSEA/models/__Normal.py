# -*- coding: utf-8 -*-


###############
## Libraries ##
###############

import numpy             as np
import scipy.stats       as sc
import scipy.interpolate as sci
import SDFC              as sd
import SDFC.tools        as sdt

from NSSEA.models.__NSAbstractModel import NSAbstractModel


#############
## Classes ##
#############

class Normal(NSAbstractModel):
	"""
	NSSEA.models.NSGaussianModel
	============================
	Non stationary Gaussian model. At each time, the law is a Gaussian law.
	We assume the mean (mu) and the scale (scale) can be written:
	mu(t) = mu0 + mu1 * X
	scale(t) = scale0 + scale1 * X
	Where X is a co-variable given in the fit function.
	
	
	Attributes
	----------
	mu0    : float
		First parameter of the mean of the NS Gaussian law
	mu1    : float
		Second parameter of the mean of the NS Gaussian law
	scale0 : float
		First parameter of the standard deviation of the NS Gaussian law
	scale1 : float
		Second parameter of the standard deviation of the NS Gaussian law
	"""
	
	#################
	## Constructor ##
	#################
	
	def __init__( self , link_loc = sdt.IdLink() , link_scale = sdt.ExpLink() , method = "MLE" , verbose = False ): ##{{{
		"""
		Initialization of the NS Gaussian Model
		"""
		NSAbstractModel.__init__(self)
		self._norm = sd.Normal( method = method )
		self._verbose = verbose
		self._link = { "loc" : link_loc , "scale" : link_scale }
		
		self.mu0     = None
		self.mu1     = None
		self.scale0  = None
		self.scale1  = None
		self._mut    = None
		self._scalet = None
	##}}}
	
	def default_arg( arg = None ):##{{{
		"""
		Dictionary of arguments of __init__
		
		Parameters
		----------
		arg : None or dict
			dictionary of arguments already fixed
		
		Returns
		-------
		default: dict
			Arguments of __init__, elements of "arg" are kept
		"""
		default = { "link_loc" : sdt.IdLink() , "link_scale" : sdt.ExpLink() , "method" : "MLE" , "verbose" : False }
		if arg is not None:
			for key in arg:
				default[key] = arg[key]
		return default
	##}}}
	
	def params_info( arg = None ):##{{{
		"""
		Dictionary containing size of ns params and character names
		
		Parameters
		----------
		arg : None or dict
			dictionary of arguments already fixed
		
		Returns
		-------
		default: dict
			The key "size" contains the size, the key "names" contains a list of names
		"""
		return { "size" : 4 , "names" : ["loc0","loc1","scale0","scale1"] }
	##}}}
	
	def link_fct_by_params( self ):##{{{
		return [self._link["loc"],self._link["loc"],self._link["scale"],self._link["scale"]]
	##}}}
	
	
	###############
	## Accessors ##
	###############
	
	def meant( self , t ): ##{{{
		"""
		Mean of the Gaussian Model at time t
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		mu : np.array
			Mean at time t
		"""
		return self._mut(t)
	##}}}
	
	def mediant( self , t ):##{{{
		return self.meant(t)
	##}}}
	
	def mut( self , t ): ##{{{
		"""
		Loc parameters of the Gaussian Model at time t
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		mu : np.array
			Mean at time t
		"""
		return self._mut(t)
	##}}}
	
	def scalet( self , t ):##{{{
		"""
		Standard deviation of the Gaussian Model at time t
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		scale : np.array
			Standard deviation at time t
		"""
		return self._scalet(t)
	##}}}
	
	def get_params( self ):##{{{
		"""
		Return a vector of the coefficients fitted, i.e.:
		np.array( [mu0,mu1,scale0,scale1] )
		"""
		return np.array( [self.mu0,self.mu1,self.scale0,self.scale1] )
	##}}}
	
	def set_params( self , coef_ ): ##{{{
		self.mu0    = coef_[0]
		self.mu1    = coef_[1]
		self.scale0 = coef_[2]
		self.scale1 = coef_[3]
	#}}}
	
	def check( self , Y , X , t = None ):##{{{
		return True
	##}}}
	
	
	#############
	## Methods ##
	#############
	
	def fit( self , Y , X ):##{{{
		"""
		Fit of the NS Gaussian law from a dataset Y and a covariable X discribing non-stationarity of Y
		
		Parameters
		----------
		Y : np.array
			Dataset to fit
		X : np.array
			Covariable
		
		Notes
		-----
		To use the model, the method set_covariable must be called by user after the fit
		"""
		self._norm.fit( Y , c_loc = X , l_loc = self._link["loc"] , c_scale = X , l_scale = self._link["scale"] )
		
		self.mu0    = self._norm.coef_[0]
		self.mu1    = self._norm.coef_[1]
		self.scale0 = self._norm.coef_[2]
		self.scale1 = self._norm.coef_[3]
	##}}}
	
	def drawn_bayesian( self , Y , X  , n_mcmc_drawn , prior ):##{{{
		norm = sd.Normal( method = "bayesian" )
		norm.fit( Y , c_loc = X , l_loc = self._link["loc"] , c_scale = X , l_scale = self._link["scale"] , n_mcmc_drawn = n_mcmc_drawn , prior = prior )
		return norm._info.draw
	##}}}
	
	def set_covariable( self , X , t = None ):##{{{
		"""
		Set the covariable of the model.
		
		Parameters
		----------
		X : np.array
			Covariable
		t : None or np.array
			Time, if None t = np.arange(0,X.size)
		
		"""
		t = t if t is not None else np.arange( 0 , X.size )
		self._mut    = sci.interp1d( t , self._link["loc"](  self.mu0    + X.ravel() * self.mu1   ) )
		self._scalet = sci.interp1d( t , self._link["scale"](self.scale0 + X.ravel() * self.scale1) )
	##}}}
	
	def rvs( self , t ):##{{{
		"""
		Random value generator
		
		Parameters
		----------
		t : np.array
			Time
		
		Returns
		-------
		Y : np.array
			A time series following the NS law
		"""
		return sc.norm.rvs( size = t.size , loc = self.mut(t) , scale = self.scalet(t) )
	##}}}
	
	def cdf( self , Y , t ):##{{{
		"""
		Cumulative Distribution Function (inverse of quantile function)
		
		Parameters
		----------
		Y : np.array
			Value to estimate the CDF
		t : np.array
			Time
		
		Returns
		-------
		q : np.array
			CDF value
		"""
		return sc.norm.cdf( Y , loc = self.mut(t) , scale = self.scalet(t) )
	##}}}
	
	def icdf( self , q , t ):##{{{
		"""
		inverse of Cumulative Distribution Function 
		
		Parameters
		----------
		q : np.array
			Values to estimate the quantile
		t : np.array
			Time
		
		Returns
		-------
		Y : np.array
			Quantile
		"""
		return sc.norm.ppf( q , loc = self.mut(t) , scale = self.scalet(t) )
	##}}}
	
	def sf( self , Y , t ):##{{{
		"""
		Survival Function (1-CDF)
		
		Parameters
		----------
		Y : np.array
			Value to estimate the survival function
		t : np.array
			Time
		
		Returns
		-------
		q : np.array
			survival value
		"""
		return sc.norm.sf( Y , loc = self.mut(t) , scale = self.scalet(t) )
	##}}}
	
	def isf( self , q , t ):##{{{
		"""
		inverse of Survival Function
		
		Parameters
		----------
		q : np.array
			Values to estimate the quantile
		t : np.array
			Time
		
		Returns
		-------
		Y : np.array
			values
		"""
		return sc.norm.isf( q , loc = self.mut(t) , scale = self.scalet(t) )
	##}}}
	



