# -*- coding: utf-8 -*-

###############
## Libraries ##
###############

import numpy             as np
import scipy.stats       as sc
import scipy.interpolate as sci
import scipy.special     as scs
import SDFC              as sd

from NSSEA.models.__NSAbstractModel import NSAbstractModel

import SDFC as sd
import SDFC.tools as sdt


#############
## Classes ##
#############



class NSGEVModel(NSAbstractModel):
	"""
	NSModel.NSGEVModel
	==================
	Non stationary Generalized Extreme value model. At each time, the law is a Generalized Extreme value law.
	We assume the location (loc), the scale (scale) and the shape (shape) can be written:
	loc(t) = loc0 + loc1 * X
	scale(t) = scale0 + scale1 * X
	shape(t) = constant
	Where X is a co-variable given to the fit function.
	
	
	Attributes
	----------
	loc        : np.array
		Loc at each time
	scale      :
		Scale at each time
	shape      :
		Shape at each time
	
	Masked (but maybe usefull) attributes
	-------------------------------------
	_loc0   : float
		First parameter of loc
	_loc1   : float
		Second parameter of loc
	_scale0 : float
		First parameter of the scale
	_scale1 : float
		Second parameter of the scale
	_shape  : float
		Value of shape estimated
	"""
	
	#################
	## Constructor ##
	#################
	
	def __init__( self , tails = "upper" , link_fct_loc = sdt.IdLinkFct() , link_fct_scale = sdt.ExpLinkFct() , link_fct_shape = sdt.LogitLinkFct( -0.5 , 0.5 ) , method = "MLE" , verbose = False ): ##{{{
		"""
		"""
		NSAbstractModel.__init__(self)
		self._tails   = 1 if tails == "upper" else -1
		self._gev     = sd.GEVLaw( method = method , link_fct_loc = link_fct_loc , link_fct_scale = link_fct_scale , link_fct_shape = link_fct_shape )
		self._verbose = verbose
		
		self._loc0    = None
		self._loc1    = None
		self._scale0  = None
		self._scale1  = None
		self._shape   = None
		
		self.loc      = None
		self.scale    = None
		self.shape    = None
		
		self._loct    = None
		self._scalet  = None
		self._shapet  = None
		
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
		default = { "tails" : "upper" , "link_fct_loc" : sdt.IdLinkFct() , "link_fct_scale" : sdt.ExpLinkFct() , "link_fct_shape" : sdt.LogitLinkFct( -0.5 , 0.5 ) , "method" : "MLE" , "verbose" : False }
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
		return { "size" : 5 , "names" : ["loc0","loc1","scale0","scale1","shape"] }
	##}}}
	
	def link_fct_by_params( self ):##{{{
		return [self._gev._loc.linkFct,self._gev._loc.linkFct,self._gev._scale.linkFct,self._gev._scale.linkFct,self._gev._shape.linkFct]
	##}}}
	
	
	#########
	## Fit ##
	#########
	
	def fit( self , Y , X ):##{{{
		"""
		Fit of the NS Generalized extreme value law from a dataset Y and a covariable X discribing non-stationarity of Y
		
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
		
		## Fit
		self._gev.fit( self._tails * Y , loc_cov = self._tails * X , scale_cov = self._tails * X )
		
		## Result
		self._loc0   = self._gev._loc.coef_[0]
		self._loc1   = self._gev._loc.coef_[1] if self._gev._loc.size == 2 else 0.
		self._scale0 = self._gev._scale.coef_[0]
		self._scale1 = self._gev._scale.coef_[1] if self._gev._scale.size == 2 else 0.
		self._shape  = self._gev._shape.coef_[0]
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
		self.loc   = self._gev._loc.linkFct(   self._loc0 + self._tails * self._loc1 * X )
		self.scale = self._gev._scale.linkFct( self._scale0 + self._tails * self._scale1 * X )
		self.shape = self._gev._shape.linkFct( np.repeat( self._shape , X.size ) )
		self._loct   = sci.interp1d( t , self.loc )
		self._scalet = sci.interp1d( t , self.scale )
		self._shapet = sci.interp1d( t , self.shape )
	##}}}
	
	def get_params( self ):##{{{
		"""
		Return a vector of the coefficients fitted
		"""
		coef_ = np.array( [self._loc0,self._loc1,self._scale0,self._scale1,self._shape] )
		return coef_
	##}}}
	
	def set_params( self , coef_ ): ##{{{
		self._loc0    = coef_[0]
		self._loc1    = coef_[1]
		self._scale0  = coef_[2]
		self._scale1  = coef_[3]
		self._shape   = coef_[4]
	#}}}
	
	
	###############
	## Accessors ##
	###############
	
	def meant( self , t ):##{{{
		cst = ( scs.gamma( 1 - self.shape[0] ) - 1 ) / self.shape[0] if np.abs( self.shape[0] ) > 1e-8 else np.euler_gamma
		return self._loct(t) + self._scalet(t) * cst
	##}}}
	
	def mediant( self , t ):##{{{
		return self._loct(t) + self._scalet(t) * ( np.pow( np.log(2) , - self._shapet(t) ) - 1. ) / self._shapet(t)
	##}}}
	
	def loct( self , t ): ##{{{
		"""
		Location of the Generalized Pareto Model at time t
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		loc : np.array
			loc at time t
		"""
		return self._loct(t)
	##}}}
	
	def scalet( self , t ):##{{{
		"""
		Scale of the Generalized Pareto Model at time t
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		scale : np.array
			scale at time t
		"""
		return self._scalet(t)
	##}}}
	
	def shapet( self , t ):##{{{
		"""
		Shape of the Generalized Pareto Model at time t
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		shape : np.array
			shape at time t
		"""
		return self._shapet(t)
	##}}}
	
	
	#############
	## Methods ##
	#############
	
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
		return self._tails * sc.genextreme.rvs( size = t.size , loc = self._loct(t) , scale = self._scalet(t) , c = - self._shapet(t) )
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
		return self._cdf( Y , t )  if self._tails > 0 else self._sf( Y , t )
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
		return self._icdf( q , t ) if self._tails > 0 else self._isf( q , t )
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
		return self._sf( Y , t )   if self._tails > 0 else self._cdf( Y , t )
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
		return self._isf( q , t )  if self._tails > 0 else self._icdf( q , t )
	##}}}
	
	
	############################################################
	## Masked methods                                         ##
	## Here the methods are available only if tail == "upper" ##
	## self.cdf (etc) switch between self._cdf and            ##
	## self._sf if tail == "lower"                            ##
	############################################################
	
	def _cdf( self , Y , t ):##{{{
		return sc.genextreme.cdf( self._tails * Y , loc = self._loct(t) , scale = self._scalet(t) , c = - self._shapet(t) )
	##}}}
	
	def _icdf( self , q , t ):##{{{
		return self._tails * sc.genextreme.ppf( q , loc = self._loct(t) , scale = self._scalet(t) , c = - self._shapet(t) )
	##}}}
	
	def _sf( self , Y , t ):##{{{
		return sc.genextreme.sf( self._tails * Y , loc = self._loct(t) , scale = self._scalet(t) , c = - self._shapet(t) )
	##}}}
	
	def _isf( self , q , t ):##{{{
		return self._tails * sc.genextreme.isf( q , loc = self._loct(t) , scale = self._scalet(t) , c = - self._shapet(t) )
	##}}}



