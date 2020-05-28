# -*- coding: utf-8 -*-

###############
## Libraries ##
###############

import numpy             as np
import scipy.stats       as sc
import scipy.interpolate as sci
import scipy.special     as scs
import SDFC              as sd
import SDFC.tools        as sdt

from .__AbstractModel import AbstractModel


#############
## Classes ##
#############

class GEVMin(AbstractModel):
	
	def __init__( self , loc_cst = False , scale_cst = False , shape_cst = True , **kwargs ):##{{{
		l_scale = kwargs.get("l_scale")
		if l_scale is None: l_scale = sdt.ExpLink()
		lparams = []
		lparams.append( { "name" : "loc"   , "is_cst" :   loc_cst , "link" : kwargs.get("l_loc")   , "name_tex" : r"\mu"    } )
		lparams.append( { "name" : "scale" , "is_cst" : scale_cst , "link" : l_scale               , "name_tex" : r"\sigma" } )
		lparams.append( { "name" : "shape" , "is_cst" : shape_cst , "link" : kwargs.get("l_shape") , "name_tex" : r"\xi"    } )
		AbstractModel.__init__( self , "GEV" , sc.genextreme , sd.GEV , lparams , **kwargs )
	##}}}
	
	
	## Fit methods
	##============
	
	def _get_sdkwargs( self , X ):##{{{
		sdkwargs = {}
		for p in self.lparams:
			sdkwargs[ "l_" + p ] = self.lparams[p].link
			if not self.lparams[p].is_cst:
				sdkwargs[ "c_" + p ] = -X
		return sdkwargs
	##}}}
	
	## Accessors
	##==========
	
	def set_covariable( self , X , t ):##{{{
		AbstractModel.set_covariable( self , -X , t )
	##}}}
	
	def fit( self , Y , X ):##{{{
		AbstractModel.fit( self , -Y , X )
	##}}}
	
	def drawn_bayesian( self , Y , X  , n_mcmc_drawn , prior , min_rate_accept = 0.25 ):##{{{
		return AbstractModel.drawn_bayesian( self , -Y , X , n_mcmc_drawn , prior , min_rate_accept )
	##}}}
	
	## Stats methods
	##==============
	
	def loct( self , t ):##{{{
		return -self.lparams["loc"](t)
	##}}}
	
	def scalet( self , t ):##{{{
		return self.lparams["scale"](t)
	##}}}
	
	def shapet( self , t ):##{{{
		return self.lparams["shape"](t)
	##}}}
	
	def meant( self , t ):##{{{
		shapet = self.shapet(t)
		idx = np.abs(shapet) > 1e-8
		cst = np.zeros(shapet) + np.euler_gamma
		cst[idx] = ( scs.gamma( 1 - shapet[idx] ) - 1 ) / shapet[idx]
		return - (self._loct(t) + self._scalet(t) * cst)
	##}}}
	
	def mediant( self , t ):##{{{
		return - (self.loct(t) + self.scalet(t) * ( np.pow( np.log(2) , - self.shapet(t) ) - 1. ) / self.shapet(t))
	##}}}
	
	def upper_boundt( self , t ):##{{{
		"""
		Upper bound of GEV model (can be infinite)
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		bound : np.array
			bound at time t
		"""
		loc   = -self.loct(t)
		scale =  self.scalet(t)
		shape =  self.shapet(t)
		bound = loc - scale / shape
		idx   = np.logical_not( shape < 0 )
		bound[idx] = np.inf
		return -bound
	##}}}
	
	def lower_boundt( self , t ):##{{{
		"""
		Lower bound of GEV model (can be -infinite)
		
		Parameters
		----------
		t : np.array
			Time
		
		Results
		-------
		bound : np.array
			bound at time t
		"""
		loc   = -self.loct(t)
		scale =  self.scalet(t)
		shape =  self.shapet(t)
		bound = loc - scale / shape
		idx   = shape < 0
		bound[idx] = - np.inf
		return -bound
	##}}}
	
	
	def _get_sckwargs( self , t ):##{{{
		sckwargs = AbstractModel._get_sckwargs( self , t )
		sckwargs["c"] = - sckwargs["shape"]
		del sckwargs["shape"]
		return sckwargs
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
		sckwargs = self._get_sckwargs(t)
		return - self.law.rvs( size = t.size , **sckwargs )
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
		sckwargs = self._get_sckwargs(t)
		return self.law.sf( -Y , **sckwargs )
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
		sckwargs = self._get_sckwargs(t)
		return -self.law.isf( q , **sckwargs )
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
		sckwargs = self._get_sckwargs(t)
		return self.law.cdf( -Y , **sckwargs )
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
		sckwargs = self._get_sckwargs(t)
		return -self.law.ppf( q , **sckwargs )
	##}}}


