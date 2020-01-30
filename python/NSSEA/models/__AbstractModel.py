 
# -*- coding: utf-8 -*-


###############
## Libraries ##
###############

import numpy             as np
import scipy.stats       as sc
import scipy.interpolate as sci
import SDFC              as sd
import SDFC.tools        as sdt


#############
## Classes ##
#############

class Params:##{{{
	
	def __init__( self , name , is_cst , link , name_tex = None ):##{{{
		self.name     = name
		self.link     = link if link is not None else sdt.IdLink()
		self.is_cst   = is_cst
		self.n_params = 1 if self.is_cst else 2
		self.coef_    = None
		self._paramst = None
		self.name_tex = name_tex
		if name_tex is None:
			self.name_tex = "\mathrm{" + self.name + "}"
	##}}}
	
	def __call__( self , t ):##{{{
		return self._paramst(t)
	##}}}
	
	def set_covariable( self , X , t ):##{{{
		if self.is_cst:
			self._paramst = lambda x : self.link(self.coef_) + np.zeros_like(x)
		else:
			self._paramst = sci.interp1d( t , self.link(  self.coef_[0] + X.ravel() * self.coef_[1] ) )
	##}}}
##}}}


class AbstractModel:
	
	def __init__( self , name , law , sdlaw , lparams , **kwargs ):##{{{
		self.name    = name
		self.law     = law
		self.sdlaw   = sdlaw
		self.lparams = { p["name"] : Params(**p) for p in lparams }
		self.n_ns_params = 0
		for p in self.lparams:
			self.n_ns_params += 1 if self.lparams[p].is_cst else 2
	##}}}
	
	def __str__( self ):##{{{
		out = ""
		out += self.name + "\n"
		out += "params: "
		for pn in self.get_params_names(False):
			out += pn + ", "
		out += "\n"
		for p in self.lparams:
			out += str(self.lparams[p].link)
		return out
	##}}}
	
	def __repr__( self ):##{{{
		return self.__str__()
	##}}}
	
	def to_netcdf( self ):##{{{
		ncargs = { "ns_law_name" : self.name }
		for p in self.lparams:
			ncargs[ "ns_law_param_" + p + "_cst" ]  = str(self.lparams[p].is_cst)
			ncargs[ "ns_law_param_" + p + "_link" ] = self.lparams[p].link
		return ncargs
	##}}}
	
	
	## Fit methods
	##============
	
	def _get_sdkwargs( self , X ):##{{{
		sdkwargs = {}
		for p in self.lparams:
			sdkwargs[ "l_" + p ] = self.lparams[p].link
			if not self.lparams[p].is_cst:
				sdkwargs[ "c_" + p ] = X
		return sdkwargs
	##}}}
	
	def fit( self , Y , X ):##{{{
		sdkwargs = self._get_sdkwargs(X)
		sdlaw = self.sdlaw( method = "MLE" )
		sdlaw.fit( Y , **sdkwargs )
		for p in self.lparams:
			self.lparams[p].coef_ = sdlaw.params._dparams[p].coef_
	##}}}
	
	def drawn_bayesian( self , Y , X  , n_mcmc_drawn , prior ):##{{{
		sdkwargs = self._get_sdkwargs(X)
		sdlaw = self.sdlaw( method = "bayesian" )
		sdlaw.fit( -Y , n_mcmc_drawn = n_mcmc_drawn , prior = prior , **sdkwargs )
		return sdlaw._info.draw
	##}}}
	
	def check( self , Y , X , t = None ):##{{{
		return True
	##}}}
	
	
	## Accessors
	##==========
	
	def get_params_names( self , tex = False ):##{{{
		names = []
		for p in self.lparams:
			for i in range(self.lparams[p].n_params):
				if not tex:
					names.append( self.lparams[p].name + str(i) )
				else:
					names.append( r"$" + self.lparams[p].name_tex + "_" + str(i) + "$" )
		
		return names
	##}}}
	
	def get_params(self):##{{{
		coef_ = np.array([])
		for p in self.lparams:
			coef_ = np.hstack( (coef_,self.lparams[p].coef_) )
		return coef_
	##}}}
	
	def set_params( self , coef_ ):##{{{
		if not coef_.size == self.n_ns_params:
			return
		a,b = 0,0
		for p in self.lparams:
			b += self.lparams[p].n_params
			self.lparams[p].coef_ = coef_[a:b]
			a = b
	##}}}
	
	def set_covariable( self , X , t ):##{{{
		for p in self.lparams:
			self.lparams[p].set_covariable( X , t )
	##}}}
	
	
	## Stats methods
	##==============
	
	def _get_sckwargs( self , t ):##{{{
		sckwargs = {}
		for p in self.lparams:
			sckwargs[p] = self.lparams[p](t)
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
		return self.law.rvs( size = t.size , **sckwargs )
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
		return self.law.cdf( Y , **sckwargs )
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
		return self.law.ppf( q , **sckwargs )
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
		return self.law.sf( Y , **sckwargs )
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
		return self.law.isf( q , **sckwargs )
	##}}}








