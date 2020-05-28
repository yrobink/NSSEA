# -*- coding: utf-8 -*-

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2020                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
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
## Copyright Yoann Robin, 2020                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
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

import numpy             as np
import scipy.stats       as sc
import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.special     as scs
import SDFC              as sd
import SDFC.tools        as sdt


#############
## Classes ##
#############

class GEVPr:
	
	def __init__( self , mle_with_bayesian = False , **kwargs ):##{{{
		self._coefs = np.zeros(4)
		self._mle_with_bayesian = mle_with_bayesian
		self.n_ns_params = 4
		self.lparams = ["loc","scale","shape","alpha"]
	##}}}
	
	def to_netcdf( self ):##{{{
		ncargs = { "ns_law_name" : "GEVPr" }
		ncargs[ "ns_law_param_loc_cst" ]  = False
		ncargs[ "ns_law_param_loc_link" ] = sdt.IdLink()
		ncargs[ "ns_law_param_scale_cst" ]  = False
		ncargs[ "ns_law_param_scale_link" ] = sdt.IdLink()
		ncargs[ "ns_law_param_shape_cst" ]  = False
		ncargs[ "ns_law_param_shape_link" ] = sdt.IdLink()
		ncargs[ "ns_law_param_alpha_cst" ]  = False
		ncargs[ "ns_law_param_alpha_link" ] = sdt.IdLink()
		return ncargs
	##}}}
	
	
	## Methods
	##{{{
	
	def meant( self , t ):##{{{
		shapet = self.shapet(t)
		idx = np.abs(shapet) > 1e-8
		cst = np.zeros(shapet) + np.euler_gamma
		cst[idx] = ( scs.gamma( 1 - shapet[idx] ) - 1 ) / shapet[idx]
		return self._loct(t) + self._scalet(t) * cst
	##}}}
	
	def mediant( self , t ):##{{{
		return self.loct(t) + self.scalet(t) * ( np.pow( np.log(2) , - self.shapet(t) ) - 1. ) / self.shapet(t)
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
		loc   = self.loct(t)
		scale = self.scalet(t)
		shape = self.shapet(t)
		bound = loc - scale / shape
		idx   = np.logical_not( shape < 0 )
		bound[idx] = np.inf
		return bound
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
		loc   = self.loct(t)
		scale = self.scalet(t)
		shape = self.shapet(t)
		bound = loc - scale / shape
		idx   = shape < 0
		bound[idx] = - np.inf
		return bound
	##}}}
	
	##}}}
	##========
	
	## Accessors
	##{{{
	
	def get_params( self ):
		return self._coefs
	
	def get_params_names( self , tex = False ): ##{{{
		if not tex:
			return ["loc","scale","shape","alpha"]
		else:
			return [r"$\mu$",r"$\sigma$",r"$\xi$",r"$\alpha$"]
	##}}}
	
	def set_params( self , coefs ):##{{{
		self._coefs = coefs
	##}}}
	
	def set_covariable( self , X , t ):##{{{
		ratio = self._coefs[3] / self._coefs[0]
		loc   = np.squeeze( self._coefs[0] * np.exp( ratio * X ) )
		scale = np.squeeze( self._coefs[1] * np.exp( ratio * X ) )
		shape = np.zeros_like(X.squeeze()) + self._coefs[2]
		self._loct   = sci.interp1d( t , loc   )
		self._scalet = sci.interp1d( t , scale )
		self._shapet = sci.interp1d( t , shape )
	##}}}
	
	def loct( self , t ):##{{{
		return self._loct(t)
	##}}}
	
	def scalet( self , t ):##{{{
		return self._scalet(t)
	##}}}
	
	def shapet( self , t ):##{{{
		return self._shapet(t)
	##}}}
	
	##}}}
	##==========
	
	## Fit methods
	##{{{
	
	def _init_mle( self ):##{{{
		if self._X.ndim == 1: self._X = self._X.reshape(-1,1)
		
		gev = sd.GEV( method = "lmoments-experimental" )
		gev.fit( self._Y , c_loc = self._X , c_scale = self._X )
		
		ratio = np.mean( gev.scale / gev.loc )
		
		tmu    = np.mean( gev.loc   * np.exp( - ratio * self._X ) )
		tscale = np.mean( gev.scale * np.exp( - ratio * self._X ) )
		alpha  = 0.1
		
		mcoefs = np.array( [tmu,tscale,float(gev.shape[0]),alpha] )
		coefs  = mcoefs.copy()
		factor = 1
		nit    = 1
		while not np.isfinite(self._negloglikelihood(coefs)) or not np.isfinite(self._gradient_nlll(coefs)).all():
			coefs = np.random.multivariate_normal( mean = mcoefs , cov = factor * np.identity(4) )
			if nit % 100 == 0: factor *= 2
			if nit == 1000: break
		
		return coefs
	##}}}
	
	def _negloglikelihood( self , coef ):##{{{
		
		## Build loc, scale and shape
		##===========================
		tloc,tscale,tshape,alpha = coef
		
		E     = np.exp( alpha / tloc * self._X )
		loc   = tloc * E
		scale = tscale * E
		shape = tshape + np.zeros_like(self._X)
		
		## Usefull values
		##===============
		Z     = ( self._Y - loc ) / scale
		Za1   = 1 + shape * Z
		Zamsi = np.power( Za1 , - 1. / shape )
		
		## Likelihood
		##===========
		res = np.sum( ( 1 + 1. / shape ) * np.log(Za1) + Zamsi + np.log(scale) )
		
		if not np.isfinite(res): res = np.inf
		
		return res
	##}}}
	
	def _gradient_nlll( self , coef ):##{{{
		
		## Build loc, scale and shape
		##===========================
		tloc,tscale,tshape,alpha = coef
		
		E      = np.exp(   alpha / tloc * self._X )
		G      = np.exp( - alpha / tloc * self._X )
		loc    = tloc * E
		scale  = tscale * E
		shape  = tshape + np.zeros_like(self._X)
		ishape = 1. / shape
		
		## Usefull values
		##===============
		Z     = ( self._Y - loc ) / scale
		Za1   = 1 + shape * Z
		Zamsi = np.power( Za1 , - ishape )
		C1    = 1 + ( 1 - Zamsi ) * ishape / Za1
		
		A0 = ( 1. + ishape - ishape * Zamsi ) / Za1
		
		
		## Gradient of loc
		##================
		grmu_sig = - scale * alpha * self._X / tloc**2
		grmu_Za1 = - shape * ( self._Y * grmu_sig / scale**2 - 1. / tscale )
		grloc    = np.sum( A0 * grmu_Za1 + grmu_sig / scale )
		
		## Gradient of scale
		##==================
		grsig_sig = E
		grsig_Za1 = - shape * Z / tscale
		grscale   = np.sum( A0 * grsig_Za1 + grsig_sig / scale )
		
		## Gradient of shape
		##==================
		grshape   = np.sum( ( Zamsi - 1 ) * np.log(Za1) / shape**2 + C1 * Z )
		
		## Gradient of alpha
		##==================
		gralp_Za1 = - shape * self._Y * self._X / scale / tloc
		gralp_sig = scale * self._X / tloc
		gralpha   = np.sum( A0 * gralp_Za1 + gralp_sig / scale )
		
		return np.array( [grloc,grscale,grshape,gralpha] )
	##}}}
	
	def fit( self , Y , X ):##{{{
		self._X = X.reshape(-1,1)
		self._Y = Y.reshape(-1,1)
		
		self._coefs = self._init_mle()
		self.optim_result = sco.minimize( self._negloglikelihood , self._coefs , method = "BFGS" )#, jac = self._gradient_nlll )
		self._coefs = self.optim_result.x
		
		del self._X,self._Y
		
		if self._mle_with_bayesian:
			prior = sc.multivariate_normal( mean = self._coefs , cov = self.optim_result.hess_inv )
			draw = self.drawn_bayesian( Y , X , 10000 , prior , 0 )
			self._coefs = draw[np.random.choice(range(5000,10000),1),:].squeeze()
		
	##}}}
	
	def _fit_bayesian( self , n_mcmc_drawn , prior ):##{{{
		## Parameters
		##===========
		n_features = 4
		transition = lambda x : x + np.random.normal( size = n_features , scale = 0.1 )
		
		## MCMC algorithm
		##===============
		draw = np.zeros( (n_mcmc_drawn,n_features) )
		accept = np.zeros( n_mcmc_drawn , dtype = np.bool )
		
		## Init values
		##============
		init = prior.rvs()
		
		draw[0,:]     = init
		lll_current   = -self._negloglikelihood(draw[0,:])
		prior_current = prior.logpdf(draw[0,:]).sum()
		p_current     = prior_current + lll_current
		
		for i in range(1,n_mcmc_drawn):
			draw[i,:] = transition(draw[i-1,:])
			
			## Likelihood and probability of new points
			lll_next   = - self._negloglikelihood(draw[i,:])
			prior_next = prior.logpdf(draw[i,:]).sum()
			p_next     = prior_next + lll_next
			
			## Accept or not ?
			p_accept = np.exp( p_next - p_current )
			if np.random.uniform() < p_accept:
				lll_current   = lll_next
				prior_current = prior_next
				p_current     = p_next
				accept[i] = True
			else:
				draw[i,:] = draw[i-1,:]
				accept[i] = False
		
		rate_accept = np.sum(accept) / n_mcmc_drawn
		
		return draw,rate_accept
	##}}}
	
	def drawn_bayesian( self , Y , X  , n_mcmc_drawn , prior , min_rate_accept = 0.25 ):##{{{
		self._X = X.reshape(-1,1)
		self._Y = Y.reshape(-1,1)
		
		test_rate = False
		while not test_rate:
			draw,rate_accept = self._fit_bayesian( n_mcmc_drawn , prior )
			test_rate = rate_accept > min_rate_accept
		
		del self._X,self._Y
		return draw
	##}}}
	
	##}}}
	##============
	
	## Statistical methods
	##{{{
	
	def check(self,*args,**kwargs):
		return True
	
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
		kwargs = { "loc" : self.loct(t) , "scale" : self.scalet(t) , "c" : - self.shapet(t) }
		return sc.genextreme.rvs( size = t.size , **kwargs )
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
		kwargs = { "loc" : self.loct(t) , "scale" : self.scalet(t) , "c" : - self.shapet(t) }
		return sc.genextreme.cdf( Y , **kwargs )
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
		kwargs = { "loc" : self.loct(t) , "scale" : self.scalet(t) , "c" : - self.shapet(t) }
		return sc.genextreme.ppf( q , **kwargs )
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
		kwargs = { "loc" : self.loct(t) , "scale" : self.scalet(t) , "c" : - self.shapet(t) }
		return sc.genextreme.sf( Y , **kwargs )
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
		kwargs = { "loc" : self.loct(t) , "scale" : self.scalet(t) , "c" : - self.shapet(t) }
		return sc.genextreme.isf( q , **kwargs )
	##}}}
	
	##}}}
	##====================
	



