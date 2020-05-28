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
	
	def drawn_bayesian( self , Y , X  , n_mcmc_drawn , prior , min_rate_accept = 0.25 ):##{{{
		sdkwargs = self._get_sdkwargs(X)
		sdlaw = self.sdlaw( method = "bayesian" )
		test_rate = False
		while not test_rate:
			sdlaw.fit( Y , n_mcmc_drawn = n_mcmc_drawn , prior = prior , **sdkwargs )
			test_rate = sdlaw._info.rate_accept > min_rate_accept
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








