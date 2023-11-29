
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

import numpy as np
import scipy.linalg as scl
import scipy.optimize as sco
import scipy.stats as sc
import pandas as pd
import xarray as xr

from.__multi_model import MultiModel
from .__tools import matrix_squareroot
from .__tools import ProgressBar

from .models.__Normal import Normal
from .models.__GEV    import GEV

from SDFC.link import ULIdentity
from SDFC.link import ULExponential

import SDFC as sd


#############
## Classes ##
#############

###############
## Functions ##
###############

## Covariate constraint
##=====================

def constrain_covariate( climIn , Xo , time_reference = None , assume_good_scale = False , verbose = False ): ##{{{
	"""
	NSSEA.constrain_covariate
	=========================
	Constrain covariates of clim by the observed covariates Xo
	
	Arguments
	---------
	climIn : NSSEA.Climatology
		clim variable
	Xo       : pandas.DataFrame
		Covariate observed
	time_reference: array
		Reference time period of Xo
	assume_good_scale : boolean
		If we assume than observations and multi-model have the same scale
	verbose  : bool
		Print (or not) state of execution
	
	Return
	------
	clim : NSSEA.Climatology, a copy is returned
	"""
	
	pb = ProgressBar( 5 , "constrain_covariate" , verbose )
	
	clim = climIn.copy()
	
	## Parameters
	##===========
	time        = clim.time
	time_Xo     = Xo.index
	n_time      = clim.n_time
	n_time_Xo   = time_Xo.size
	n_mm_coef   = clim.data["mm_mean"].size
	n_coef      = clim.n_coef
	n_sample    = clim.n_sample
	samples     = clim.X.sample
	pb.print()
	
	## Projection matrix H
	##====================
	cx = xr.DataArray( np.zeros( (n_time,n_time) )       , coords = [time,time]       , dims = ["time1","time2"] )
	cy = xr.DataArray( np.zeros( (n_time_Xo,n_time_Xo) ) , coords = [time_Xo,time_Xo] , dims = ["time1","time2"] )
	if not assume_good_scale:
		cx.loc[:,time_reference] = 1. / time_reference.size
	if not assume_good_scale:
		cy.loc[:,time_reference] = 1. / time_reference.size
	
	centerX  = np.identity(n_time)    - cx.values
	centerY  = np.identity(n_time_Xo) - cy.values
	extractX = np.hstack( ( np.identity(n_time) , np.zeros( (n_time,n_time) ) , np.zeros( (n_time,n_coef) ) ) )
	H_full   = pd.DataFrame( centerX @ extractX , index = time )	# Centering * extracting
	H        = H_full.loc[time_Xo,:].values								# Restriction to the observed period
	pb.print()
	
	
	# Other inputs : x, SX, y, SY
	##===========================
	X  = clim.data["mm_mean"].values
	SX = clim.data["mm_cov"].values
	Y  = np.ravel(centerY @ Xo)
	SY = centerY @ centerY.T
	pb.print()
	
	## Rescale SY
	##===========
	if not assume_good_scale:
		res = Y - H @ X
		K   = H @ SX @ H.T
		
		def fct_to_root(lbda):
			SY_tmp = lbda * SY
			iSig = np.linalg.pinv( K + SY_tmp )
			out = np.sum( np.diag( iSig @ SY_tmp ) ) - res.T @ iSig @ SY_tmp @ iSig @ res
			return out
		
		a,b = 1e-2,1e2
		while fct_to_root(a) * fct_to_root(b) > 0:
			a /= 2
			b *= 2
		
		lbda = sco.brentq( fct_to_root , a = a , b = b )
		SY = lbda * SY
	pb.print()
	
	## Apply constraints
	##==================
	
	Sinv = np.linalg.pinv( H @ SX @ H.T + SY )
	K	 = SX @ H.T @ Sinv
	clim.data["mm_mean"].values = X + K @ ( Y - H @ X )
	clim.data["mm_cov"].values  = SX - SX @ H.T @ Sinv @ H @ SX
	pb.print()
	
	
	## Sample from it
	##===============
	law       = MultiModel()
	law.mean  = clim.data["mm_mean"].values
	law.cov   = clim.data["mm_cov"].values
	cx_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,2) ) , coords = [ clim.X.time , samples , clim.X.forcing ] , dims = ["time","sample","forcing"] )
	
	cx_sample.loc[:,"BE","F"] = law.mean[:n_time]
	cx_sample.loc[:,"BE","C"] = law.mean[n_time:(2*n_time)]
	
	for s in samples[1:]:
		draw = law.rvs()
		cx_sample.loc[:,s,"F"] = draw[:n_time]
		cx_sample.loc[:,s,"C"] = draw[n_time:(2*n_time)]
	
	for m in clim.model:
		clim.X.loc[:,:,:,m] = cx_sample.values
	
	pb.end()
	
	return clim
##}}}


## Bayesian constraint
##====================

def _constrain_law_all( climIn , Yo , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose , **kwargs ):##{{{
	clim = climIn.copy()
	
	pb = ProgressBar( clim.n_sample + 1 , "constrain_law" , verbose )
	
	
	min_rate_accept = kwargs.get("min_rate_accept")
	if min_rate_accept is None:
		min_rate_accept = 0.05
	else:
		kwargs.pop("min_rate_accept")
	
	## Define prior
	prior_mean   = clim.data["mm_mean"][-clim.n_coef:].values
	prior_cov    = clim.data["mm_cov"][-clim.n_coef:,-clim.n_coef:].values
	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
	
	## And now MCMC loop
	for s in clim.sample:
		pb.print()
		X   = clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
		n_mcmc_drawn = np.random.randint( n_mcmc_drawn_min , n_mcmc_drawn_max )
		draw = clim.ns_law.drawn_bayesian( Yo.values.squeeze() , X , n_mcmc_drawn , prior_law , min_rate_accept , **kwargs )
		clim.law_coef.loc[:,s,"Multi_Synthesis"] = draw[-1,:]
	
	clim.law_coef.loc[:,"BE",:] = clim.law_coef[:,1:,:].median( dim = "sample" )
	clim.BE_is_median = True
	
	pb.end()
	
	return clim
##}}}

def _constrain_law_keep( climIn , Yo , keep , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose , **kwargs ):##{{{
	
	clim = climIn.copy()
	
	## Cut indexes
	n_sample = clim.n_sample + 1
	n_keep   = int(keep * n_sample)
	
	index_keep = np.random.choice( n_sample , n_keep , replace = False )
	index_supp = np.array( [i for i in range(n_sample) if i not in index_keep] , dtype = np.int )
	assoc      = np.random.choice( index_keep , n_sample - n_keep )
	
	l_index = []
	for i in index_keep:
		l_index.append([i])
		l_index[-1] = l_index[-1] + index_supp[assoc==i].tolist()
	
	pb = ProgressBar( n_keep , "constrain_law" , verbose )
	
	
	min_rate_accept = kwargs.get("min_rate_accept")
	if min_rate_accept is None: min_rate_accept = 0.05
	
	## Define prior
	prior_mean   = clim.data["mm_mean"][-clim.n_coef:].values
	prior_cov    = clim.data["mm_cov"][-clim.n_coef:,-clim.n_coef:].values
	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
	
	## And now MCMC loop
	for index in l_index:
		pb.print()
		X   = clim.X.loc[Yo.index,clim.sample[index[0]],"F","Multi_Synthesis"].values.squeeze()
		n_mcmc_drawn = np.random.randint( n_mcmc_drawn_min , n_mcmc_drawn_max )
		draw = clim.ns_law.drawn_bayesian( Yo.values.squeeze() , X , n_mcmc_drawn , prior_law , min_rate_accept  , **kwargs )
		draw = draw[n_mcmc_drawn_min:,:]
		
		clim.law_coef.loc[:,clim.sample[index],"Multi_Synthesis"] = draw[np.random.choice(draw.shape[0],len(index),False),:].T
		newX = np.zeros( (clim.time.size,len(index),2) )
		for i in range(len(index)):
			newX[:,i,:] = clim.X.loc[:,clim.sample[index[0]],:,"Multi_Synthesis"].values.squeeze()
		clim.X.loc[:,clim.sample[index],:,"Multi_Synthesis"] = newX
	
	clim.law_coef.loc[:,"BE",:] = clim.law_coef[:,1:,:].median( dim = "sample" )
	clim.BE_is_median = True
	
	pb.end()
	
	return clim
	
	
	pass
##}}}

def _constrain_law_ess( climIn , Yo , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose , **kwargs ):##{{{
	#Warning : Sample are saved given another coordinate : "sample_MCMC" with X sample attached
	clim = climIn.copy()
	
	pb = ProgressBar( clim.n_sample + 1 , "constrain_law" , verbose )
	
	n_ess = kwargs.get("n_ess")
	if n_ess is None:
		n_ess = 10	
	min_rate_accept = 0
	ns_params_names = clim.ns_law.get_params_names()
	n_features = clim.ns_law.n_ns_params
	## Define prior
	prior_mean   = clim.data["mm_mean"][-clim.n_coef:].values
	prior_cov    = clim.data["mm_cov"][-clim.n_coef:,-clim.n_coef:].values
	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
	
	#OUtput
	results=np.array([])
	sample_names =[s+"_"+str(i) for i in range(n_ess) for s in clim.sample[1:]]+["BE"]

	law_coef_bay   = xr.DataArray( np.zeros( (n_features,(clim.n_sample)*n_ess + 1,1) ) , coords = [ ns_params_names , sample_names , ["Multi_Synthesis"] ] , dims = ["coef","sample_MCMC","model"] )
	## And now MCMC loop
	for s in clim.sample[1:]:
		pb.print()
		X   = clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
		n_mcmc_drawn = np.random.randint( n_mcmc_drawn_min , n_mcmc_drawn_max )
		draw = clim.ns_law.drawn_bayesian( Yo.values.squeeze() , X , n_mcmc_drawn , prior_law , min_rate_accept , **kwargs )
		n_tirage=(len(draw)//n_ess)
		select=draw[0::n_tirage][:n_ess]
		law_coef_bay.loc[:,[s+"_"+str(i) for i in range(n_ess)],"Multi_Synthesis"]=select.T
	
	clim.law_coef=law_coef_bay
	clim.law_coef.loc[:,"BE",:] = clim.law_coef[:,1:,:].median( dim = "sample_MCMC" )
	clim.BE_is_median = True
	
	pb.end()
	
	return clim

def constrain_law( climIn , Yo , keep = "all" , n_mcmc_drawn_min = 5000 , n_mcmc_drawn_max = 10000 , verbose = False , **kwargs ):##{{{
	"""
	NSSEA.constrain_law
	===================
	Constrain the law_coef of the clim with a MCMC approach.
	
	Arguments
	---------
	climIn : [NSSEA.Climatology] clim variable
	Yo       : [pandas.DataFrame] Observations of ns_law
	keep     : [ "all", "ess", or a float between 0 and 1] If keep < 1, only a ratio of 
	          keep covariates is used, and many coefficients are drawn for the
	          same covariate. Faster, but can reduce confidence interval
	          uncertainty.
	n_mcmc_draw_min: [integer] Minimum number of coef to draw for each covariate
	n_mcmc_draw_max: [integer] Maximum number of coef to draw for each covariate
	verbose  : [bool] Print (or not) state of execution
	
	Return
	------
	clim : [NSSEA.Climatology] A copy is returned
	"""
	if keep == "ess":
		return _constrain_law_ess( climIn , Yo , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose , **kwargs )	
	elif keep == "all" or not keep < 1:
		#Maybe avoid a possible comparison between str and int ? put percentage as a kwarg ?
		return _constrain_law_all( climIn , Yo , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose , **kwargs )
	else:
		return _constrain_law_keep( climIn , Yo , keep , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose , **kwargs )
	
#	clim = climIn.copy()
#	
#	pb = ProgressBar( clim.n_sample + 1 , "constrain_law" , verbose )
#	
#	
#	min_rate_accept = kwargs.get("min_rate_accept")
#	if min_rate_accept is None: min_rate_accept = 0.25
#	
#	## Define prior
#	prior_mean   = clim.data["mm_mean"][-clim.n_coef:].values
#	prior_cov    = clim.data["mm_cov"][-clim.n_coef:,-clim.n_coef:].values
#	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
#	
#	## And now MCMC loop
#	for s in clim.sample:
#		pb.print()
#		X   = clim.X.loc[Yo.index,s,"F","Multi_Synthesis"].values.squeeze()
#		n_mcmc_drawn = np.random.randint( n_mcmc_drawn_min , n_mcmc_drawn_max )
#		draw = clim.ns_law.drawn_bayesian( Yo.values.squeeze() , X , n_mcmc_drawn , prior_law , min_rate_accept  , **kwargs )
#		clim.law_coef.loc[:,s,"Multi_Synthesis"] = draw[-1,:]
#	
#	clim.law_coef.loc[:,"BE",:] = clim.law_coef[:,1:,:].median( dim = "sample" )
#	clim.BE_is_median = True
#	
#	pb.end()
#	
#	return clim
##}}}


## C0 constraints
##===============

def constraint_C0_Normal( climIn , Yo , verbose = False ): ##{{{
	pb = ProgressBar( 4 , "constraint_C0" , verbose )
	
	clim      = climIn.copy()
	n_model   = clim.n_model
	n_sample  = clim.n_sample
	models    = clim.model
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	samples   = clim.sample
	
	# New NS_param
	law_coef = clim.law_coef
	law_coef.loc["scale1",:,:] = law_coef.loc["scale1",:,:] / law_coef.loc["scale0",:,:]
	pb.print()
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,samples] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_model) ) , coords = [time_Yo,samples,models] , dims = ["time","sample","model"] )
	Yo_bs.loc[:,"BE"] = np.ravel(Yo)
	Xo_bs.loc[:,"BE",:] = clim.X.loc[time_Yo,"BE","F",:]
	for s in samples[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"F",m].values
	pb.print()
	
	
	## Correction of loc0
	mu1X = xr.zeros_like(Xo_bs)
	mu1X = Xo_bs * law_coef.loc["loc1",:,:]
	Yo_bs_mean_corrected  = Yo_bs - mu1X
	law_coef.loc["loc0",:,:] = Yo_bs_mean_corrected.mean( axis = 0 )
	pb.print()
	
	
	## Correction of loc and scale
	Yo_bs_mu0 = Yo_bs - law_coef.loc["loc0",:,:]
	sig1X     = Xo_bs * law_coef.loc["scale1",:,:]
	Yo_bs_full_corrected   = (Yo_bs_mu0 - mu1X) / ( 1. + sig1X )
	law_coef.loc["scale0",:,:] = Yo_bs_full_corrected.std( axis = 0 )
	pb.print()
	
	## Save
	law_coef.loc["scale1",:,:] = law_coef.loc["scale1",:,:] * climIn.law_coef.loc["scale0",:,:]
	clim.law_coef.values = law_coef.values
	
	pb.end()
	
	return clim
##}}}

def constraint_C0_Normal_exp( climIn , Yo , verbose = False ): ##{{{
	pb = ProgressBar( 4 , "constraint_C0" , verbose )
	
	clim  = climIn.copy()
	n_model   = clim.n_model
	n_sample  = clim.n_sample
	models    = clim.model
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	samples   = clim.sample
	
	# New NS_param
	law_coef = clim.law_coef
	pb.print()
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,samples] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_model) ) , coords = [time_Yo,samples,models] , dims = ["time","sample","model"] )
	Yo_bs.loc[:,"BE"] = np.ravel(Yo)
	Xo_bs.loc[:,"BE",:] = clim.X.loc[time_Yo,"BE","F",:]
	for s in samples[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"F",m].values
	pb.print()
	
	
	## Correction of loc0
	mu1X = xr.zeros_like(Xo_bs)
	mu1X = Xo_bs * law_coef.loc["loc1",:,:]
	Yo_bs_mean_corrected  = Yo_bs - mu1X
	law_coef.loc["loc0",:,:] = Yo_bs_mean_corrected.mean( axis = 0 )
	pb.print()
	
	
	## Correction of loc and scale
	Yo_bs_mu0 = Yo_bs - law_coef.loc["loc0",:,:]
	sig1X     = np.exp( Xo_bs * law_coef.loc["scale1",:,:] )
	Yo_bs_full_corrected   = (Yo_bs_mu0 - mu1X) / sig1X
	law_coef.loc["scale0",:,:] = np.log( Yo_bs_full_corrected.std( axis = 0 ) )
	pb.print()
	
	## Save
	clim.data.law_coef.values = law_coef.values
	
	pb.end()
	
	return clim
##}}}


def constraint_C0_GEV( climIn , Yo , verbose = False ): ##{{{
	
	pb = ProgressBar( climIn.n_sample * climIn.n_model + 1 , "constraint_C0" , verbose )
	
	clim      = climIn.copy()
	n_model   = clim.n_model
	n_sample  = clim.n_sample
	models    = clim.X.model
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	samples   = clim.sample
	
	# New NS_param
	clim.law_coef.loc["scale1",:,:] = clim.law_coef.loc["scale1",:,:] / clim.law_coef.loc["scale0",:,:]
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,samples] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_model) ) , coords = [time_Yo,samples,models] , dims = ["time","sample","model"] )
	Yo_bs.loc[:,"BE"] = np.ravel(Yo)
	Xo_bs.loc[:,"BE",:] = clim.X.loc[time_Yo,"BE","F",:]
	for s in samples[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"F",m].values
	
	## Fit loc0
	clim.law_coef.loc["loc0",:,:] = ( Yo_bs - clim.law_coef.loc["loc1",:,:] * Xo_bs ).quantile( np.exp(-1) , dim = "time" )
	
	## Fit scale0 and shape
	Yo_GEV_stats = ( Yo_bs - clim.law_coef.loc["loc1",:,:] * Xo_bs - clim.law_coef.loc["loc0",:,:]) / ( 1 + clim.law_coef.loc["scale1",:,:] * Xo_bs ) ## Hypothesis : follow GEV(0,scale0,shape)
	for s in samples:
		for m in models:
			gev = sd.GEV()
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , f_loc = 0 , l_scale = climIn.ns_law.lparams["scale"].link , l_shape = climIn.ns_law.lparams["shape"].link )
			clim.law_coef.loc["scale0",s,m] = gev.coef_[0]
			clim.law_coef.loc["shape0",s,m] = gev.coef_[1]
			pb.print()
	
	
	## Save
	clim.law_coef.loc["scale1",:,:] = clim.law_coef.loc["scale1",:,:] * climIn.law_coef.loc["scale0",:,:]
	
	pb.end()
	
	return clim
##}}}

def constraint_C0_GEV_exp( climIn , Yo , verbose = False ): ##{{{
	
	pb = ProgressBar( climIn.n_sample * climIn.n_model + 1 , "constraint_C0" , verbose )
	
	clim      = climIn.copy()
	n_model   = clim.n_model
	n_sample  = clim.n_sample
	models    = clim.X.model
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	samples   = clim.sample
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,samples] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_model) ) , coords = [time_Yo,samples,models] , dims = ["time","sample","model"] )
	Yo_bs.loc[:,"BE"] = np.ravel(Yo)
	Xo_bs.loc[:,"BE",:] = clim.X.loc[time_Yo,"BE","F",:]
	for s in samples[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"F",m].values
	
	## Fit loc0
	clim.law_coef.loc["loc0",:,:] = ( Yo_bs - clim.law_coef.loc["loc1",:,:] * Xo_bs ).quantile( np.exp(-1) , dim = "time" )
	
	## Fit scale0 and shape
	Yo_GEV_stats = ( Yo_bs - clim.law_coef.loc["loc1",:,:] * Xo_bs - clim.law_coef.loc["loc0",:,:]) / np.exp( clim.law_coef.loc["scale1",:,:] * Xo_bs ) ## Hypothesis : follow GEV(0,scale0,shape)
	for s in samples:
		for m in models:
			gev = sd.GEV()
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , f_loc = 0 , l_scale = clim.ns_law.lparams["scale"].link , l_shape = clim.ns_law.lparams["shape"].link )
			clim.law_coef.loc["scale0",s,m] = gev.coef_[0]
			clim.law_coef.loc["shape0",s,m] = gev.coef_[1]
			pb.print()
	
	
	pb.end()
	
	return clim
##}}}


def constraint_C0( climIn , Yo , verbose = False ): ##{{{
	"""
	NSSEA.constraintsC0
	===================
	Constrain stationary parameters by observations
	
	Arguments
	---------
	climIn : NSSEA.Climatology
		clim variable
	Yo       : pandas.DataFrame
		Observations
	verbose  : bool
		Print (or not) state of execution
	
	Returns
	-------
	clim : NSSEA.Climatology
		A COPY of climIn constrained by Yo. climIn is NOT MODIFIED.
	"""
	
	if isinstance(climIn.ns_law,Normal):
		if isinstance(climIn.ns_law.lparams["scale"].link,ULIdentity):
			return constraint_C0_Normal( climIn , Yo , verbose )
		if isinstance(climIn.ns_law.lparams["scale"].link,ULExponential):
			return constraint_C0_Normal_exp( climIn , Yo , verbose )
	
	if isinstance(climIn.ns_law,GEV):
		if isinstance(climIn.ns_law.lparams["scale"].link,ULIdentity):
			return constraint_C0_GEV( climIn , Yo , verbose )
		if isinstance(climIn.ns_law.lparams["scale"].link,ULExponential):
			return constraint_C0_GEV_exp( climIn , Yo , verbose )
	
	return climIn.copy()
##}}}



