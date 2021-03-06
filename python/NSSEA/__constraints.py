
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

from .__tools import matrix_squareroot
from .__tools import ProgressBar

from .models.__Normal import Normal
from .models.__GEV    import GEV

from SDFC.tools import IdLink
from SDFC.tools import ExpLink

import SDFC as sd


#############
## Classes ##
#############

###############
## Functions ##
###############

## CX constraints
##===============


def constraints_CX( climIn , Xo , time_reference = None , assume_good_scale = False , verbose = False ): ##{{{
	"""
	NSSEA.constraintsCX
	===================
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
	
	Returns
	-------
	clim : NSSEA.Climatology
		A COPY of climIn constrained by Xo. climIn is NOT MODIFIED.
	"""
	
	if verbose: print( "Constraints CX" , end = "\r" )
	
	## Parameters
	##===========
	clim = climIn.copy()
	time        = clim.time
	time_Xo     = Xo.index
	n_time      = clim.n_time
	n_time_Xo   = time_Xo.size
	n_mm_params = clim.n_mm_params
	n_ns_params = clim.n_ns_params
	n_sample    = clim.n_sample
	sample      = clim.X.sample
	
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
	extractX = np.hstack( ( np.identity(n_time) , np.zeros( (n_time,n_time) ) , np.zeros( (n_time,n_ns_params) ) ) )
	H_full   = pd.DataFrame( centerX @ extractX , index = time )	# Centering * extracting
	H        = H_full.loc[time_Xo,:].values								# Restriction to the observed period
	
	
	# Other inputs : x, SX, y, SY
	##===========================
	X  = clim.mm_params.mean
	SX = clim.mm_params.cov
	Y  = np.ravel(centerY @ Xo)
	SY = centerY @ centerY.T
	
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
	
	## Apply constraints
	##==================
	
	Sinv = np.linalg.pinv( H @ SX @ H.T + SY )
	K	 = SX @ H.T @ Sinv
	clim.mm_params.mean = X + K @ ( Y - H @ X )
	clim.mm_params.cov  = SX - SX @ H.T @ Sinv @ H @ SX
	
	
	## Sample from it
	##===============
	cx_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,3) ) , coords = [ clim.X.time , sample , clim.X.forcing ] , dims = ["time","sample","forcing"] )
	
	cx_sample.loc[:,"be","all"] = clim.mm_params.mean[:n_time]
	cx_sample.loc[:,"be","nat"] = clim.mm_params.mean[n_time:(2*n_time)]
	
	for s in sample[1:]:
		draw = clim.mm_params.rvs()
		cx_sample.loc[:,s,"all"] = draw[:n_time]
		cx_sample.loc[:,s,"nat"] = draw[n_time:(2*n_time)]
	
	for m in clim.X.models:
		clim.X.loc[:,:,:,m] = cx_sample.values
	
	clim.X.loc[:,:,"ant",:] = clim.X.loc[:,:,"all",:] - clim.X.loc[:,:,"nat",:]
	
	if verbose: print( "Constraints CX (Done)" )
	
	return clim
##}}}


## C0 constraints
##===============

def constraints_C0_Normal( climIn , Yo , verbose = False ): ##{{{
	if verbose: print( "Constraints C0 (Normal)" , end = "\r" )
	
	clim  = climIn.copy()
	n_models  = clim.X.models.size
	n_sample  = clim.n_sample
	models    = clim.X.models
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = clim.X.sample
	
	# New NS_param
	ns_params = clim.ns_params
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] / ns_params.loc["scale0",:,:]
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,clim.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,clim.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = clim.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"all",m].values
	
	
	## Correction of loc0
	mu1X = xr.zeros_like(Xo_bs)
	mu1X = Xo_bs * ns_params.loc["loc1",:,:]
	Yo_bs_mean_corrected  = Yo_bs - mu1X
	ns_params.loc["loc0",:,:] = Yo_bs_mean_corrected.mean( axis = 0 )
	
	
	## Correction of loc and scale
	Yo_bs_mu0 = Yo_bs - ns_params.loc["loc0",:,:]
	sig1X     = Xo_bs * ns_params.loc["scale1",:,:]
	Yo_bs_full_corrected   = (Yo_bs_mu0 - mu1X) / ( 1. + sig1X )
	ns_params.loc["scale0",:,:] = Yo_bs_full_corrected.std( axis = 0 )
	
	## Save
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] * climIn.ns_params.loc["scale0",:,:]
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (Normal, Done)" )
	
	return clim
##}}}

def constraints_C0_Normal_exp( climIn , Yo , verbose = False ): ##{{{
	if verbose: print( "Constraints C0 (NormalExp)" , end = "\r" )
	
	clim  = climIn.copy()
	n_models  = clim.X.models.size
	n_sample  = clim.n_sample
	models    = clim.X.models
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = clim.X.sample
	
	# New NS_param
	ns_params = clim.ns_params
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,clim.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,clim.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = clim.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"all",m].values
	
	
	## Correction of loc0
	mu1X = xr.zeros_like(Xo_bs)
	mu1X = Xo_bs * ns_params.loc["loc1",:,:]
	Yo_bs_mean_corrected  = Yo_bs - mu1X
	ns_params.loc["loc0",:,:] = Yo_bs_mean_corrected.mean( axis = 0 )
	
	
	## Correction of loc and scale
	Yo_bs_mu0 = Yo_bs - ns_params.loc["loc0",:,:]
	sig1X     = np.exp( Xo_bs * ns_params.loc["scale1",:,:] )
	Yo_bs_full_corrected   = (Yo_bs_mu0 - mu1X) / sig1X
	ns_params.loc["scale0",:,:] = np.log( Yo_bs_full_corrected.std( axis = 0 ) )
	
	## Save
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (NormalExp, Done)" )
	
	return clim
##}}}


def constraints_C0_GEV( climIn , Yo , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEV)" , end = "\r" )
	
	clim  = climIn.copy()
	n_models  = clim.X.models.size
	n_sample  = clim.n_sample
	models    = clim.X.models
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample    = clim.X.sample
	
	# New NS_param
	ns_params = clim.ns_params
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] / ns_params.loc["scale0",:,:]
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,clim.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,clim.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = clim.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"all",m].values
	
	## Fit loc0
	ns_params.loc["loc0",:,:] = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs ).quantile( np.exp(-1) , dim = "time" )
	
	## Fit scale0 and shape
	Yo_GEV_stats = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs - ns_params.loc["loc0",:,:]) / ( 1 + ns_params.loc["scale1",:,:] * Xo_bs ) ## Hypothesis : follow GEV(0,scale0,shape)
	for s in sample:
		for m in models:
			gev = sd.GEV()
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , f_loc = 0 , l_scale = climIn.ns_law.lparams["scale"].link , l_shape = climIn.ns_law.lparams["shape"].link )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape0",s,m] = gev.coef_[1]
	
	
	## Save
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] * climIn.ns_params.loc["scale0",:,:]
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEV, Done)" )
	
	return clim
##}}}

def constraints_C0_GEV_exp( climIn , Yo , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEVExp)" , end = "\r" )
	
	clim  = climIn.copy()
	n_models  = clim.X.models.size
	n_sample  = clim.n_sample
	models    = clim.X.models
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = clim.X.sample
	
	# New NS_param
	ns_params = clim.ns_params
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,clim.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,clim.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = clim.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = clim.X.loc[idx,s,"all",m].values
	
	## Fit loc0
	ns_params.loc["loc0",:,:] = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs ).quantile( np.exp(-1) , dim = "time" )
	
	## Fit scale0 and shape
	Yo_GEV_stats = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs - ns_params.loc["loc0",:,:]) / np.exp( ns_params.loc["scale1",:,:] * Xo_bs ) ## Hypothesis : follow GEV(0,scale0,shape)
	for s in sample:
		for m in models:
			gev = sd.GEV()
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , f_loc = 0 , l_scale = climIn.ns_law.lparams["scale"].link , l_shape = climIn.ns_law.lparams["shape"].link )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape0",s,m] = gev.coef_[1]
	
	
	## Save
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEVExp, Done)" )
	
	return clim
##}}}

def constraints_C0_GEV_bound_valid( climIn , Yo , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEV)" , end = "\r" )
	
	clim  = climIn.copy()
	n_models  = clim.X.models.size
	n_sample  = clim.n_sample
	models    = clim.X.models
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = clim.X.sample
	
	# New NS_param
	ns_params = clim.ns_params
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] / ns_params.loc["scale0",:,:]
	
	
	gev = sd.GEVLaw(  method = clim.ns_law_args["method"] , link_fct_shape = clim.ns_law_args["link_fct_shape"] )
	for m in models:
		X   = clim.X.loc[time_Yo,"be","all",m].values.squeeze()
		Ybs = Yo.values.squeeze()
		ns_params.loc["loc0","be",m] = np.quantile( Ybs - float(ns_params.loc["loc1","be",m]) * X , np.exp(-1) )
		Yo_sta = ( Ybs - float(ns_params.loc["loc1","be",m]) * X - float(ns_params.loc["loc0","be",m]) ) / ( 1. + float(ns_params.loc["scale1","be",m]) * X )
		gev.fit( Yo_sta , floc = 0 )
		ns_params.loc["scale0","be",m] = gev.coef_[0]
		ns_params.loc["shape","be",m]  = gev.coef_[1]
		
		for s in sample[1:]:
			test = False
			while not test:
				idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
				X   = clim.X.loc[idx,s,"all",m].values.squeeze()
				Xcont = clim.X.loc[time_Yo,s,"all",m].values.squeeze()
				Ybs = Yo.loc[idx].values.squeeze()
				ns_params.loc["loc0",s,m] = np.quantile( Ybs - float(ns_params.loc["loc1",s,m]) * X , np.exp(-1) )
				Yo_sta  = ( Ybs - float(ns_params.loc["loc1",s,m]) * X     - float(ns_params.loc["loc0",s,m]) ) / ( 1. + float(ns_params.loc["scale1",s,m]) * X )
				Yo_cont = ( Yo.values.squeeze()  - float(ns_params.loc["loc1",s,m]) * Xcont - float(ns_params.loc["loc0",s,m]) ) / np.exp( float(ns_params.loc["scale1",s,m]) * Xcont )
				gev.fit( Yo_sta , floc = 0 )
				
				## Here I test if the bound from params from bootstrap is compatible with observed values
				law = clim.ns_law( **clim.ns_law_args )
				law.set_params( np.array( [ns_params.loc["loc0",s,m],ns_params.loc["loc1",s,m],gev.coef_[0],ns_params.loc["scale1",s,m],gev.coef_[1]] , dtype = np.float ) )
				law.set_covariable( clim.X.loc[time_Yo,s,"all",m].values , time_Yo )
#				print( np.all( Yo < law.upper_boundt(time_Yo) ) )
				test = np.all( np.logical_and( Yo.values.squeeze() > law.lower_boundt(time_Yo) , Yo.values.squeeze() < law.upper_boundt(time_Yo) ) )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	## Save
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] * climIn.ns_params.loc["scale0",:,:]
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEV, Done)" )
	
	return clim
##}}}

def constraints_C0_GEV_exp_bound_valid( climIn , Yo , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEVExp)" , end = "\r" )
	
	clim  = climIn.copy()
	n_models  = clim.X.models.size
	n_sample  = clim.n_sample
	models    = clim.X.models
	time      = clim.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = clim.X.sample
	
	# New NS_param
	ns_params = clim.ns_params
	
	
	gev = sd.GEVLaw(  method = clim.ns_law_args["method"] , link_fct_scale = ExpLink() , link_fct_shape = clim.ns_law_args["link_fct_shape"] )
	for m in models:
		X   = clim.X.loc[time_Yo,"be","all",m].values.squeeze()
		Ybs = Yo.values.squeeze()
		ns_params.loc["loc0","be",m] = np.quantile( Ybs - float(ns_params.loc["loc1","be",m]) * X , np.exp(-1) )
		Yo_sta = ( Ybs - float(ns_params.loc["loc1","be",m]) * X - float(ns_params.loc["loc0","be",m]) ) / np.exp( float(ns_params.loc["scale1","be",m]) * X )
		gev.fit( Yo_sta , floc = 0 )
		ns_params.loc["scale0","be",m] = gev.coef_[0]
		ns_params.loc["shape","be",m]  = gev.coef_[1]
		
		for s in sample[1:]:
			test = False
			while not test:
				idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
				X   = clim.X.loc[idx,s,"all",m].values.squeeze()
				Xcont = clim.X.loc[time_Yo,s,"all",m].values.squeeze()
				Ybs = Yo.loc[idx].values.squeeze()
				ns_params.loc["loc0",s,m] = np.quantile( Ybs - float(ns_params.loc["loc1",s,m]) * X , np.exp(-1) )
				Yo_sta  = ( Ybs - float(ns_params.loc["loc1",s,m]) * X     - float(ns_params.loc["loc0",s,m]) ) / np.exp( float(ns_params.loc["scale1",s,m]) * X )
				Yo_cont = ( Yo.values.squeeze()  - float(ns_params.loc["loc1",s,m]) * Xcont - float(ns_params.loc["loc0",s,m]) ) / np.exp( float(ns_params.loc["scale1",s,m]) * Xcont )
				gev.fit( Yo_sta , floc = 0 )
				
				## Here I test if the bound from params from bootstrap is compatible with observed values
				law = clim.ns_law( **clim.ns_law_args )
				law.set_params( np.array( [ns_params.loc["loc0",s,m],ns_params.loc["loc1",s,m],gev.coef_[0],ns_params.loc["scale1",s,m],gev.coef_[1]] , dtype = np.float ) )
				test = law.check( Yo.values.squeeze() , clim.X.loc[time_Yo,s,"all",m] , time_Yo )
#				law.set_covariable( clim.X.loc[time_Yo,s,"all",m].values , time_Yo )
#				print( np.all( Yo < law.upper_boundt(time_Yo) ) )
#				test = np.all( np.logical_and( Yo.values.squeeze() > law.lower_boundt(time_Yo) , Yo.values.squeeze() < law.upper_boundt(time_Yo) ) )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	## Save
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEVExp, Done)" )
	
	return clim
##}}}


def constraints_C0( climIn , Yo , gev_bound_valid = False , verbose = False ): ##{{{
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
	gev_bound_valid : bool
		For GEVLaw, use only bootstrap where observations are between the bound of the GEV.
	verbose  : bool
		Print (or not) state of execution
	
	Returns
	-------
	clim : NSSEA.Climatology
		A COPY of climIn constrained by Yo. climIn is NOT MODIFIED.
	"""
	
#	if climIn.ns_law == Normal:
#		if isinstance(climIn.ns_law_args["link_scale"],IdLink):
#			return constraints_C0_Normal( climIn , Yo , verbose )
#		elif isinstance(climIn.ns_law_args["link_scale"],ExpLink):
#			return constraints_C0_Normal_exp( climIn , Yo , verbose )
#	if climIn.ns_law == GEV:
#		if isinstance(climIn.ns_law_args["link_scale"],IdLink):
#			if gev_bound_valid:
#				return constraints_C0_GEV_bound_valid( climIn , Yo , verbose )
#			else:
#				return constraints_C0_GEV( climIn , Yo , verbose )
#		elif isinstance(climIn.ns_law_args["link_scale"],ExpLink):
#			if gev_bound_valid:
#				return constraints_C0_GEV_exp_bound_valid( climIn , Yo , verbose )
#			else:
#				return constraints_C0_GEV_exp( climIn , Yo , verbose )
	
	if isinstance(climIn.ns_law,Normal):
		if isinstance(climIn.ns_law.lparams["scale"].link,IdLink):
			return constraints_C0_Normal( climIn , Yo , verbose )
		if isinstance(climIn.ns_law.lparams["scale"].link,ExpLink):
			return constraints_C0_Normal_exp( climIn , Yo , verbose )
	
	if isinstance(climIn.ns_law,GEV):
		if isinstance(climIn.ns_law.lparams["scale"].link,IdLink):
			return constraints_C0_GEV( climIn , Yo , verbose )
		if isinstance(climIn.ns_law.lparams["scale"].link,ExpLink):
			return constraints_C0_GEV_exp( climIn , Yo , verbose )
	
	return climIn.copy()
##}}}


## Bayesian constraint
##====================

def constraints_bayesian( clim , Yo , n_mcmc_drawn_min = 5000 , n_mcmc_drawn_max = 10000 , min_rate_accept = 0.25 , verbose = False ):##{{{
	
	pb = ProgressBar( "Constraints Bayesian" , clim.n_sample + 1 )
	
	climCB = clim.copy()
	
	## Define prior
#	n_params  = clim.ns_law.n_ns_params
#	prior_law = sc.multivariate_normal( mean = climCB.mm_params.mean[-n_params:] , cov = climCB.mm_params.cov[-n_params:,-n_params:] , allow_singular = True )
	prior_sample = climCB.ns_params.loc[:,:,"multi"].values.T
	prior_mean   = np.mean(prior_sample,axis=0).squeeze()
	prior_cov    = np.cov( prior_sample.T )
	prior_law    = sc.multivariate_normal( mean = prior_mean , cov = prior_cov , allow_singular = True )
	
	
	for s in clim.X.sample:
		if verbose: pb.print()
		X   = clim.X.loc[Yo.index,s,"all","multi"].values.squeeze()
		n_mcmc_drawn = np.random.randint( n_mcmc_drawn_min , n_mcmc_drawn_max )
		draw = clim.ns_law.drawn_bayesian( Yo.values.squeeze() , X , n_mcmc_drawn , prior_law , min_rate_accept )
		climCB.ns_params.loc[:,s,"multi"] = draw[-1,:]
	
	climCB.ns_params.loc[:,"be","multi"] = climCB.ns_params[:,1:,:].loc[:,:,"multi"].median( dim = "sample" )
	
	if verbose: pb.end()
	
	return climCB
##}}}


