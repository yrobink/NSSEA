

#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import numpy as np
import scipy.linalg as scl
import scipy.optimize as sco
import pandas as pd
import xarray as xr

from NSSEA.__tools import matrix_squareroot

from NSSEA.models.__Normal import Normal
from NSSEA.models.__GEV    import GEV


from SDFC.tools import IdLink
from SDFC.tools import ExpLink


import SDFC as sd


#############
## Classes ##
#############

class GenericConstraint: ##{{{
	"""
	NSSEA.GenericConstraint
	=======================
	
	Attributes
	----------
	mean   : array
		CX mean
	cov    : array
		CX covariance matrix
	std    : array
		Square root of CX covariance matrix
	
	"""
	def __init__( self , x , SX , y , SY , H ):
		"""
		Do not use, call NSSEA.constraints_CX
		"""
		Sinv      = np.linalg.pinv( H @ SX @ H.T + SY )
		K	      = SX @ H.T @ Sinv
		self.mean = x + K @ ( y - H @ x )
		self.cov  = SX - SX @ H.T @ Sinv @ H @ SX
		self.std  = matrix_squareroot(self.cov)
##}}}

###############
## Functions ##
###############

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
	X  = clim.mm_params.mean
	SX = clim.mm_params.cov
	Y  = np.ravel(centerY @ Xo)
	SY = centerY @ centerY.T
	
	## Rescale SY
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
	
	## Constraints and sample
	gc = GenericConstraint( X , SX , Y , SY , H )
	cx_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,3) ) , coords = [ clim.X.time , sample , clim.X.forcing ] , dims = ["time","sample","forcing"] )
	
	cx_sample.loc[:,"be","all"] = gc.mean[:n_time]
	cx_sample.loc[:,"be","nat"] = gc.mean[n_time:(2*n_time)]
	
	for s in sample[1:]:
		draw = gc.mean + gc.std @ np.random.normal(size = n_mm_params)
		cx_sample.loc[:,s,"all"] = draw[:n_time]
		cx_sample.loc[:,s,"nat"] = draw[n_time:(2*n_time)]
	
	for m in clim.X.models:
		clim.X.loc[:,:,:,m] = cx_sample.values
	
	clim.X.loc[:,:,"ant",:] = clim.X.loc[:,:,"all",:] - clim.X.loc[:,:,"nat",:]
	
	if verbose: print( "Constraints CX (Done)" )
	
	return clim
##}}}


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
	
	## Fit loc0
	ns_params.loc["loc0",:,:] = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs ).quantile( np.exp(-1) , dim = "time" )
	
	## Fit scale0 and shape
	Yo_GEV_stats = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs - ns_params.loc["loc0",:,:]) / ( 1 + ns_params.loc["scale1",:,:] * Xo_bs ) ## Hypothesis : follow GEV(0,scale0,shape)
	for s in sample:
		for m in models:
			gev = sd.GEVLaw( method = clim.ns_law_args["method"] , link_fct_shape = clim.ns_law_args["link_fct_shape"] )
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , floc = 0 )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	
	## Save
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] * climIn.ns_params.loc["scale0",:,:]
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEV, Done)" )
	
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
			gev = sd.GEVLaw(  method = clim.ns_law_args["method"] , link_fct_scale = ExpLink() , link_fct_shape = clim.ns_law_args["link_fct_shape"] )
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , floc = 0 )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	
	## Save
	clim.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEVExp, Done)" )
	
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
	
	if climIn.ns_law == Normal:
		if isinstance(climIn.ns_law_args["link_scale"],IdLink):
			return constraints_C0_Normal( climIn , Yo , verbose )
		elif isinstance(climIn.ns_law_args["link_scale"],ExpLink):
			return constraints_C0_Normal_exp( climIn , Yo , verbose )
	if climIn.ns_law == GEV:
		if isinstance(climIn.ns_law_args["link_scale"],IdLink):
			if gev_bound_valid:
				return constraints_C0_GEV_bound_valid( climIn , Yo , verbose )
			else:
				return constraints_C0_GEV( climIn , Yo , verbose )
		elif isinstance(climIn.ns_law_args["link_scale"],ExpLink):
			if gev_bound_valid:
				return constraints_C0_GEV_exp_bound_valid( climIn , Yo , verbose )
			else:
				return constraints_C0_GEV_exp( climIn , Yo , verbose )
	
	return climIn.copy()
##}}}



