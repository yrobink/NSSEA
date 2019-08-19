

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

from NSSEA.models.__NSGaussianModel import NSGaussianModel
from NSSEA.models.__NSGEVModel import NSGEVModel


from SDFC.tools import IdLinkFct
from SDFC.tools import ExpLinkFct


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

def constraints_CX( coffeeIn , Xo , cx_params , Sigma = None , verbose = False ): ##{{{
	"""
	NSSEA.constraintsCX
	===================
	Constrain covariates of coffee by the observed covariates Xo
	
	Arguments
	---------
	coffeeIn : NSSEA.Coffee
		coffee variable
	Xo       : pandas.DataFrame
		Covariate observed
	cx_params: NSSEA.CXParams
		Parameters
	Sigma    : Matrix
		Auto covariance matrix. If None, identity is used
	verbose  : bool
		Print (or not) state of execution
	
	Returns
	-------
	coffee : NSSEA.Coffee
		A COPY of coffeeIn constrained by Xo. coffeIn is NOT MODIFIED.
	"""
	
	if verbose: print( "Constraints CX" , end = "\r" )
	
	## Parameters
	coffee = coffeeIn.copy()
	time        = coffee.time
	time_Xo     = Xo.index
	n_time      = coffee.n_time
	n_time_Xo   = time_Xo.size
	n_mm_params = coffee.n_mm_params
	n_ns_params = coffee.n_ns_params
	n_sample    = coffee.n_sample
	sample      = coffee.X.sample
	Sigma       = Sigma if Sigma is not None else np.identity(n_time)
	Sigma       = xr.DataArray( Sigma , coords = [time,time] , dims = ["time1","time2"] )
	
	## Projection matrix H
	cx = xr.DataArray( np.zeros( (n_time,n_time) )       , coords = [time,time]       , dims = ["time1","time2"] )
	cy = xr.DataArray( np.zeros( (n_time_Xo,n_time_Xo) ) , coords = [time_Xo,time_Xo] , dims = ["time1","time2"] )
	if not cx_params.trust:
		cx.loc[:,cx_params.ref] = 1. / cx_params.ref.size
	if not cx_params.trust:
		cy.loc[:,cx_params.ref] = 1. / cx_params.ref.size
	
	centerX  = np.identity(n_time)    - cx.values
	centerY  = np.identity(n_time_Xo) - cy.values
	extractX = np.hstack( ( np.identity(n_time) , np.zeros( (n_time,n_time) ) , np.zeros( (n_time,n_ns_params) ) ) )
	H_full   = pd.DataFrame( centerX @ extractX , index = time )	# Centering * extracting
	H        = H_full.loc[time_Xo,:].values								# Restriction to the observed period
	
	
	# Other inputs : x, SX, y, SY
	X  = coffee.mm_params.mean
	SX = coffee.mm_params.cov
	Y  = np.ravel(centerY @ Xo)
	SY = centerY @ Sigma.loc[time_Xo,time_Xo].values @ centerY.T
	
	## Rescale SY
	if not cx_params.trust:
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
	cx_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,3) ) , coords = [ coffee.X.time , sample , coffee.X.forcing ] , dims = ["time","sample","forcing"] )
	
	cx_sample.loc[:,"be","all"] = gc.mean[:n_time]
	cx_sample.loc[:,"be","nat"] = gc.mean[n_time:(2*n_time)]
	
	for s in sample[1:]:
		draw = gc.mean + gc.std @ np.random.normal(size = n_mm_params)
		cx_sample.loc[:,s,"all"] = draw[:n_time]
		cx_sample.loc[:,s,"nat"] = draw[n_time:(2*n_time)]
	
	for m in coffee.X.models:
		coffee.X.loc[:,:,:,m] = cx_sample.values
	
	coffee.X.loc[:,:,"ant",:] = coffee.X.loc[:,:,"all",:] - coffee.X.loc[:,:,"nat",:]
	
	if verbose: print( "Constraints CX (Done)" )
	
	return coffee
##}}}


def constraints_C0_Gaussian( coffeeIn , Yo , event , verbose = False ): ##{{{
	if verbose: print( "Constraints C0 (Gaussian)" , end = "\r" )
	
	coffee  = coffeeIn.copy()
	n_models  = coffee.X.models.size
	n_sample  = coffee.n_sample
	models    = coffee.X.models
	time      = coffee.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = coffee.X.sample
	
	# New NS_param
	ns_params = coffee.ns_params
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] / ns_params.loc["scale0",:,:]
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,coffee.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,coffee.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = coffee.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = coffee.X.loc[idx,s,"all",m].values
	
	
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
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] * coffeeIn.ns_params.loc["scale0",:,:]
	coffee.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (Gaussian, Done)" )
	
	return coffee
##}}}

def constraints_C0_Gaussian_exp( coffeeIn , Yo , event , verbose = False ): ##{{{
	if verbose: print( "Constraints C0 (GaussianExp)" , end = "\r" )
	
	coffee  = coffeeIn.copy()
	n_models  = coffee.X.models.size
	n_sample  = coffee.n_sample
	models    = coffee.X.models
	time      = coffee.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = coffee.X.sample
	
	# New NS_param
	ns_params = coffee.ns_params
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,coffee.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,coffee.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = coffee.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = coffee.X.loc[idx,s,"all",m].values
	
	
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
	coffee.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GaussianExp, Done)" )
	
	return coffee
##}}}

def constraints_C0_GEV( coffeeIn , Yo , event , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEV)" , end = "\r" )
	
	coffee  = coffeeIn.copy()
	n_models  = coffee.X.models.size
	n_sample  = coffee.n_sample
	models    = coffee.X.models
	time      = coffee.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = coffee.X.sample
	
	# New NS_param
	ns_params = coffee.ns_params
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] / ns_params.loc["scale0",:,:]
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,coffee.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,coffee.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = coffee.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = coffee.X.loc[idx,s,"all",m].values
	
	## Fit loc0
	ns_params.loc["loc0",:,:] = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs ).quantile( np.exp(-1) , dim = "time" )
	
	## Fit scale0 and shape
	Yo_GEV_stats = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs - ns_params.loc["loc0",:,:]) / ( 1 + ns_params.loc["scale1",:,:] * Xo_bs ) ## Hypothesis : follow GEV(0,scale0,shape)
	for s in sample:
		for m in models:
			gev = sd.GEVLaw( method = coffee.ns_law_args["method"] , link_fct_shape = coffee.ns_law_args["link_fct_shape"] )
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , floc = 0 )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	
	## Save
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] * coffeeIn.ns_params.loc["scale0",:,:]
	coffee.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEV, Done)" )
	
	return coffee
##}}}

def constraints_C0_GEV_bound_valid( coffeeIn , Yo , event , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEV)" , end = "\r" )
	
	coffee  = coffeeIn.copy()
	n_models  = coffee.X.models.size
	n_sample  = coffee.n_sample
	models    = coffee.X.models
	time      = coffee.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = coffee.X.sample
	
	# New NS_param
	ns_params = coffee.ns_params
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] / ns_params.loc["scale0",:,:]
	
	
	gev = sd.GEVLaw(  method = coffee.ns_law_args["method"] , link_fct_shape = coffee.ns_law_args["link_fct_shape"] )
	for m in models:
		X   = coffee.X.loc[time_Yo,"be","all",m].values.squeeze()
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
				X   = coffee.X.loc[idx,s,"all",m].values.squeeze()
				Xcont = coffee.X.loc[time_Yo,s,"all",m].values.squeeze()
				Ybs = Yo.loc[idx].values.squeeze()
				ns_params.loc["loc0",s,m] = np.quantile( Ybs - float(ns_params.loc["loc1",s,m]) * X , np.exp(-1) )
				Yo_sta  = ( Ybs - float(ns_params.loc["loc1",s,m]) * X     - float(ns_params.loc["loc0",s,m]) ) / ( 1. + float(ns_params.loc["scale1",s,m]) * X )
				Yo_cont = ( Yo.values.squeeze()  - float(ns_params.loc["loc1",s,m]) * Xcont - float(ns_params.loc["loc0",s,m]) ) / np.exp( float(ns_params.loc["scale1",s,m]) * Xcont )
				gev.fit( Yo_sta , floc = 0 )
				
				## Here I test if the bound from params from bootstrap is compatible with observed values
				law = coffee.ns_law( **coffee.ns_law_args )
				law.set_params( np.array( [ns_params.loc["loc0",s,m],ns_params.loc["loc1",s,m],gev.coef_[0],ns_params.loc["scale1",s,m],gev.coef_[1]] , dtype = np.float ) )
				law.set_covariable( coffee.X.loc[time_Yo,s,"all",m].values , time_Yo )
#				print( np.all( Yo < law.upper_boundt(time_Yo) ) )
				test = np.all( np.logical_and( Yo.values.squeeze() > law.lower_boundt(time_Yo) , Yo.values.squeeze() < law.upper_boundt(time_Yo) ) )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	## Save
	ns_params.loc["scale1",:,:] = ns_params.loc["scale1",:,:] * coffeeIn.ns_params.loc["scale0",:,:]
	coffee.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEV, Done)" )
	
	return coffee
##}}}

def constraints_C0_GEV_exp( coffeeIn , Yo , event , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEVExp)" , end = "\r" )
	
	coffee  = coffeeIn.copy()
	n_models  = coffee.X.models.size
	n_sample  = coffee.n_sample
	models    = coffee.X.models
	time      = coffee.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = coffee.X.sample
	
	# New NS_param
	ns_params = coffee.ns_params
	
	## Bootstrap on Yo
	Yo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1) ) , coords = [time_Yo,coffee.X.sample] , dims = ["time","sample"] )
	Xo_bs = xr.DataArray( np.zeros( (n_time_Yo,n_sample+1,n_models) ) , coords = [time_Yo,coffee.X.sample,models] , dims = ["time","sample","models"] )
	Yo_bs.loc[:,"be"] = np.ravel(Yo)
	Xo_bs.loc[:,"be",:] = coffee.X.loc[time_Yo,"be","all",:]
	for s in sample[1:]:
		idx = np.random.choice( time_Yo , n_time_Yo , replace = True )
		Yo_bs.loc[:,s] = np.ravel( Yo.loc[idx].values )
		for m in models:
			Xo_bs.loc[:,s,m] = coffee.X.loc[idx,s,"all",m].values
	
	## Fit loc0
	ns_params.loc["loc0",:,:] = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs ).quantile( np.exp(-1) , dim = "time" )
	
	## Fit scale0 and shape
	Yo_GEV_stats = ( Yo_bs - ns_params.loc["loc1",:,:] * Xo_bs - ns_params.loc["loc0",:,:]) / np.exp( ns_params.loc["scale1",:,:] * Xo_bs ) ## Hypothesis : follow GEV(0,scale0,shape)
	for s in sample:
		for m in models:
			gev = sd.GEVLaw(  method = coffee.ns_law_args["method"] , link_fct_scale = ExpLinkFct() , link_fct_shape = coffee.ns_law_args["link_fct_shape"] )
			gev.fit( Yo_GEV_stats.loc[:,s,m].values , floc = 0 )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	
	## Save
	coffee.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEVExp, Done)" )
	
	return coffee
##}}}

def constraints_C0_GEV_exp_bound_valid( coffeeIn , Yo , event , verbose = False ): ##{{{
	
	if verbose: print( "Constraints C0 (GEVExp)" , end = "\r" )
	
	coffee  = coffeeIn.copy()
	n_models  = coffee.X.models.size
	n_sample  = coffee.n_sample
	models    = coffee.X.models
	time      = coffee.time
	time_Yo   = Yo.index
	n_time_Yo = Yo.size
	sample = coffee.X.sample
	
	# New NS_param
	ns_params = coffee.ns_params
	
	
	gev = sd.GEVLaw(  method = coffee.ns_law_args["method"] , link_fct_scale = ExpLinkFct() , link_fct_shape = coffee.ns_law_args["link_fct_shape"] )
	for m in models:
		X   = coffee.X.loc[time_Yo,"be","all",m].values.squeeze()
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
				X   = coffee.X.loc[idx,s,"all",m].values.squeeze()
				Xcont = coffee.X.loc[time_Yo,s,"all",m].values.squeeze()
				Ybs = Yo.loc[idx].values.squeeze()
				ns_params.loc["loc0",s,m] = np.quantile( Ybs - float(ns_params.loc["loc1",s,m]) * X , np.exp(-1) )
				Yo_sta  = ( Ybs - float(ns_params.loc["loc1",s,m]) * X     - float(ns_params.loc["loc0",s,m]) ) / np.exp( float(ns_params.loc["scale1",s,m]) * X )
				Yo_cont = ( Yo.values.squeeze()  - float(ns_params.loc["loc1",s,m]) * Xcont - float(ns_params.loc["loc0",s,m]) ) / np.exp( float(ns_params.loc["scale1",s,m]) * Xcont )
				gev.fit( Yo_sta , floc = 0 )
				
				## Here I test if the bound from params from bootstrap is compatible with observed values
				law = coffee.ns_law( **coffee.ns_law_args )
				law.set_params( np.array( [ns_params.loc["loc0",s,m],ns_params.loc["loc1",s,m],gev.coef_[0],ns_params.loc["scale1",s,m],gev.coef_[1]] , dtype = np.float ) )
				test = law.check( Yo.values.squeeze() , coffee.X.loc[time_Yo,s,"all",m] , time_Yo )
#				law.set_covariable( coffee.X.loc[time_Yo,s,"all",m].values , time_Yo )
#				print( np.all( Yo < law.upper_boundt(time_Yo) ) )
#				test = np.all( np.logical_and( Yo.values.squeeze() > law.lower_boundt(time_Yo) , Yo.values.squeeze() < law.upper_boundt(time_Yo) ) )
			ns_params.loc["scale0",s,m] = gev.coef_[0]
			ns_params.loc["shape",s,m]  = gev.coef_[1]
	
	## Save
	coffee.ns_params = ns_params
	
	if verbose: print( "Constraints C0 (GEVExp, Done)" )
	
	return coffee
##}}}

def constraints_C0( coffeeIn , Yo , event , gev_bound_valid = False , verbose = False ): ##{{{
	"""
	NSSEA.constraintsC0
	===================
	Constrain stationary parameters by observations
	
	Arguments
	---------
	coffeeIn : NSSEA.Coffee
		coffee variable
	Yo       : pandas.DataFrame
		Observations
	event    : NSSEA.Event
		Event
	gev_bound_valid : bool
		For GEVLaw, use only bootstrap where observations are between the bound of the GEV.
	verbose  : bool
		Print (or not) state of execution
	
	Returns
	-------
	coffee : NSSEA.Coffee
		A COPY of coffeeIn constrained by Yo. coffeeIn is NOT MODIFIED.
	"""
	
	if coffeeIn.ns_law == NSGaussianModel:
		if isinstance(coffeeIn.ns_law_args["link_fct_scale"],IdLinkFct):
			return constraints_C0_Gaussian( coffeeIn , Yo , event , verbose )
		elif isinstance(coffeeIn.ns_law_args["link_fct_scale"],ExpLinkFct):
			return constraints_C0_Gaussian_exp( coffeeIn , Yo , event , verbose )
	if coffeeIn.ns_law == NSGEVModel:
		if isinstance(coffeeIn.ns_law_args["link_fct_scale"],IdLinkFct):
			if gev_bound_valid:
				return constraints_C0_GEV_bound_valid( coffeeIn , Yo , event , verbose )
			else:
				return constraints_C0_GEV( coffeeIn , Yo , event , verbose )
		elif isinstance(coffeeIn.ns_law_args["link_fct_scale"],ExpLinkFct):
			if gev_bound_valid:
				return constraints_C0_GEV_exp_bound_valid( coffeeIn , Yo , event , verbose )
			else:
				return constraints_C0_GEV_exp( coffeeIn , Yo , event , verbose )
	
	return coffeeIn.copy()
##}}}



