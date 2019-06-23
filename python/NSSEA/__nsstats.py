
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import sys
import numpy as np
import xarray as xr

from NSSEA.__tools import ProgressBar


###############
## Functions ##
###############

def extremes_stats( coffee , event , threshold_by_world = False , verbose = False , tol = sys.float_info.epsilon ):##{{{
	"""
	NSSEA.extremes_stats
	====================
	Compute extremes statistics and add it to a Coffee.
	
	Arguments
	---------
	coffee : NSSEA.Coffee
		A coffee variable
	event  : NSSEA.Event
		An event variable
	threshold_by_world: bool
		If False, the treshold is the same for the factual and counter-factual world, otherwise different threshold is used
	verbose: bool
		Print state of execution or not
	tol    : float
		Numerical tolerance, default is sys.float_info.epsilon
	
	Return
	------
	coffee : NSSEA.Coffee
		A coffee variable with coffee.stats set
	
	Statistics computed
	-------------------
	The variable coffee.stats is an xarray with dimensions (n_time,n_sample+1,n_stats,n_models), stats available are:
	
	pall: Probability of event.anom at time event.time in factual world
	pnat: Probability of event.anom at time event.time in counter factual world
	rr  : Risk ratio (pall / pnat)
	iall: Event with same probability than the probability of event.anom at each time in factual world
	inat: Event with same probability than the probability of event.anom at each time in counter factual world
	di  : iall - inat
	
	"""
	## Usefull variables
	time           = coffee.time
	n_time         = coffee.n_time
	models         = coffee.models
	n_models       = coffee.n_models
	n_sample       = coffee.n_sample
	n_stats        = 6
	upper_side     = event.side == "high"
	coffee.n_stats = n_stats
	event_time     = event.time
	
	
	## Output
	stats = xr.DataArray( np.zeros( (n_time,n_sample + 1,n_stats,n_models) ) , coords = [coffee.X.time , coffee.X.sample , ["pnat","pall","rr","inat","iall","di"] , coffee.X.models ] , dims = ["time","sample","stats","models"] )
	
	## 
	law = coffee.ns_law( **coffee.ns_law_args )
	pb = ProgressBar( "Statistics" , n_models * (n_sample + 1) )
	for m in coffee.X.models:
		for s in coffee.X.sample:
			if verbose: pb.print()
			
			law.set_params( coffee.ns_params.loc[:,s,m].values )
			
			## Find threshold
			law.set_covariable( coffee.X.loc[:,s,"all",m].values , time )
			threshold = np.zeros(n_time) + np.mean( law.meant(event.ref_anom) ) + event.anom
			
			## Find pall
			stats.loc[:,s,"pall",m] = law.sf( threshold , time ) if upper_side else law.cdf( threshold , time )
			
			## Find probability of the event in factual world
			pf = np.zeros(n_time) + ( law.sf( np.array([threshold[0]]) , np.array([event_time]) ) if upper_side else law.cdf( np.array([threshold[0]]) , np.array([event_time]) ) )
			pf[ np.logical_not(pf>0) ] = tol
			pf[ np.logical_not(pf<1) ] = 1. - tol
			
			## I1
			stats.loc[:,s,"iall",m] = law.isf( pf , time ) if upper_side else law.icdf( pf , time )
			
			## Find pnat
			law.set_covariable( coffee.X.loc[:,s,"nat",m].values , time )
			if threshold_by_world:
	 			threshold = np.zeros(n_time) + np.mean( law.meant(event.ref_anom) ) + event.anom
			stats.loc[:,s,"pnat",m] = law.sf( threshold , time ) if upper_side else law.cdf( threshold , time )
			
			## I0
			stats.loc[:,s,"inat",m] = law.isf( pf , time ) if upper_side else law.icdf( pf , time )
			
	
#	stats.loc[:,:,"pall",:] = stats.loc[:,:,"pall",:].where( stats.loc[:,:,"pall",:] > tol , tol )
#	stats.loc[:,:,"pnat",:] = stats.loc[:,:,"pnat",:].where( stats.loc[:,:,"pnat",:] > tol , tol )
	
	## RR
	stats.loc[:,:,"rr",:] = stats.loc[:,:,"pall",:] / stats.loc[:,:,"pnat",:]
	stats.loc[:,:,"rr",:] = stats.loc[:,:,"rr",:].where( stats.loc[:,:,"rr",:] > 0 , np.inf )
	
	
	## deltaI
	stats.loc[:,:,"di",:] = stats.loc[:,:,"iall",:] - stats.loc[:,:,"inat",:]
	
	coffee.stats = stats
	if verbose: pb.end()
	
	return coffee
##}}}

def RR_correction( S , tol = 1e-10 ):##{{{
	
	## 
	idx_all = S.loc[:,:,"pall",:] < tol
	idx_nat = S.loc[:,:,"pnat",:] < tol
	cidx_all = np.logical_not(idx_all)
	cidx_nat = np.logical_not(idx_nat)
	
	S.loc[:,:,"pall",:] = S.loc[:,:,"pall",:].where( cidx_all , np.nan )
	S.loc[:,:,"pnat",:] = S.loc[:,:,"pnat",:].where( cidx_nat , np.nan )
	
	## Remove all problematic values from RR
	idx = np.logical_and( cidx_all , cidx_nat )
	S.loc[:,:,"rr",:]   = S.loc[:,:,"rr",:].where( idx , np.nan )
	
	## pall > 0, pnat > 0, Don't touch!!!
	
	## pall > 0, pnat = 0
	idx = np.logical_not( np.logical_and( cidx_all , idx_nat ) )
	S.loc[:,:,"rr",:]   = S.loc[:,:,"rr",:].where( idx , np.Inf )
	
	## pall = 0, pnat > 0
	idx = np.logical_not( np.logical_and( idx_all , cidx_nat ) )
	S.loc[:,:,"rr",:]   = S.loc[:,:,"rr",:].where( idx , 0. )
	
	## pall = 0, pnat = 0, here we replace by 1
	idx = np.logical_not( np.logical_and( idx_all , idx_nat ) )
	S.loc[:,:,"rr",:] = S.loc[:,:,"rr",:].where( idx , 1 )
	
	## 
	S.loc[:,:,"pall",:] = S.loc[:,:,"pall",:].where( cidx_all , 0 )
	S.loc[:,:,"pnat",:] = S.loc[:,:,"pnat",:].where( cidx_nat , 0 )
	
	return S
##}}}

def stats_relative_event( stats , time_event ):##{{{
	"""
	NSSEA.stats_relative_event
	==========================
	Compute relative statistics at a fixed time
	
	Arguments
	---------
	stats     : xarray
		coffee.stats
	time_event: Time of event
	
	Return
	------
	satstEvent: xarray
		Similar to coffee.stats
	"""
	statsEvent = xr.zeros_like(stats)
	statsEvent[:,:,:3,:] = stats[:,:,:3,:] / stats.loc[time_event,:,["pall","pnat","rr"],:]
	statsEvent[:,:,3:,:] = stats[:,:,3:,:] - stats.loc[time_event,:,["iall","inat","di"],:]
	return statsEvent
##}}}

