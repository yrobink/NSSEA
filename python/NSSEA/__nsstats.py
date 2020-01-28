
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

def extremes_stats( clim , event , verbose = False , tol = sys.float_info.epsilon ):##{{{
	"""
	NSSEA.extremes_stats
	====================
	Compute extremes statistics and add it to a Climatology.
	
	Arguments
	---------
	clim : NSSEA.Climatology
		A clim variable
	event  : NSSEA.Event
		An event variable
	verbose: bool
		Print state of execution or not
	tol    : float
		Numerical tolerance, default is sys.float_info.epsilon
	
	Return
	------
	clim : NSSEA.Climatology
		A clim variable with clim.stats set
	
	Statistics computed
	-------------------
	The variable clim.stats is an xarray with dimensions (n_time,n_sample+1,n_stats,n_models), stats available are:
	
	pF: Probability of event.anom at time event.time in factual world
	pC: Probability of event.anom at time event.time in counter factual world
	PR: Probability ratio (pF / pC)
	IF: Event with same probability than the probability of event.anom at each time in factual world
	IC: Event with same probability than the probability of event.anom at each time in counter factual world
	dI: IF - IC
	
	"""
	## Usefull variables
	time           = clim.time
	n_time         = clim.n_time
	models         = clim.models
	n_models       = clim.n_models
	n_sample       = clim.n_sample
	n_stats        = 6
	upper_side     = event.side == "upper"
	event_time     = event.time
	
	
	## Output
	stats = xr.DataArray( np.zeros( (n_time,n_sample + 1,n_stats,n_models) ) , coords = [clim.X.time , clim.X.sample , ["pC","pF","PR","IC","IF","dI"] , clim.X.models ] , dims = ["time","sample","stats","models"] )
	
	## 
	law = clim.ns_law( **clim.ns_law_args )
	pb = ProgressBar( "Statistics" , n_models * (n_sample + 1) )
	for m in clim.X.models:
		for s in clim.X.sample:
			if verbose: pb.print()
			
			law.set_params( clim.ns_params.loc[:,s,m].values )
			
			## Find threshold
			law.set_covariable( clim.X.loc[:,s,"all",m].values , time )
			if event.type_event == "threshold":
				threshold = np.zeros(n_time) + np.mean( law.meant(event.reference) ) + event.anomaly
			else:
				threshold = np.zeros(n_time) + event.anomaly
			
			## Find pF
			stats.loc[:,s,"pF",m] = law.sf( threshold , time ) if upper_side else law.cdf( threshold , time )
			
			## Find probability of the event in factual world
			pf = np.zeros(n_time) + ( law.sf( np.array([threshold[0]]) , np.array([event.time]) ) if upper_side else law.cdf( np.array([threshold[0]]) , np.array([event.time]) ) )
			
			## I1
			stats.loc[:,s,"IF",m] = law.isf( pf , time ) if upper_side else law.icdf( pf , time )
			
			## Find pC
			law.set_covariable( clim.X.loc[:,s,"nat",m].values , time )
			stats.loc[:,s,"pC",m] = law.sf( threshold , time ) if upper_side else law.cdf( threshold , time )
			
			## I0
			stats.loc[:,s,"IC",m] = law.isf( pf , time ) if upper_side else law.icdf( pf , time )
	
	
	## RR
	stats.loc[:,:,"PR",:] = stats.loc[:,:,"pF",:] / stats.loc[:,:,"pC",:]
#	stats.loc[:,:,"PR",:] = stats.loc[:,:,"PR",:].where( stats.loc[:,:,"PR",:] > 0 , np.inf )
	
	
	## deltaI
	stats.loc[:,:,"dI",:] = stats.loc[:,:,"IF",:] - stats.loc[:,:,"IC",:]
	
	clim.stats = stats
	if verbose: pb.end()
	
	return clim
##}}}

def PR_correction( S , tol = 1e-10 ):##{{{
	
	## 
	idx_all = S.loc[:,:,"pF",:] < tol
	idx_nat = S.loc[:,:,"pC",:] < tol
	cidx_all = np.logical_not(idx_all)
	cidx_nat = np.logical_not(idx_nat)
	
	S.loc[:,:,"pF",:] = S.loc[:,:,"pF",:].where( cidx_all , np.nan )
	S.loc[:,:,"pC",:] = S.loc[:,:,"pC",:].where( cidx_nat , np.nan )
	
	## Remove all problematic values from RR
	idx = np.logical_and( cidx_all , cidx_nat )
	S.loc[:,:,"PR",:]   = S.loc[:,:,"PR",:].where( idx , np.nan )
	
	## pF > 0, pC > 0, Don't touch!!!
	
	## pF > 0, pC = 0
	idx = np.logical_not( np.logical_and( cidx_all , idx_nat ) )
	S.loc[:,:,"PR",:]   = S.loc[:,:,"PR",:].where( idx , np.Inf )
	
	## pF = 0, pC > 0
	idx = np.logical_not( np.logical_and( idx_all , cidx_nat ) )
	S.loc[:,:,"PR",:]   = S.loc[:,:,"PR",:].where( idx , 0. )
	
	## pF = 0, pC = 0, here we replace by 1
	idx = np.logical_not( np.logical_and( idx_all , idx_nat ) )
	S.loc[:,:,"PR",:] = S.loc[:,:,"PR",:].where( idx , 1 )
	
	## 
	S.loc[:,:,"pF",:] = S.loc[:,:,"pF",:].where( cidx_all , 0 )
	S.loc[:,:,"pC",:] = S.loc[:,:,"pC",:].where( cidx_nat , 0 )
	
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
		clim.stats
	time_event: Time of event
	
	Return
	------
	satstEvent: xarray
		Similar to clim.stats
	"""
	statsEvent = xr.zeros_like(stats)
	statsEvent[:,:,:3,:] = stats[:,:,:3,:] / stats.loc[time_event,:,["pF","pC","PR"],:]
	statsEvent[:,:,3:,:] = stats[:,:,3:,:] - stats.loc[time_event,:,["IF","IC","dI"],:]
	return statsEvent
##}}}

