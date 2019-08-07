
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import numpy  as np
import xarray as xr

from NSSEA.__tools import ProgressBar


###############
## Functions ##
###############

def nslaw_fit( lY , coffee , verbose = False ):
	"""
	NSSEA.nslaw_fit
	===============
	Fit non stationary parameters
	
	Arguments
	---------
	lY     : list
		List of models
	coffee : NSSEA.Coffee
		Coffee variable
	verbose: bool
		Print or not state of execution
	
	Return
	------
	coffee : NSSEA.coffee
		Coffee variable with ns_params fitted
	"""
	## Parameters
	models      = coffee.models
	n_models    = coffee.n_models
	sample      = coffee.X.sample.values.tolist()
	n_sample    = coffee.n_sample
	ns_law      = coffee.ns_law
	ns_law_args = coffee.ns_law_args
	
#	n_ns_params        = 4
#	ns_params_names    = [ "loc0" , "loc1" , "scale0" , "scale1" ]
	ns_params_info     = ns_law.params_info(ns_law_args)
	n_ns_params        = ns_params_info["size"]
	ns_params_names    = ns_params_info["names"]
	
	coffee.ns_params   = xr.DataArray( np.zeros( (n_ns_params,n_sample + 1,n_models) ) , coords = [ ns_params_names , sample , models ] , dims = ["ns_params","sample","models"] )
	coffee.n_ns_params = n_ns_params
	
	pb = ProgressBar( "NS fit" , n_models * n_sample )
	for i in range(n_models):
		Y  = lY[i]
		tY = Y.index
		X  = coffee.X.loc[tY,"be","all",models[i]]
		
		law = ns_law(**ns_law_args)
		law.fit(Y.values,X.values)
		coffee.ns_params.loc[:,"be",models[i]] = law.get_params()
		
		for j in range(n_sample):
			if verbose: pb.print()
			
			idx = np.random.choice( tY.size , tY.size , replace = True )
			
			tYs = tY.values[idx]
			Ys = Y.iloc[idx].values
			Xs = coffee.X.loc[tYs,sample[j+1],"all",models[i]].values
			law = ns_law(**ns_law_args)
			law.fit(Ys,Xs)
			coffee.ns_params.loc[:,sample[j+1],models[i]] = law.get_params()
	
	if verbose: pb.end()
	
	return coffee



