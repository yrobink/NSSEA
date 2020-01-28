
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

def nslaw_fit( lY , clim , verbose = False ):
	"""
	NSSEA.nslaw_fit
	===============
	Fit non stationary parameters
	
	Arguments
	---------
	lY     : list
		List of models
	clim : NSSEA.Climatology
		Climatology variable
	verbose: bool
		Print or not state of execution
	
	Return
	------
	clim : NSSEA.clim
		Climatology variable with ns_params fitted
	"""
	## Parameters
	models      = clim.models
	n_models    = clim.n_models
	sample      = clim.X.sample.values.tolist()
	n_sample    = clim.n_sample
	ns_law      = clim.ns_law
	ns_law_args = clim.ns_law_args
	
	ns_params_info     = ns_law.params_info(ns_law_args)
	n_ns_params        = ns_params_info["size"]
	ns_params_names    = ns_params_info["names"]
	
	clim.ns_params   = xr.DataArray( np.zeros( (n_ns_params,n_sample + 1,n_models) ) , coords = [ ns_params_names , sample , models ] , dims = ["ns_params","sample","models"] )
	
	pb = ProgressBar( "NS fit" , n_models * n_sample )
	for i in range(n_models):
		Y  = lY[i]
		tY = Y.index
		X  = clim.X.loc[tY,"be","all",models[i]]
		
		law = ns_law(**ns_law_args)
		law.fit(Y.values,X.values)
		clim.ns_params.loc[:,"be",models[i]] = law.get_params()
		
		for j in range(n_sample):
			if verbose: pb.print()
			
			fit_is_valid = False
			while not fit_is_valid:
			
				idx = np.random.choice( tY.size , tY.size , replace = True )
				
				tYs = tY.values[idx]
				Ys = Y.iloc[idx].values
				Xs = clim.X.loc[tYs,sample[j+1],"all",models[i]].values
				law = ns_law(**ns_law_args)
				law.fit(Ys,Xs)
				fit_is_valid = law.check( Y.values.squeeze() , X.values.squeeze() , np.arange( 0 , tY.size , 1 ) )
			clim.ns_params.loc[:,sample[j+1],models[i]] = law.get_params()
	
	if verbose: pb.end()
	
	return clim



