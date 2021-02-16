# -*- coding: utf-8 -*-

#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import sys,os
import warnings
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

import numpy as np
import pandas as pd
import xarray as xr

import SDFC.link as sdl
import NSSEA as ns
import NSSEA.plot as nsp
import NSSEA.models as nsm

#from NSSEA.plot import LinkParams



####################
## Paramètres mpl ##
####################

#mpl.rcParams['font.size'] = 30
#plt.rc('text',usetex=True)
#plt.rcParams['text.latex.unicode'] = True


###############
## Fonctions ##
###############

def correct_miss( X , lo =  100 , up = 350 ):##{{{
	mod = str(X.columns[0])
	bad = np.logical_or( X < lo , X > up )
	bad = np.logical_or( bad , np.isnan(X) )
	bad = np.logical_or( bad , np.logical_not(np.isfinite(X)) )
	if np.any(bad):
		idx,_ = np.where(bad)
		idx_co = np.copy(idx)
		for i in range(idx.size):
			j = 0
			while idx[i] + j in idx:
				j += 1
			idx_co[i] += j
		X.iloc[idx] = X.iloc[idx_co].values
	return X
##}}}

def load_models_obs( path ):##{{{
	
	## List of models X
	##=================
	list_X = os.listdir( os.path.join( path , "X" ) )
	list_X.sort()
	
	modelsX = list()
	for lx in list_X:
		## Extract name
		split_name = lx.split(".")[0].split("_")
		mod = split_name[-2] + "_" + split_name[-1]
		modelsX.append(mod)
	
	## List of models Y
	##=================
	list_Y = os.listdir( os.path.join( path , "Y" ) )
	list_Y.sort()
	
	modelsY = list()
	for ly in list_Y:
		## Extract name
		split_name = ly.split(".")[0].split("_")
		mod = split_name[-2] + "_" + split_name[-1]
		modelsY.append(mod)
	
	## Merge the two lists to keep only common models
	##===============================================
	modelsX.sort()
	modelsY.sort()
	models = list(set(modelsX) & set(modelsY))
	models.sort()
	
	## Now load X
	##===========
	lX = list()
	for lx in list_X:
		## Extract name
		split_name = lx.split(".")[0].split("_")
		mod = split_name[-2] + "_" + split_name[-1]
		
		if mod in models:
			## Read model
			df = xr.open_dataset( os.path.join( path , "X" , lx ) , decode_times = False )
			time = np.array( df.time.values.tolist() , dtype = np.int )
			X = pd.DataFrame( df.tas.values.ravel() , columns = [mod] , index = time )
			lX.append( correct_miss(X) )
	
	## And load Y
	##===========
	lY = list()
	for ly in list_Y:
		## Extract name
		split_name = ly.split(".")[0].split("_")
		mod = split_name[-2] + "_" + split_name[-1]
		
		if mod in models:
			## Read model
			df = xr.open_dataset( os.path.join( path , "Y" , ly ) , decode_times = False )
			time = np.array( df.time.values.tolist() , dtype = np.int )
			Y = pd.DataFrame( df.pr.values.ravel() , columns = [mod] , index = time )
			lY.append( Y )
	
	## And finally load observations
	##==============================
	dXo = xr.open_dataset( os.path.join( path , "Xo.nc" ) )
	Xo  = pd.DataFrame( dXo.temperature_anomaly.values.squeeze() , columns = ["Xo"] , index = np.arange( 1850 , 2019 , 1 , dtype = np.int ) )
	
	dYo = xr.open_dataset( os.path.join( path , "Yo.nc" ) )
	Yo  = pd.DataFrame( dYo.rr.values.squeeze() , columns = ["Yo"] , index = np.arange( 1950 , 2019 , 1 , dtype = np.int ) )
	
	return models,lX,lY,Xo,Yo
##}}}

class NumpyLog: ##{{{
	def __init__(self):
		self._msg = []
	
	def __repr__(self):
		return self.__str__()
	
	def __str__(self):
		return "".join(self._msg)
	
	def write( self , msg ):
		self._msg.append(msg)
##}}}


##########
## main ##
##########

if __name__ == "__main__":
	
	## Test or not
	##============
	is_test = "--not-test" not in sys.argv
	set_seed = ~("--no-seed" in sys.argv)
	if set_seed: np.random.seed(42) ## A good seed is always the big answer
	
	## Set a log class, the GEV produces sometimes overflow
	##=====================================================
	nplog = NumpyLog()
	np.seterrcall(nplog)
	np.seterr( all = "log" , invalid = "log" )
	warnings.simplefilter("ignore")
	
	## Path
	##=====
	basepath = os.path.dirname(os.path.abspath(__file__))
	pathInp  = os.path.join( basepath , "input/GEVRLSC"  )
	pathOut  = os.path.join( basepath , "output/GEVRLSC" )
	assert(os.path.exists(pathInp))
	assert(os.path.exists(pathOut))
	
	## Some global parameters
	##=======================
	time_period    = np.arange( 1850 , 2101 , 1 , dtype = np.int )
	time_reference = np.arange( 1961 , 1991 , 1 , dtype = np.int )
	bayes_kwargs = { "n_mcmc_drawn_min" : 2500 if is_test else  5000 , "n_mcmc_drawn_max" : 5000 if is_test else 10000 , "min_rate_accept" : 0.05 , "keep" : "all" if is_test else 0.2 }
	n_sample    = 1000 if not is_test else 10
	ns_law      = nsm.GEVRLSC()
	event       = ns.Event( "FL13D4" , 2013 , time_reference , type_ = "hard" , variable = "PR4X" , unit = "mm" )
	verbose     = True
	ci          = 0.05 if not is_test else 0.1
	
	
	## Load models and observations
	##=============================
	models,lX,lY,Xo,Yo = load_models_obs( pathInp )
	
	
	## Remove models with problems
	##============================
#	for i in [3,-5,-5,-1,2]:
#		print(models[i])
#		del lY[i]
#		del lX[i]
#		del models[i]
	
	
	## Anomaly from observations
	##==========================
	event.anomaly = float(Yo.loc[event.time])
	
	
	## Models in anomaly
	##==================
	for X in lX:
		X -= X.loc[event.reference].mean()
	for Y in lY:
		Y *= 86400
	
	
	## Define clim variable from input
	##================================
	clim = ns.Climatology( event , time_period , models , n_sample , ns_law )
	
	
	## Decomposition of covariates
	##============================
	Xebm   = ns.EBM().draw_sample( clim.time , n_sample + 1 , fix_first = 0 )
	clim   = ns.covariates_FC_GAM( clim , lX , Xebm , verbose = verbose )
	
	
	## Fit distribution
	##=================
	clim = ns.nslaw_fit( lY , clim , verbose = verbose )
	
	
	## Multi-model
	##============
	clim = ns.infer_multi_model( clim , verbose = verbose )
	climMM = clim.copy()
	climMM.keep_models( "Multi_Synthesis" )
	
	
	## Apply constraints
	##==================
	climCX     = ns.constrain_covariate( climMM , Xo , time_reference , verbose = verbose )
	climCXCB   = ns.constrain_law( climCX , Yo , verbose = verbose , **bayes_kwargs )
	
	
	## Compute stats
	##==============
	clim       = ns.extreme_statistics( clim     , verbose = verbose )
	climCX     = ns.extreme_statistics( climCX   , verbose = verbose )
	climCXCB   = ns.extreme_statistics( climCXCB , verbose = verbose )
	
	params     = ns.build_params_along_time( clim     , verbose = verbose )
	paramsCX   = ns.build_params_along_time( climCX   , verbose = verbose )
	paramsCXCB = ns.build_params_along_time( climCXCB , verbose = verbose )
	
	
	## Save in netcdf
	##===============
	for c,s in zip([clim,climCX,climCXCB],["","CX","CXCB"]):
		c.to_netcdf( os.path.join( pathOut , "{}_clim{}.nc".format(event.name,s) ) )
	for p,s in zip([params,paramsCX,paramsCXCB],["","CX","CXCB"]):
		p.to_dataset( name = "params{}".format(s) ).to_netcdf( os.path.join( pathOut , "{}_params{}.nc".format(event.name,s) ) )
	
	## Reload
	##=======
	clim,climCX,climCXCB = ( ns.Climatology.from_netcdf( os.path.join( pathOut , "{}_clim{}.nc".format(event.name,s) ) , ns_law ) for s in ["","CX","CXCB"] )
	params,paramsCX,paramsCXCB = ( xr.open_dataset( os.path.join( pathOut , "{}_params{}.nc".format(event.name,s) ) )["params{}".format(s)] for s in ["","CX","CXCB"] )
	
	## Plot
	##=====
	pltkwargs = { "verbose" : verbose , "ci" : ci }
	nsp.GAM_decomposition( clim , lX , os.path.join( pathOut , "GAM_decomposition.pdf" ) , **pltkwargs )
	nsp.constraint_covariate( clim , climCXCB , Xo , os.path.join( pathOut , "constraint_covariate.pdf" )  , **pltkwargs )
	nsp.summary( clim     , pathOut , t1 = 2040 , params = params     , **pltkwargs )
	nsp.summary( climCX   , pathOut , t1 = 2040 , params = paramsCX   , suffix = "CX"   , **pltkwargs )
	nsp.summary( climCXCB , pathOut , t1 = 2040 , params = paramsCXCB , suffix = "CXCB" , **pltkwargs )
	nsp.constraint_law( climCX , climCXCB , ofile = os.path.join( pathOut , "constraint_law.pdf" ) , **pltkwargs )
	nsp.statistics_time( [clim,climCX,climCXCB] , os.path.join( pathOut , "Statistics_time.pdf" ) , labels = clim.model.tolist() + ["Multi_CX","Multi_CXCB"] , colors = ["red","blue","green"] , **pltkwargs )
	
	
	print("Done")


