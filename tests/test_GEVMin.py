# -*- coding: utf-8 -*-

#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import sys,os
import pickle as pk
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

import numpy as np
import pandas as pd
import xarray as xr

import SDFC.tools as sdt
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
#	return X
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
			Y = pd.DataFrame( df.tas.values.ravel() , columns = [mod] , index = time )
			lY.append( correct_miss(Y) )
	
	## And finally load observations
	##==============================
	dXo = xr.open_dataset( os.path.join( path , "Xo.nc" ) )
	Xo  = pd.DataFrame( dXo.temperature_anomaly.values.squeeze() , columns = ["Xo"] , index = np.arange( 1850 , 2019 , 1 , dtype = np.int ) )
	
	dYo = xr.open_dataset( os.path.join( path , "Yo.nc" ) )
	Yo  = pd.DataFrame( dYo.Tm.values.squeeze() , columns = ["Yo"] , index = np.arange( 1947 , 2020 , 1 , dtype = np.int ) )
	
	return models,lX,lY,Xo,Yo
##}}}


####################
## Plot functions ##
####################

#############
## Classes ##
#############

##########
## main ##
##########

if __name__ == "__main__":
	
	## Test or not
	##============
	is_test = True
	if is_test: np.random.seed(42) ## A good seed is always the big answer
	
	## Path
	##=====
	basepath = os.path.dirname(os.path.abspath(__file__))
	pathInp  = os.path.join( basepath , "input/GEV"  )
	pathOut  = os.path.join( basepath , "output/GEVMin" )
	assert(os.path.exists(pathInp))
	assert(os.path.exists(pathOut))
	
	
	## Some global parameters
	##=======================
	time_period    = np.arange( 1850 , 2101 , 1 , dtype = np.int )
	time_reference = np.arange( 1961 , 1991 , 1 , dtype = np.int )
	n_mcmc_drawn_min = 500  if is_test else  5000
	n_mcmc_drawn_max = 1000 if is_test else 10000
	n_sample    = 1000 if not is_test else 10
	ns_law      = nsm.GEVMin()
	event       = ns.Event( "HW19D3_min" , 2019 , None , time_reference , type_event = "hard" , side = "lower" , name_variable = "-TX_3D" , unit_variable = "K" )
	verbose     = True
	ci          = 0.05 if not is_test else 0.1
	
	
	## Load models and observations
	##=============================
	models,lX,lY,Xo,Yo = load_models_obs( pathInp )
	
	
	## Anomaly from observations
	##==========================
	Yo -= Yo.loc[event.reference].mean()
	Yo = -Yo
	event.anomaly = float(Yo.loc[event.time])
	
	
	## Models in anomaly
	##==================
	
	for X in lX:
		X -= X.loc[event.reference].mean()
	for i,Y in enumerate(lY):
		Y -= Y.loc[event.reference].mean()
		lY[i] = -Y
	
	
	## Define clim variable from input
	##================================
	clim = ns.Climatology( time_period , models , ns_law )
	
	
	## Decomposition of covariates
	##============================
	Xebm   = ns.EBM().draw_sample( clim.time , n_sample + 1 , fix_first = 0 )
	clim.X = ns.gam_decomposition( lX , Xebm , verbose = verbose )
	
	
	## Fit distribution
	##=================
	clim = ns.nslaw_fit( lY , clim , verbose = verbose )
	
	
	## Multi-model
	##============
	climMM = ns.infer_multi_model( clim , verbose = verbose )
	
	
	## Keep only multi-model
	##======================
	climMM.keep_models( ["multi"] )
	
	
	## Apply constraints
	##==================
	climCX     = ns.constraints_CX( climMM , Xo , time_reference = time_reference , verbose = verbose )
	climCXCB   = ns.constraints_bayesian( climCX , Yo , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose = verbose )
	climC0     = ns.constraints_C0( climMM , Yo , verbose = verbose )
	climCXC0   = ns.constraints_C0( climCX , Yo , verbose = verbose )
	climCXC0CB = ns.constraints_bayesian( climCXC0 , Yo , n_mcmc_drawn_min , n_mcmc_drawn_max , verbose = verbose )
	
	## Compute stats
	##==============
	climMM     = ns.extremes_stats( climMM     , event , verbose = verbose )
	climCX     = ns.extremes_stats( climCX     , event , verbose = verbose )
	climCXCB   = ns.extremes_stats( climCXCB   , event , verbose = verbose )
	climC0     = ns.extremes_stats( climC0     , event , verbose = verbose )
	climCXC0   = ns.extremes_stats( climCXC0   , event , verbose = verbose )
	climCXC0CB = ns.extremes_stats( climCXC0CB , event , verbose = verbose )
	
	
	## Save in netcdf
	##===============
	ns.to_netcdf( climMM     , event , os.path.join( pathOut , "HW19D3_GEVMin_clim.nc"       ) , ""       )
	ns.to_netcdf( climCX     , event , os.path.join( pathOut , "HW19D3_GEVMin_climCX.nc"     ) , "CX"     )
	ns.to_netcdf( climCXCB   , event , os.path.join( pathOut , "HW19D3_GEVMin_climCXCB.nc"   ) , "CXCB"   )
	ns.to_netcdf( climC0     , event , os.path.join( pathOut , "HW19D3_GEVMin_climC0.nc"     ) , "C0"     )
	ns.to_netcdf( climCXC0   , event , os.path.join( pathOut , "HW19D3_GEVMin_climCXC0.nc"   ) , "CXC0"   )
	ns.to_netcdf( climCXC0CB , event , os.path.join( pathOut , "HW19D3_GEVMin_climCXC0CB.nc" ) , "CXC0CB" )
	
	
	## Plot
	##=====
	nsp.write_package_tabular( climCXCB   , event , os.path.join( pathOut , "SummaryCXCB.txt"   ) , verbose = verbose )
	nsp.write_package_tabular( climCXC0   , event , os.path.join( pathOut , "SummaryCXC0.txt"   ) , verbose = verbose )
	nsp.write_package_tabular( climCXC0CB , event , os.path.join( pathOut , "SummaryCXC0CB.txt" ) , verbose = verbose )
	nsp.decomposition( lX , clim.X , event                      , os.path.join( pathOut , "decomposition.pdf" ) , verbose = verbose )
	nsp.constraints_CX( climMM , climCXCB , Xo , time_reference , os.path.join( pathOut , "constraintCX.pdf" )  , verbose = verbose )
	nsp.plot_classic_packages( climMM     , event , path = pathOut , suffix = "MM"     , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCX     , event , path = pathOut , suffix = "CX"     , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCXCB   , event , path = pathOut , suffix = "CXCB"   , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climC0     , event , path = pathOut , suffix = "C0"     , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCXC0   , event , path = pathOut , suffix = "CXC0"   , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCXC0CB , event , path = pathOut , suffix = "CXC0CB" , ci = ci , verbose = verbose )
	nsp.ns_params_comparison( climMM , climCXC0   , ofile = os.path.join( pathOut , "ns_paramsMM_CXC0.pdf"   ) , ci = ci , verbose = verbose )
	nsp.ns_params_comparison( climMM , climCXCB   , ofile = os.path.join( pathOut , "ns_paramsMM_CXCB.pdf"   ) , ci = ci , verbose = verbose )
	nsp.ns_params_comparison( climMM , climCXC0CB , ofile = os.path.join( pathOut , "ns_paramsMM_CXC0CB.pdf" ) , ci = ci , verbose = verbose )
	
	print("Done")


