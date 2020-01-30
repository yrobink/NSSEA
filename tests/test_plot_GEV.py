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
	is_test = False
	if is_test: np.random.seed(42) ## A good seed is always the big answer
	
	## Path
	##=====
	basepath = os.path.dirname(os.path.abspath(__file__))
	pathInp  = os.path.join( basepath ,  "input/Normal"  )
	pathOut  = os.path.join( basepath , "output/Normal" )
	assert(os.path.exists(pathInp))
	assert(os.path.exists(pathOut))
	
	
	## Some global parameters
	##=======================
	time_period    = np.arange( 1850 , 2101 , 1 , dtype = np.int )
	time_reference = np.arange( 1961 , 1991 , 1 , dtype = np.int )
	n_mcmc_drawn_min = 5000  
	n_mcmc_drawn_max = 10000 
	n_sample    = 1000 if not is_test else 10
	ns_law      = nsm.Normal()
#	event       = ns.Event( "HW19D3" , 2019 , None , time_reference , type_event = "hard" , name_variable = "TX_3D" , unit_variable = "K" )
	verbose     = True
	ci          = 0.05 if not is_test else 0.1
	
	## Load data
	##==========
#	climCXCB,_ = ns.from_netcdf( os.path.join( pathOut , "HW19D3_GEV_climCXCB.nc" ) , ns_law )
#	climCXCB,event = ns.from_netcdf( os.path.join( pathOut , "HW03_Normal_climCXCB.nc" ) , ns_law )
#	
#	ns.to_netcdf( climCXCB , event , "test.nc" , "CXCB" )
	
	c,e = ns.from_netcdf( "test.nc" )
	
#	nsp.probabilities( climCXCB , event , "Proba.pdf" )
	
	
	
	
	print("Done")


