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
			lX.append( X )
	
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
			lY.append( Y )
	
	## And finally load observations
	##==============================
	dXo = xr.open_dataset( os.path.join( path , "Xo.nc" ) )
	Xo  = pd.DataFrame( dXo.temperature_anomaly.values.squeeze() , columns = ["Xo"] , index = np.arange( 1850 , 2019 , 1 , dtype = np.int ) )
	
	dYo = xr.open_dataset( os.path.join( path , "Yo.nc" ) )
	Yo  = pd.DataFrame( dYo.temperature_anomaly.values.squeeze() , columns = ["Yo"] , index = np.arange( 1850 , 2019 , 1 , dtype = np.int ) )
	
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
	pathInp  = os.path.join( basepath , "input/Normal"  )
	pathOut  = os.path.join( basepath , "output/Normal" )
	assert(os.path.exists(pathInp))
	assert(os.path.exists(pathOut))
	
	
	## Some global parameters
	##=======================
	time_period    = np.arange( 1850 , 2101 , 1 , dtype = np.int )
	time_reference = np.arange( 1961 , 1991 , 1 , dtype = np.int )
	
	n_sample    = 1000 if not is_test else 10
	ns_law      = nsm.Normal
	ns_law_args = { "link_scale" : sdt.ExpLink() }
#	ns_law_args = { "link_scale" : sdt.IdLink() }
	event       = ns.Event( "HW03" , 2003 , None , time_reference , name_variable = "T" , unit_variable = "K" )
	verbose     = True
	ci          = 0.05
	
	
	## Load models and observations
	##=============================
	models,lX,lY,Xo,Yo = load_models_obs( pathInp )
	
	
	## Anomaly from observations
	##==========================
	Yo -= Yo.loc[event.reference].mean()
	event.anomaly = float(Yo.loc[event.time])
	
	
	## Models in anomaly
	##==================
	if is_test:
		models = models[:5]
		lX = lX[:5]
		lY = lY[:5]
	
	for X in lX:
		X -= X.loc[event.reference].mean()
	for Y in lY:
		Y -= Y.loc[event.reference].mean()
	
	
	## Define clim variable from input
	##================================
	clim = ns.Climatology( time_period , n_sample , models , ns_law , ns_law_args )
	
	
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
	climCX   = ns.constraints_CX( climMM , Xo , time_reference = time_reference , verbose = verbose )
	climCXCB = ns.constraints_bayesian( climCX , Yo , 1000 , 500 , verbose = verbose )
	climC0   = ns.constraints_C0( climMM , Yo , verbose = verbose )
	climCXC0 = ns.constraints_C0( climCX , Yo , verbose = verbose )
	
	
	## Compute stats
	##==============
	climMM   = ns.extremes_stats( climMM   , event , verbose = verbose )
	climCX   = ns.extremes_stats( climCX   , event , verbose = verbose )
	climCXCB = ns.extremes_stats( climCXCB , event , verbose = verbose )
	climC0   = ns.extremes_stats( climC0   , event , verbose = verbose )
	climCXC0 = ns.extremes_stats( climCXC0 , event , verbose = verbose )
	
	
	## Save in netcdf
	##===============
	ns.to_netcdf( climMM   , event , os.path.join( pathOut , "HW03_Normal_clim.nc"     ) , ""     )
	ns.to_netcdf( climCX   , event , os.path.join( pathOut , "HW03_Normal_climCX.nc"   ) , "CX"   )
	ns.to_netcdf( climCXCB , event , os.path.join( pathOut , "HW03_Normal_climCXCB.nc" ) , "CX"   )
	ns.to_netcdf( climC0   , event , os.path.join( pathOut , "HW03_Normal_climC0.nc"   ) , "C0"   )
	ns.to_netcdf( climCXC0 , event , os.path.join( pathOut , "HW03_Normal_climCXC0.nc" ) , "CXC0" )
	
#	climCXC0,event = ns.netcdf2clim( os.path.join( pathOut , "HW03_Normal_climCXC0.nc" ) , ns_law , ns_law_args )
	
	
	## Write stats in txt file
	##========================
	with open( os.path.join( pathOut , "SummaryCXCB.txt" ) , "w" ) as f:
		f.write( str(event) + "\n\n" )
		f.write( nsp.print_time_stats( climCXCB.stats , 2003 , model = "multi" , digit = 6 , ci = ci , verbose = verbose ) + "\n" )
		f.write( nsp.print_time_stats( climCXCB.stats , 2040 , model = "multi" , digit = 6 , ci = ci , verbose = verbose ) + "\n" )
		f.write( nsp.print_relative_time_stats( climCXCB.stats , 2040 , 2003 , model = "multi" , digit = 6 , ci = ci , verbose = verbose ) + "\n" )
	
	
	## Plot
	##=====
	nsp.decomposition( lX , clim.X , event                      , os.path.join( pathOut , "decomposition.pdf" ) , verbose = verbose )
	nsp.constraints_CX( climMM , climCXC0 , Xo , time_reference , os.path.join( pathOut , "constraintCX.pdf" )  , verbose = verbose )
	nsp.plot_classic_packages( climMM   , event , path = pathOut , suffix = "MM"   , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCX   , event , path = pathOut , suffix = "CX"   , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCXCB , event , path = pathOut , suffix = "CXCB" , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climC0   , event , path = pathOut , suffix = "C0"   , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCXC0 , event , path = pathOut , suffix = "CXC0" , ci = ci , verbose = verbose )
	nsp.ns_params_comparison( climMM , climCXC0   , ofile = os.path.join( pathOut , "ns_paramsMM_CXC0.pdf") , ci = ci , verbose = verbose )
	nsp.ns_params_comparison( climMM , climCXCB   , ofile = os.path.join( pathOut , "ns_paramsMM_CXCB.pdf") , ci = ci , verbose = verbose )
	
	print("Done")


