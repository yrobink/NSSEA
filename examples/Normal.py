# -*- coding: utf-8 -*-

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2020                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## This software is a computer program that is part of the NSSEA                ##
## (Non-Stationary Statistics for Extreme Attribution) This library makes it    ##
## possible to infer the probability of an (extreme) event in the factual /     ##
## counter-factual world (without anthropic forcing) to attribute it to climate ##
## change.                                                                      ##
##                                                                              ##
## This software is governed by the CeCILL-C license under French law and       ##
## abiding by the rules of distribution of free software.  You can  use,        ##
## modify and/ or redistribute the software under the terms of the CeCILL-C     ##
## license as circulated by CEA, CNRS and INRIA at the following URL            ##
## "http://www.cecill.info".                                                    ##
##                                                                              ##
## As a counterpart to the access to the source code and  rights to copy,       ##
## modify and redistribute granted by the license, users are provided only      ##
## with a limited warranty  and the software's author,  the holder of the       ##
## economic rights,  and the successive licensors  have only  limited           ##
## liability.                                                                   ##
##                                                                              ##
## In this respect, the user's attention is drawn to the risks associated       ##
## with loading,  using,  modifying and/or developing or reproducing the        ##
## software by the user in light of its specific status of free software,       ##
## that may mean  that it is complicated to manipulate,  and  that  also        ##
## therefore means  that it is reserved for developers  and  experienced        ##
## professionals having in-depth computer knowledge. Users are therefore        ##
## encouraged to load and test the software's suitability as regards their      ##
## requirements in conditions enabling the security of their systems and/or     ##
## data to be ensured and,  more generally, to use and operate it in the        ##
## same conditions as regards security.                                         ##
##                                                                              ##
## The fact that you are presently reading this means that you have had         ##
## knowledge of the CeCILL-C license and that you accept its terms.             ##
##                                                                              ##
##################################################################################
##################################################################################

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2020                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## Ce logiciel est un programme informatique faisant partie de la librairie     ##
## NSSEA (Non-Stationary Statistics for Extreme Attribution). Cette librairie   ##
## permet d'estimer la probabilité d'un evenement (extreme) dans le monde       ##
## factuel / contre factuel (sans forcage anthropogenique) et de l'attribuer au ##
## changement climatique.                                                       ##
##                                                                              ##
## Ce logiciel est régi par la licence CeCILL-C soumise au droit français et    ##
## respectant les principes de diffusion des logiciels libres. Vous pouvez      ##
## utiliser, modifier et/ou redistribuer ce programme sous les conditions       ##
## de la licence CeCILL-C telle que diffusée par le CEA, le CNRS et l'INRIA     ##
## sur le site "http://www.cecill.info".                                        ##
##                                                                              ##
## En contrepartie de l'accessibilité au code source et des droits de copie,    ##
## de modification et de redistribution accordés par cette licence, il n'est    ##
## offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,    ##
## seule une responsabilité restreinte pèse sur l'auteur du programme, le       ##
## titulaire des droits patrimoniaux et les concédants successifs.              ##
##                                                                              ##
## A cet égard  l'attention de l'utilisateur est attirée sur les risques        ##
## associés au chargement,  à l'utilisation,  à la modification et/ou au        ##
## développement et à la reproduction du logiciel par l'utilisateur étant       ##
## donné sa spécificité de logiciel libre, qui peut le rendre complexe à        ##
## manipuler et qui le réserve donc à des développeurs et des professionnels    ##
## avertis possédant  des  connaissances  informatiques approfondies.  Les      ##
## utilisateurs sont donc invités à charger  et  tester  l'adéquation  du       ##
## logiciel à leurs besoins dans des conditions permettant d'assurer la         ##
## sécurité de leurs systèmes et ou de leurs données et, plus généralement,     ##
## à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.           ##
##                                                                              ##
## Le fait que vous puissiez accéder à cet en-tête signifie que vous avez       ##
## pris connaissance de la licence CeCILL-C, et que vous en avez accepté les    ##
## termes.                                                                      ##
##                                                                              ##
##################################################################################
##################################################################################

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
	n_mcmc_drawn_min = 2500 if is_test else  5000
	n_mcmc_drawn_max = 5000 if is_test else 10000
	min_rate_accept  = 0.05
	n_sample    = 1000 if not is_test else 10
	ns_law      = nsm.Normal( l_scale = sdt.ExpLink() )
	event       = ns.Event( "HW03" , 2003 , None , time_reference , name_variable = "T" , unit_variable = "K" )
	verbose     = True
	ci          = 0.05 if not is_test else 0.1
	
	
	## Load models and observations
	##=============================
	models,lX,lY,Xo,Yo = load_models_obs( pathInp )
	
	
	## Anomaly from observations
	##==========================
	Yo -= Yo.loc[event.reference].mean()
	event.anomaly = float(Yo.loc[event.time])
	
	
	## Models in anomaly
	##==================
	
	for X in lX:
		X -= X.loc[event.reference].mean()
	for Y in lY:
		Y -= Y.loc[event.reference].mean()
	
	
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
	clim = ns.infer_multi_model( clim , verbose = verbose )
	
	
	## Keep only multi-model
	##======================
	climMM = clim.copy()
	climMM.keep_models( ["multi"] )
	
	
	## Apply constraints
	##==================
	climCX     = ns.constraints_CX( climMM , Xo , time_reference = time_reference , verbose = verbose )
	climCXCB   = ns.constraints_bayesian( climCX , Yo , n_mcmc_drawn_min , n_mcmc_drawn_max , min_rate_accept = min_rate_accept , verbose = verbose )
	climC0     = ns.constraints_C0( climMM , Yo , verbose = verbose )
	climCXC0   = ns.constraints_C0( climCX , Yo , verbose = verbose )
	
	
	## Compute stats
	##==============
	clim       = ns.extremes_stats( clim     , event , verbose = verbose )
	climCX     = ns.extremes_stats( climCX   , event , verbose = verbose )
	climCXCB   = ns.extremes_stats( climCXCB , event , verbose = verbose )
	climC0     = ns.extremes_stats( climC0   , event , verbose = verbose )
	climCXC0   = ns.extremes_stats( climCXC0 , event , verbose = verbose )
	
	
	## Save in netcdf
	##===============
	ns.to_netcdf( clim     , event , os.path.join( pathOut , "HW03_Normal_clim.nc"       ) , ""       )
	ns.to_netcdf( climCX   , event , os.path.join( pathOut , "HW03_Normal_climCX.nc"     ) , "CX"     )
	ns.to_netcdf( climCXCB , event , os.path.join( pathOut , "HW03_Normal_climCXCB.nc"   ) , "CXCB"   )
	ns.to_netcdf( climC0   , event , os.path.join( pathOut , "HW03_Normal_climC0.nc"     ) , "C0"     )
	ns.to_netcdf( climCXC0 , event , os.path.join( pathOut , "HW03_Normal_climCXC0.nc"   ) , "CXC0"   )
	
	
	## Plot
	##=====
	nsp.write_package_tabular( climCXCB , event , os.path.join( pathOut , "SummaryCXCB.txt"   ) , verbose = verbose )
	nsp.write_package_tabular( climCXC0 , event , os.path.join( pathOut , "SummaryCXC0.txt"   ) , verbose = verbose )
	nsp.decomposition( lX , clim.X , event                      , os.path.join( pathOut , "decomposition.pdf" ) , verbose = verbose )
	nsp.constraints_CX( climMM , climCXCB , Xo , time_reference , os.path.join( pathOut , "constraintCX.pdf" )  , verbose = verbose )
	nsp.plot_classic_packages( clim     , event , path = pathOut , suffix = "MM"     , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCX   , event , path = pathOut , suffix = "CX"     , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCXCB , event , path = pathOut , suffix = "CXCB"   , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climC0   , event , path = pathOut , suffix = "C0"     , ci = ci , verbose = verbose )
	nsp.plot_classic_packages( climCXC0 , event , path = pathOut , suffix = "CXC0"   , ci = ci , verbose = verbose )
	nsp.ns_params_comparison( climMM , climCXC0   , ofile = os.path.join( pathOut , "ns_paramsMM_CXC0.pdf"   ) , ci = ci , verbose = verbose )
	nsp.ns_params_comparison( climMM , climCXCB   , ofile = os.path.join( pathOut , "ns_paramsMM_CXCB.pdf"   ) , ci = ci , verbose = verbose )
	
	
	print("Done")


