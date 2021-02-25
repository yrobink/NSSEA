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
import tarfile
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import SDFC.link as sdl
import NSSEA as ns
import NSSEA.plot as nsp
import NSSEA.models as nsm


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
	with tarfile.open( os.path.join( path , "GEV.tar.gz" ) , "r" ) as tf:
		tf.extractall( path )
	
	## List of models X
	modelsX = [ "_".join(f.split("/")[-1][:-3].split("_")[-3:-1]) for f in os.listdir(os.path.join(pathInp,"GEV/X")) ]
	modelsX.sort()
	
	## List of models Y
	modelsY = [ "_".join(f.split("/")[-1][:-3].split("_")[-3:-1]) for f in os.listdir(os.path.join(pathInp,"GEV/Y")) ]
	modelsY.sort()
	
	## Merge the two lists to keep only common models
	modelsX.sort()
	modelsY.sort()
	models = list(set(modelsX) & set(modelsY))
	models.sort()
	
	## Load X and Y
	lX = []
	lY = []
	for m in models:
		
		## Load X
		df   = xr.open_dataset( os.path.join( pathInp , "GEV/X/tas_mon_historical-rcp85_{}_1850-2099.nc".format(m) ) )
		time = df.time["time.year"].values
		X    = pd.DataFrame( df.tas.values.ravel() , columns = [m] , index = time )
		lX.append( correct_miss(X) )
		
		## Load Y
		df   = xr.open_dataset( os.path.join( pathInp , "GEV/Y/tas_day_historical-rcp85_{}_1850-2099.nc".format(m) ) )
		time = df.time["time.year"].values
		Y    = pd.DataFrame( df.tas.values.ravel() , columns = [m] , index = time )
		lY.append( correct_miss(Y) )
	
	## Load Xo
	dXo = xr.open_dataset("input/GEV/Xo.nc")
	Xo  = pd.DataFrame( dXo.tas_mean.values.squeeze() , columns = ["Xo"] , index = dXo.time["time.year"].values )
	
	dYo = xr.open_dataset("input/GEV/Yo.nc")
	Yo  = pd.DataFrame( dYo.TM.values.squeeze() , columns = ["Yo"] , index = dYo.time["time.year"].values )
	
	
	for f in os.listdir( os.path.join( pathInp , "GEV/X" ) ):
		os.remove( os.path.join( pathInp , "GEV/X/{}".format(f) ) )
	for f in os.listdir( os.path.join( pathInp , "GEV/Y" ) ):
		os.remove( os.path.join( pathInp , "GEV/Y/{}".format(f) ) )
	os.remove( os.path.join( pathInp , "GEV/Xo.nc" ) )
	os.remove( os.path.join( pathInp , "GEV/Yo.nc" ) )
	os.rmdir( os.path.join( pathInp , "GEV/X" ) )
	os.rmdir( os.path.join( pathInp , "GEV/Y" ) )
	os.rmdir( os.path.join( pathInp , "GEV" ) )
	
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
	is_test  = "--not-test" not in sys.argv
	set_seed = ~("--no-seed" in sys.argv)
	if set_seed: np.random.seed(42) ## A good seed is always the big answer
	
	## Set a log class, the GEV produces sometimes overflow
	##=====================================================
	nplog = NumpyLog()
	np.seterrcall(nplog)
	np.seterr( all = "log" , invalid = "log" )
	warnings.simplefilter("ignore")
	
	## Law
	##====
	if "--Normal" in sys.argv:
		law = "Normal"
		name = "HW03"
		ns_law  = nsm.Normal()
	elif "--GEV" in sys.argv:
		law = "GEV"
		name = "HW19"
		ns_law  = nsm.GEV()
	else:
		assert(False)
	
	## Path
	##=====
	basepath = os.path.dirname(os.path.abspath(__file__))
	pathInp  = os.path.join( basepath , "input"  )
	pathOut  = os.path.join( basepath , "output/{}".format(law) )
	assert(os.path.exists(pathInp))
	assert(os.path.exists(pathOut))
	
	## Some global parameters
	##=======================
	verbose = "--not-verbose" not in sys.argv
	ci      = 0.05 if not is_test else 0.1
	
	## Compute stats
	##==============
	climCXCB = ns.Climatology.from_netcdf( os.path.join( pathOut , "{}_climCXCB.nc".format(name) ) , ns_law )
	bias = { "Multi_Synthesis" : climCXCB.statistics.loc[1961:1991,"BE","IF","Multi_Synthesis"].mean() }
	climCXCB.event.type  = "Rt"
	for value in [2,10,20,30,50,100,1000]:
		climCXCB.event.value = value
		climCXCB = ns.statistics_fixed_pF( climCXCB , verbose = verbose )
		climCXCB = ns.add_bias( climCXCB , bias , verbose = verbose )
		nsp.intensities( climCXCB , os.path.join( pathOut , "Intensities_CXCB_Rt_{}.pdf".format(climCXCB.event.value) ) , ci = ci , verbose = verbose )
		climCXCB.statistics.to_dataset( name = "statistics_Rt{}".format(value) ).to_netcdf( os.path.join( pathOut , "statistics_Rt{}.nc".format(value) ) )
	
	print("Done")


