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
#mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

import numpy as np
import scipy.stats as sc
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
	
	## Define dataset
	##===============
	t,X,_,_ = sdt.Dataset.covariates(2000)
	
	loc   =   1 - 2   * X
	scale = 0.5 + 0.5 * X
	shape = np.repeat( -0.2 , t.size )
	
	Y = - sc.genextreme.rvs( loc = loc , scale = scale , c = -shape )
	
	
	## 
	mod = nsm.GEVMin( l_scale = sdt.IdLink() )
	mod.fit( Y , X )
	mod.set_covariable( X , t )
	ub = mod.upper_boundt(t)
	lb = mod.lower_boundt(t)
	Y2 = mod.rvs(t)
	print(mod.get_params())
	
	## Plot
	##=====
	nrow,ncol = 2,2
	fig = plt.figure( figsize = (12,8) )
	
	ax = fig.add_subplot( nrow , ncol , 1 )
	ax.plot( t , X  , color = "red" )
	ax.plot( t , Y  , color = "blue"  , linestyle = "" , marker = "." )
	ax.plot( t , Y2 , color = "green" , linestyle = "" , marker = "." )
	ax.plot( t , mod.loct(t) , color = "black" )
	ax.plot( t , ub , color = "black" )
	ax.plot( t , lb , color = "black" )
	
	ax = fig.add_subplot( nrow , ncol , 2 )
	cdf_lev = np.zeros_like(t)
	ax.plot( t , mod.cdf(cdf_lev,t) , color = "red"  , label = "cdf" )
	ax.plot( t , mod.sf(cdf_lev,t)  , color = "blue" , label = "sf"  )
	ax.legend()
	
	ax = fig.add_subplot( nrow , ncol , 3 )
	icdf_lev = np.zeros_like(t) + 0.2
	ax.plot( t , mod.icdf(icdf_lev,t) , color = "red"  , label = "icdf" )
	ax.plot( t , mod.isf(icdf_lev,t)  , color = "blue" , label = "isf"  )
	ax.legend()
	
	fig.set_tight_layout(True)
	
	plt.show()
	
	print("Done")
