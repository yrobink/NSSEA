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
#import matplotlib.backends.backend_pdf as mpdf

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as sc

import SDFC as sd
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
	
	## Data
	##=====
	t,X,_,_ = sdt.Dataset.covariates(2000)
	loc = -1 + 2 * X
	scale = 0.5 + X / 2
	Y = sc.norm.rvs( loc = loc , scale = scale )
	
	## Model
	##======
	mod = nsm.N_Normal()
	mod.fit( Y , X )
	mod.set_covariable( X , t )
	eloc   = mod.loct(t)
	escale = mod.scalet(t)
	
	cdf = mod.cdf( eloc + escale , t )
	
	## Plot
	##=====
	fig = plt.figure( figsize = (16,6) )
	ax = fig.add_subplot( 1 , 2 , 1 )
	ax.plot( t , Y , color = "blue" , linestyle = "" , marker = "." )
	
	ax.plot( t , loc , color = "green" , linestyle = "-" )
	ax.plot( t , loc + scale , color = "green" , linestyle = "--" )
	ax.plot( t , loc - scale , color = "green" , linestyle = "--" )
	
	ax.plot( t , eloc , color = "red" , linestyle = "-" )
	ax.plot( t , eloc + escale , color = "red" , linestyle = "--" )
	ax.plot( t , eloc - escale , color = "red" , linestyle = "--" )
	
	ax = fig.add_subplot( 1 , 2 , 2 )
	ax.plot( t , cdf , color = "blue" )
	
	fig.set_tight_layout(True)
	plt.show()
	
	print("Done")

