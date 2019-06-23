
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import numpy  as np
import pandas as pd
import xarray as xr

try:
	import matplotlib.pyplot as plt
except:
	import matplotlib as mpl
	mpl.use("Qt5Agg")
	import matplotlib.pyplot as plt

import matplotlib.backends.backend_pdf as mpdf

from NSSEA.plot.__linkParams import LinkParams


###############
## Functions ##
###############


def constraints_CX( coffee , coffeeCX , Xo , cx_params , ofile , ci = 0.05 , verbose = False ):
	"""
	NSSEA.plot.constraints_CX
	=========================
	
	Plot effect of constraints CX
	
	Arguments
	---------
	coffee    : NSSEA.Coffee
		coffee variable without constraints
	coffeeCX  : NSSEA.Coffee
		coffee variable with constraints
	cx_params : NSSEA.CXParams
		Constraints parameters
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot constraintsCX" , end = "\r" )
	
	X   = coffee.X.loc[:,:,"all","multi"]
	cX = coffeeCX.X.loc[:,:,"all","multi"]
	
	X  = X  - X.loc[cx_params.ref,:].mean( dim = "time" )
	cX = cX - cX.loc[cx_params.ref,:].mean( dim = "time" )
	Xo = Xo - Xo.loc[cx_params.ref].mean()
	
	Xl  = X[:,1:].quantile(  ci / 2.      , dim = "sample" )
	Xu  = X[:,1:].quantile(  1. - ci / 2. , dim = "sample" )
	cXl = cX[:,1:].quantile( ci / 2.      , dim = "sample" )
	cXu = cX[:,1:].quantile( 1. - ci / 2. , dim = "sample" )
	
	fig = plt.figure( figsize = (10,10) )
	
	ax = fig.add_subplot(1,1,1)
	ax.plot( X.time  , X[:,0]     , color = "grey"  , linestyle = "-" , marker = ""  )
	ax.plot( cX.time , cX[:,0]    , color = "black" , linestyle = "-" , marker = ""  )
	ax.plot( Xo.index , Xo.values , color = "black" , linestyle = ""  , marker=  "." )
	
	ax.fill_between( X.time  , Xl  , Xu  , color = "red" , alpha = 0.2 )
	ax.fill_between( cX.time , cXl , cXu , color = "red" , alpha = 0.5 )
	
	plt.tight_layout()
	plt.savefig( ofile )
	
	if verbose: print( "Plot constraints CX (Done)" )

