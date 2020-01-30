#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import os
import numpy  as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from .__linkParams    import LinkParams
from .__probabilities import probabilities
from .__ns_params     import ns_params
from .__intensities   import intensities
from .__stats_event   import stats_event
from .__stats_event   import stats_relative


###############
## Functions ##
###############


def plot_classic_packages( clim , event , path , suffix = "" , be_is_median = False , ci = 0.05 , verbose = False ):
	"""
	NSSEA.plot.plot_classic_packages
	================================
	
	Just a function which call:
		- NSSEA.plot.probabilities
		- NSSEA.plot.ns_params
		- NSSEA.plot.intensities
		- NSSEA.plot.stats_event
		- NSSEA.plot.stats_relative
	
	
	Arguments
	---------
	clim    : NSSEA.Climatology
		A clim variable
	event     : NSSEA.Event
		Event variable
	suffix    : str
		suffix for the name of files
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot classic_packages ({}): 0/5".format(suffix) , end = "\r" )
	probabilities(  clim , event , ofile = os.path.join( path , "Probabilities" + suffix + ".pdf"                       ) , ci = ci , be_is_median = be_is_median , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 1/5".format(suffix) , end = "\r" )
	intensities(    clim , event , ofile = os.path.join( path , "Intensities"   + suffix + ".pdf"                       ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 2/5".format(suffix) , end = "\r" )
	stats_event(    clim , event , ofile = os.path.join( path , "StatsEvent"    + suffix + "_{}.pdf".format(event.time) ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 3/5".format(suffix) , end = "\r" )
	stats_relative( clim , event , ofile = os.path.join( path , "StatsRelative" + suffix + "_{}.pdf".format(event.time) ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 4/5".format(suffix) , end = "\r" )
	ns_params(      clim ,         ofile = os.path.join( path , "ns_params"     + suffix + ".pdf"                       ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 5/5".format(suffix) , end = "\n" )
	

