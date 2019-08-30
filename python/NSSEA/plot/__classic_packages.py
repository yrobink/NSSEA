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

from NSSEA.plot.__linkParams    import LinkParams
from NSSEA.plot.__probabilities import probabilities
from NSSEA.plot.__ns_params     import ns_params
from NSSEA.plot.__intensities   import intensities
from NSSEA.plot.__stats_event   import stats_event
from NSSEA.plot.__stats_event   import stats_relative


###############
## Functions ##
###############


def plot_classic_packages( clim , event , suffix , ci = 0.05 , verbose = False ):
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
	probabilities(  clim.stats , event              , ofile = os.path.join( event.dir_name , "Probability"   + suffix + ".pdf"                       ) , ci = ci , verbose = verbose )
	ns_params(      clim                            , ofile = os.path.join( event.dir_name , "ns_params"     + suffix + ".pdf"                       ) , ci = ci , verbose = verbose )
	intensities(    clim.stats , event              , ofile = os.path.join( event.dir_name , "Intensity"     + suffix + ".pdf"                       ) , ci = ci , verbose = verbose )
	stats_event(    clim       , event.time , event , ofile = os.path.join( event.dir_name , "StatsEvent"    + suffix + "_{}.pdf".format(event.time) ) , ci = ci , verbose = verbose )
	stats_relative( clim.stats , event              , ofile = os.path.join( event.dir_name , "StatsRelative" + suffix + ".pdf"                       ) , ci = ci , verbose = verbose )


