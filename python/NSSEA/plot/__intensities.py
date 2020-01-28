
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

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from .__linkParams import LinkParams


###############
## Functions ##
###############

def intensities( clim , event , ofile , ci = 0.05 , verbose = False ): ##{{{
	"""
	NSSEA.plot.intensities
	======================
	
	Plot intensities (IF,IC,dI) along time
	
	Arguments
	---------
	clim      : NSSEA.Climatology
		Climatology with stats computed
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot intensities" , end = "\r" )
	
	stats = clim.stats
	statsl = stats[:,1:,3:,:].quantile( ci / 2.      , dim = "sample" )
	statsu = stats[:,1:,3:,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	pdf = mpdf.PdfPages( ofile )
	
	yminI  = min( stats[:,:,3:5,:].min() , statsu[:,:2,:].min() , statsl[:,:2,:].min() )
	ymaxI  = max( stats[:,:,3:5,:].max() , statsu[:,:2,:].max() , statsl[:,:2,:].max() )
	ymindI = min( stats[:,:,5,:].min()   , statsu[:,2,:].min()  , statsl[:,2,:].min()  )
	ymaxdI = max( stats[:,:,5,:].max()   , statsu[:,2,:].max()  , statsl[:,2,:].max()  )
	
	ylabel = "\mathrm{(" + event.unit_variable + ")}"
	
	for m in stats.models:
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( stats.time , stats.loc[:,"be","IF",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"IF",m] , statsu.loc[:,"IF",m] , color = "red" , alpha = 0.5 )
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_ylim( (yminI,ymaxI) )
		ax.set_xticks([])
		ax.set_ylabel( r"${}$".format( "\mathbf{I}_F(t)\ " + ylabel ) )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.hlines( stats.loc[event.time,"be","IF",m] , xlim[0] , xlim[1] , color = "black" , linestyle = "--" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( stats.time , stats.loc[:,"be","IC",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"IC",m] , statsu.loc[:,"IC",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (yminI,ymaxI) )
		ax.set_xticks([])
		ax.set_ylabel( r"${}$".format( "\mathbf{I}_C(t)\ " + ylabel ) )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.hlines( stats.loc[event.time,"be","IC",m] , xlim[0] , xlim[-1] , color = "black" , linestyle = "--" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		ax = fig.add_subplot( nrow , ncol , 3 )
		ax.plot( stats.time , stats.loc[:,"be","dI",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"dI",m] , statsu.loc[:,"dI",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (ymindI,ymaxdI) )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( r"${}$".format( "\Delta\mathbf{I}(t)\ " + ylabel ) )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim  , linestyle = "--" , marker = "" , color = "black" )
		ax.plot( xlim                    , [0,0] , linestyle = "-"  , marker = "" , color = "black" )
		ax.hlines( stats.loc[event.time,"be","dI",m] , xlim[0] , xlim[-1] , color = "black" , linestyle = "--" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose: print( "Plot intensities (Done)" )
##}}}


