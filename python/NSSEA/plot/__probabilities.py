
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

from NSSEA.plot.__linkParams import LinkParams


###############
## Functions ##
###############

def probabilities( stats , event , ofile , ci = 0.05 , verbose = False ): ##{{{
	"""
	NSSEA.plot.probabilities
	========================
	
	Plot probabilities (pall,pnat,rr) along time
	
	Arguments
	---------
	stats     : xarray
		NSSEA.Climatology.stats
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot probabilities" , end = "\r" )
	
	statsl = stats[:,1:,:3,:].quantile( ci / 2.      , dim = "sample" )
	statsu = stats[:,1:,:3,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	lp = LinkParams()
	
	pdf = mpdf.PdfPages( ofile )
	
	for m in stats.models:
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( stats.time , lp.fp(stats.loc[:,"be","pall",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.fp(statsl.loc[:,"pall",m]) , lp.fp(statsu.loc[:,"pall",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_xticks([])
		ax.set_ylabel( r"$p_1(t)$" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( stats.time , lp.fp(stats.loc[:,"be","pnat",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.fp(statsl.loc[:,"pnat",m]) , lp.fp(statsu.loc[:,"pnat",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_xticks([])
		ax.set_ylabel( r"$p_0(t)$" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		ax = fig.add_subplot( nrow , ncol , 3 )
		ax.plot( stats.time , lp.frr(stats.loc[:,"be","rr",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.frr(statsl.loc[:,"rr",m]) , lp.frr(statsu.loc[:,"rr",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.rr.values.min(),lp.rr.values.max()) )
		ax.set_yticks( lp.rr.values )
		ax.set_yticklabels( lp.rr.names )
		ax.set_xlabel( r"$\mathrm{Time}$" )
		ax.set_ylabel( r"$\mathrm{RR}(t)$" )
		ax2 = fig.add_subplot( nrow , ncol , 3 , sharex = ax , frameon = False )
		ax2.yaxis.tick_right()
		ax2.set_yticks( lp.rr.values )
		ax2.set_yticklabels( lp.far.names )
		ax2.yaxis.set_label_position( "right" )
		ax2.set_ylabel( r"$\mathrm{FAR}(t)$" )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( xlim                    , lp.frr([1,1]) , linestyle = "-"  , marker = "" , color = "black" )
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose: print( "Plot probabilities (Done)" )
##}}}

