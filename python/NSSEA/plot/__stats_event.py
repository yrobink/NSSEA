
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
from NSSEA.__nsstats         import stats_relative_event


###############
## Functions ##
###############

def stats_event( coffee , event_time , ofile , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.stats_event
	======================
	
	Plot statistics at a time
	
	Arguments
	---------
	coffee    : NSSEA.Coffee
		A coffee variable
	event_time: time
		time to plot
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot stats event" , end = "\r" )
	
	## Extract parameters
	statsEvent = coffee.stats.loc[event_time,:,:,:]
	n_models   = statsEvent.models.size
	
	## Figure + matplotlib parameters
	fig = plt.figure( figsize = (25,30) )
	lp = LinkParams()
	whis = [ 100 * ci / 2. , 100 * (1. - ci / 2.) ]
	boxprops = dict( facecolor = "red" , color = "red" )
	medianprops = dict( color = "black" )
	widths = 0.8
	
	
	## Probabilities plot
	ax = fig.add_subplot(3,2,1)
	for i,m in enumerate(statsEvent.models):
		ax.boxplot( lp.fp(statsEvent.loc[:,"pall",m]) , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops , widths = widths )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.p.values )
	ax.set_xticklabels( lp.p.names )
	ax.set_title( "pall" )
	
	ax = fig.add_subplot(3,2,2)
	for i,m in enumerate(statsEvent.models):
		ax.boxplot( lp.fp(statsEvent.loc[:,"pnat",m]) , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops  , widths = widths )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.p.values )
	ax.set_xticklabels( lp.p.names )
	ax.set_title( "pnat" )
	
	ax = fig.add_subplot(3,2,3)
	for i,m in enumerate(statsEvent.models):
		ax.boxplot( lp.frr(statsEvent.loc[:,"rr",m]) , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops  , widths = widths )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.rr.values )
	ax.set_xticklabels( lp.rr.names )
	ax.set_title( "RR" )
	
	
	## Limits of intensities
	xminI  = float( statsEvent[:,3:5,:].min() )
	xmaxI  = float( statsEvent[:,3:5,:].max() )
	delta  = (xmaxI - xminI) / 10.
	xminI -= delta
	xmaxI += delta
	xmindI = float( statsEvent[:,5,:].min()   )
	xmaxdI = float( statsEvent[:,5,:].max()   )
	delta  = (xmaxdI - xmindI) / 10.
	xmindI -= delta
	xmaxdI += delta
	
	
	## Intensities plot
	ax = fig.add_subplot(3,2,4)
	for i,m in enumerate(statsEvent.models):
		ax.boxplot( statsEvent.loc[:,"iall",m].values.ravel() , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops  , widths = widths )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xlim( (xminI,xmaxI) )
	ax.set_title( "iall" )

	ax = fig.add_subplot(3,2,5)
	for i,m in enumerate(statsEvent.models):
		ax.boxplot( statsEvent.loc[:,"inat",m].values.ravel() , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops  , widths = widths )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xlim( (xminI,xmaxI) )
	ax.set_title( "inat" )
	
	ax = fig.add_subplot(3,2,6)
	for i,m in enumerate(statsEvent.models):
		ax.boxplot( statsEvent.loc[:,"di",m].values.ravel() , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops  , widths = widths )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xlim( (xmindI,xmaxdI) )
	ax.set_title( "di" )
	
	
	fig.tight_layout()
	fig.savefig( ofile )
	plt.close(fig)
	
	if verbose: print( "Plot stats event (Done)" )
##}}}

def stats_relative( statsIn , event , ofile , time_event = None , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.stats_relative
	=========================
	
	Plot probabilities rr/rr[time_event] and di - di[time_event] along time
	
	Arguments
	---------
	stats     : xarray
		NSSEA.Coffee.stats
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot stats_relative" , end = "\r" )
	
	## Compute stats events
	if time_event is None:
		time_event = event.time
	stats = stats_relative_event( statsIn , time_event )
	statsu = stats[:,1:,:,:].quantile( ci / 2.      , dim = "sample" )
	statsl = stats[:,1:,:,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	ymindI = min( stats.loc[:,:,"di",:].min()   , statsu.loc[:,"di",:].min()  , statsl.loc[:,"di",:].min()  )
	ymaxdI = max( stats.loc[:,:,"di",:].max()   , statsu.loc[:,"di",:].max()  , statsl.loc[:,"di",:].max()  )
	ylabel = "\mathrm{(" + event.unit + ")}"
	
	lp = LinkParams()
	
	pdf = mpdf.PdfPages( ofile )
	
	for m in stats.models:
		nrow,ncol = 2,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.6 * fs * nrow ) )
		
		## Probabilities
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( stats.time , lp.frr(stats.loc[:,"be","rr",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.frr(statsl.loc[:,"rr",m]) , lp.frr(statsu.loc[:,"rr",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.rr.values.min(),lp.rr.values.max()) )
		ax.set_yticks( lp.rr.values )
		ax.set_yticklabels( lp.rr.names )
		ax.set_xlabel( r"$\mathrm{Time}$" )
		ax.set_ylabel( r"$\mathrm{RR}(t)$" )
		ax2 = fig.add_subplot( nrow , ncol , 1 , sharex = ax , frameon = False )
		ax2.yaxis.tick_right()
		ax2.set_yticks( lp.rr.values )
		ax2.set_yticklabels( lp.far.names )
		ax2.yaxis.set_label_position( "right" )
		ax2.set_ylabel( r"$\mathrm{FAR}(t)$" )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( xlim                    , lp.frr([1,1]) , linestyle = "-"  , marker = "" , color = "black" )
		ax.plot( [time_event,time_event] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		## Intensities
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( stats.time , stats.loc[:,"be","di",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"di",m] , statsu.loc[:,"di",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (ymindI,ymaxdI) )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( r"${}$".format( "\delta\mathbf{i}(t)\ " + ylabel ) )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [time_event,time_event] , ylim  , linestyle = "--" , marker = "" , color = "black" )
		ax.plot( xlim                    , [0,0] , linestyle = "-"  , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		fig.tight_layout()
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose: print( "Plot stats_relative (Done)" )
##}}}

