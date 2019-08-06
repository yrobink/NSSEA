
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
from NSSEA.__nsstats         import stats_relative_event


###############
## Functions ##
###############


def stats_event( coffee , event_time , event , ofile , ci = 0.05 , verbose = False ):##{{{
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
	Sq = statsEvent[1:,:,:].quantile( [ci/2,1-ci/2] , dim = "sample" ).assign_coords( quantile = ["q0","q1"] )
	n_models   = statsEvent.models.size
	
	
	## Scale
	rr_min = float(min(Sq.loc[:,"rr",:].min() , statsEvent.loc["be","rr",:].min()))
	rr_max = float(max(Sq.loc[:,"rr",:].max() , statsEvent.loc["be","rr",:].max()))
	p_min = float(min(Sq.loc[:,["pnat","pall"],:].min() , statsEvent.loc["be",["pnat","pall"],:].min()))
	p_max = float(max(Sq.loc[:,["pnat","pall"],:].max() , statsEvent.loc["be",["pnat","pall"],:].max()))
	lp = LinkParams( rr_min = rr_min , rr_max = rr_max , p_min = p_min , p_max = p_max )
	
	
	## Figure + matplotlib parameters
	nrow,ncol = 2,3
	fig = plt.figure( figsize = (30,25) )
	whis = [ 100 * ci / 2. , 100 * (1. - ci / 2.) ]
	boxprops = dict( facecolor = "red" , color = "red" )
	medianprops = dict( color = "black" )
	widths = 0.8
	fontsize = 20
	
	## Probabilities plot
	ax = fig.add_subplot(nrow,ncol,1)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( lp.fp( Sq.loc[:,"pall",m] ) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( lp.fp(statsEvent.loc["be","pall",m]) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in lp.p.values:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.p.values )
	ax.set_xticklabels( lp.p.names , fontsize = fontsize )
	ax.set_xlabel( r"$p_1$" + "({})".format(event_time) , fontsize = fontsize )
	
	ax = fig.add_subplot(nrow,ncol,2)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( lp.fp( Sq.loc[:,"pnat",m] ) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( lp.fp(statsEvent.loc["be","pnat",m]) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in lp.p.values:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
#	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_yticklabels([])
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.p.values )
	ax.set_xticklabels( lp.p.names , fontsize = fontsize )
	ax.set_xlabel( r"$p_0$" + "({})".format(event_time) , fontsize = fontsize )
	
	ax = fig.add_subplot(nrow,ncol,3)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( lp.frr( Sq.loc[:,"rr",m] ) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( lp.frr(statsEvent.loc["be","rr",m]) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
#		ax.boxplot( lp.frr(statsEvent.loc[:,"rr",m]) , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops  , widths = widths )
	for x in lp.rr.values:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
#	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_yticklabels([])
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.rr.values )
	ax.set_xticklabels( lp.rr.names , fontsize = fontsize )
	ax.set_xlabel( r"$\mathrm{RR}$" + "({})".format(event_time) , fontsize = fontsize )
	
	
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
	
	dx = 1.
	i = 0
	xvalI = np.array([])
	while xvalI.size < 6:
		xvalI = np.arange( round(xminI,i) , round(xmaxI,i) + dx , dx )
		dx /= 2
		i += 1
	
	dx = 1.
	i = 0
	dxvalI = np.array([])
	while dxvalI.size < 6:
		dxvalI = np.arange( round(xmindI,i) , round(xmaxdI,i) + dx , dx )
		dx /= 2
		i += 1
	
	## Intensities plot
	ax = fig.add_subplot(nrow,ncol,4)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( Sq.loc[:,"iall",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( statsEvent.loc["be","iall",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in xvalI:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( xvalI )
	ax.set_xticklabels( xvalI , fontsize = fontsize )
	ax.set_xlabel( r"$\mathrm{i}_1$" + "({})".format(event_time) + " ({})".format(event.unit) , fontsize = fontsize )
	ax.set_xlim( (xminI,xmaxI) )

	ax = fig.add_subplot(nrow,ncol,5)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( Sq.loc[:,"inat",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( statsEvent.loc["be","inat",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in xvalI:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( [] )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( xvalI )
	ax.set_xticklabels( xvalI , fontsize = fontsize )
	ax.set_xlabel( r"$\mathrm{i}_0$" + "({})".format(event_time) + " ({})".format(event.unit) , fontsize = fontsize )
	ax.set_xlim( (xminI,xmaxI) )
	
	ax = fig.add_subplot(nrow,ncol,6)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( Sq.loc[:,"di",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( statsEvent.loc["be","di",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in dxvalI:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( [] )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( dxvalI )
	ax.set_xticklabels( dxvalI , fontsize = fontsize )
	ax.set_xlabel( r"$\delta\mathrm{i}$" + "({})".format(event_time) + " ({})".format(event.unit) , fontsize = fontsize )
	ax.set_xlim( (xmindI,xmaxdI) )
	
	
	fig.set_tight_layout(True)
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
		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose: print( "Plot stats_relative (Done)" )
##}}}


