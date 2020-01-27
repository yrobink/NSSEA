
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


def stats_event( clim , event , ofile , time = None , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.stats_event
	======================
	
	Plot statistics at a time
	
	Arguments
	---------
	clim    : NSSEA.Climatology
		A clim variable
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	time: time
		time to plot
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot stats event" , end = "\r" )
	
	if time is None:
		time = event.time
	## Extract parameters
	statsEvent = clim.stats.loc[time,:,:,:]
	Sq = statsEvent[1:,:,:].quantile( [ci/2,1-ci/2] , dim = "sample" ).assign_coords( quantile = ["q0","q1"] )
	n_models   = statsEvent.models.size
	
	
	## Scale
	PR_min = float(min(Sq.loc[:,"PR",:].min() , statsEvent.loc["be","PR",:].min()))
	PR_max = float(max(Sq.loc[:,"PR",:].max() , statsEvent.loc["be","PR",:].max()))
	p_min = float(min(Sq.loc[:,["pC","pF"],:].min() , statsEvent.loc["be",["pC","pF"],:].min()))
	p_max = float(max(Sq.loc[:,["pC","pF"],:].max() , statsEvent.loc["be",["pC","pF"],:].max()))
	lp = LinkParams( rr_min = PR_min , rr_max = PR_max , p_min = p_min , p_max = p_max )
	
	
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
		ax.fill_between( lp.fp( Sq.loc[:,"pF",m] ) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( lp.fp(statsEvent.loc["be","pF",m]) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in lp.p.values:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.p.values )
	ax.set_xticklabels( lp.p.names , fontsize = fontsize )
	ax.set_xlabel( r"$p_\mathrm{F}$" + "({})".format(time) , fontsize = fontsize )
	
	ax = fig.add_subplot(nrow,ncol,2)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( lp.fp( Sq.loc[:,"pC",m] ) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( lp.fp(statsEvent.loc["be","pC",m]) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in lp.p.values:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
#	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_yticklabels([])
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.p.values )
	ax.set_xticklabels( lp.p.names , fontsize = fontsize )
	ax.set_xlabel( r"$p_\mathrm{C}$" + "({})".format(time) , fontsize = fontsize )
	
	ax = fig.add_subplot(nrow,ncol,3)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( lp.frr( Sq.loc[:,"PR",m] ) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( lp.frr(statsEvent.loc["be","PR",m]) , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
#		ax.boxplot( lp.frr(statsEvent.loc[:,"rr",m]) , positions = [n_models - i] , vert = False , whis = whis , patch_artist = True , boxprops = boxprops , medianprops = medianprops  , widths = widths )
	for x in lp.rr.values:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
#	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_yticklabels([])
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( lp.rr.values )
	ax.set_xticklabels( lp.rr.names , fontsize = fontsize )
	ax.set_xlabel( r"$\mathrm{PR}$" + "({})".format(time) , fontsize = fontsize )
	
	
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
		ax.fill_between( Sq.loc[:,"IF",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( statsEvent.loc["be","IF",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in xvalI:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( statsEvent.models.values.tolist() , fontsize = fontsize )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( xvalI )
	ax.set_xticklabels( xvalI , fontsize = fontsize )
	ax.set_xlabel( r"$\mathbf{I}_\mathrm{F}$" + "({})".format(time) + " ({})".format(event.unit_variable) , fontsize = fontsize )
	ax.set_xlim( (xminI,xmaxI) )

	ax = fig.add_subplot(nrow,ncol,5)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( Sq.loc[:,"IC",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( statsEvent.loc["be","IC",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in xvalI:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( [] )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( xvalI )
	ax.set_xticklabels( xvalI , fontsize = fontsize )
	ax.set_xlabel( r"$\mathbf{I}_\mathrm{C}$" + "({})".format(time) + " ({})".format(event.unit_variable) , fontsize = fontsize )
	ax.set_xlim( (xminI,xmaxI) )
	
	ax = fig.add_subplot(nrow,ncol,6)
	for i,m in enumerate(statsEvent.models):
		ax.fill_between( Sq.loc[:,"dI",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "red" )
		ax.vlines( statsEvent.loc["be","dI",m] , n_models - i - 1 / 3, n_models - i + 1 / 3 , color = "black" )
	for x in dxvalI:
		ax.vlines( x , 0 , n_models + 1 , color = "grey" , linestyle = "--" , alpha = 0.3 )
	ax.set_yticks( range(n_models,0,-1) )
	ax.set_yticklabels( [] )
	ax.set_ylim( (1-1,n_models + 1) )
	ax.set_xticks( dxvalI )
	ax.set_xticklabels( dxvalI , fontsize = fontsize )
	ax.set_xlabel( r"$\Delta\mathbf{I}$" + "({})".format(time) + " ({})".format(event.unit_variable) , fontsize = fontsize )
	ax.set_xlim( (xmindI,xmaxdI) )
	
	
	fig.set_tight_layout(True)
	fig.savefig( ofile )
	plt.close(fig)
	
	if verbose: print( "Plot stats event (Done)" )
##}}}

def stats_relative( clim , event , ofile , time = None , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.stats_relative
	=========================
	
	Plot probabilities PR/PR[time_event] and di - di[time_event] along time
	
	Arguments
	---------
	clim    : NSSEA.Climatology
		A clim variable
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	time: time
		time to plot
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot stats_relative" , end = "\r" )
	
	statsIn = clim.stats
	## Compute stats events
	if time is None:
		time = event.time
	stats = stats_relative_event( statsIn , time )
	statsu = stats[:,1:,:,:].quantile( ci / 2.      , dim = "sample" )
	statsl = stats[:,1:,:,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	ymindI = min( stats.loc[:,:,"dI",:].min()   , statsu.loc[:,"dI",:].min()  , statsl.loc[:,"dI",:].min()  )
	ymaxdI = max( stats.loc[:,:,"dI",:].max()   , statsu.loc[:,"dI",:].max()  , statsl.loc[:,"dI",:].max()  )
	ylabel = "\mathrm{(" + event.unit_variable + ")}"
	
	lp = LinkParams()
	
	pdf = mpdf.PdfPages( ofile )
	
	for m in stats.models:
		nrow,ncol = 2,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.6 * fs * nrow ) )
		
		## Probabilities
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( stats.time , lp.frr(stats.loc[:,"be","PR",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.frr(statsl.loc[:,"PR",m]) , lp.frr(statsu.loc[:,"PR",m]) , color = "red" , alpha = 0.5 )
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
		ax.plot( [time,time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		## Intensities
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( stats.time , stats.loc[:,"be","dI",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"dI",m] , statsu.loc[:,"dI",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (ymindI,ymaxdI) )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( r"${}$".format( "\delta\mathbf{i}(t)\ " + ylabel ) )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [time,time] , ylim  , linestyle = "--" , marker = "" , color = "black" )
		ax.plot( xlim                    , [0,0] , linestyle = "-"  , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose: print( "Plot stats_relative (Done)" )
##}}}


