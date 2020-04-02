
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


def probabilities( clim , event , ofile , be_is_median = False , ci = 0.05 , verbose = False ): ##{{{
	"""
	NSSEA.plot.probabilities
	========================
	
	Plot probabilities (pF,pC,PR) along time
	
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
	
	if verbose: print( "Plot probabilities" , end = "\r" )
	
	stats  = clim.stats[:,:,:3,:]
	
	## Find impossible values
	##=======================
	nan_idx    = np.logical_and( stats[:,1:,:,:].loc[:,:,"pF",:] == 0 , stats[:,1:,:,:].loc[:,:,"pC",:] == 0 )
	nan_values = nan_idx.sum( dim = "sample" ) / ( stats.sample.size - 1 )
	imp_values = ( stats[:,1:,:,:].loc[:,:,"pC",:] == 0 ).sum( dim = "sample" ) / ( stats.sample.size - 1 )
	
	
	## Find quantiles
	##===============
	qstats = stats[:,1:,:,:].quantile( [ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample" ).assign_coords( quantile = [ "ql" , "qu" , "be" ] )
	if not be_is_median: qstats.loc["be",:,:,:] = stats[:,0,:,:]
	
	## Special case : PR
	##==================
	qstats.loc["qu",:,"PR",:] = qstats.loc["qu",:,"PR",:].where( nan_values < ci , np.inf )
	qstats.loc["ql",:,"PR",:] = qstats.loc["ql",:,"PR",:].where( nan_values < ci , 0      )
	
	## Plot itself
	##============
	
	lp = LinkParams()
	
	pdf = mpdf.PdfPages( ofile )
	
	for m in stats.models:
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_axes( [0.08,0.67,0.85,0.3] )
		ax.plot( stats.time , lp.fp(qstats.loc["be",:,"pF",m]) , color = "red"  , linestyle = "-" , marker = "" , label = r"$p_\mathrm{F}(t)$" )
		ax.plot( stats.time , lp.fp(qstats.loc["be",:,"pC",m]) , color = "blue" , linestyle = "-" , marker = "" , label = r"$p_\mathrm{C}(t)$" )
		ax.fill_between( stats.time , lp.fp(qstats.loc["ql",:,"pF",m]) , lp.fp(qstats.loc["qu",:,"pF",m]) , color = "red" , alpha = 0.5 )
		ax.fill_between( stats.time , lp.fp(qstats.loc["ql",:,"pC",m]) , lp.fp(qstats.loc["qu",:,"pC",m]) , color = "blue" , alpha = 0.5 )
		ax.legend( loc = "upper left" )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_xticks([])
		ax.set_ylabel( "Probabilities" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		ax = fig.add_axes( [0.08,0.50,0.85,0.15] )
		ax.plot( stats.time , lp.fp(nan_values.loc[:,m]) , color = "red"  , label = r"$p_\mathrm{F}=p_\mathrm{C}=0$" )
		ax.plot( stats.time , lp.fp(imp_values.loc[:,m]) , color = "blue" , label = r"$p_\mathrm{C}=0$" )
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_ylabel( "Probabilities" )
		xlim = ax.get_xlim()
		ax.hlines( lp.fp(ci) , xlim[0] , xlim[1] , color = "grey" , label = "Threshold" )
		ax.set_xlim(xlim)
		ax.set_xticks([])
		ax.legend( loc = "upper right" )
		
		ax = fig.add_axes( [0.08,0.08,0.85,0.4] )
		ax.plot( stats.time , lp.frr(qstats.loc["be",:,"PR",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.frr(qstats.loc["ql",:,"PR",m]) , lp.frr(qstats.loc["qu",:,"PR",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.rr.values.min(),lp.rr.values.max()) )
		ax.set_yticks( lp.rr.values )
		ax.set_yticklabels( lp.rr.names )
		ax.set_xlabel( r"$\mathrm{Time}$" )
		ax.set_ylabel( r"$\mathrm{PR}(t)$" )
		ax2 = fig.add_axes( [0.08,0.08,0.85,0.4] , sharex = ax , frameon = False )
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

		
#		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose: print( "Plot probabilities (Done)" )
##}}}

def probabilities_not_zero( clim , event , ofile , ci = 0.05 , verbose = False ): ##{{{
	"""
	NSSEA.plot.probabilities_not_zero
	=================================
	
	Plot probabilities (pF,pC,PR) along time, but assume than pC,pF > 0
	
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
	
	if verbose: print( "Plot probabilities" , end = "\r" )
	
	stats = clim.stats
	statsl = stats[:,1:,:3,:].quantile( ci / 2.      , dim = "sample" )
	statsu = stats[:,1:,:3,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	
	lp = LinkParams()
	
	pdf = mpdf.PdfPages( ofile )
	
	for m in stats.models:
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( stats.time , lp.fp(stats.loc[:,"be","pF",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.fp(statsl.loc[:,"pF",m]) , lp.fp(statsu.loc[:,"pF",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_xticks([])
		ax.set_ylabel( r"$p_\mathrm{F}(t)$" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( stats.time , lp.fp(stats.loc[:,"be","pC",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.fp(statsl.loc[:,"pC",m]) , lp.fp(statsu.loc[:,"pC",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_xticks([])
		ax.set_ylabel( r"$p_\mathrm{C}(t)$" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		ax = fig.add_subplot( nrow , ncol , 3 )
		ax.plot( stats.time , lp.frr(stats.loc[:,"be","PR",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.frr(statsl.loc[:,"PR",m]) , lp.frr(statsu.loc[:,"PR",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.rr.values.min(),lp.rr.values.max()) )
		ax.set_yticks( lp.rr.values )
		ax.set_yticklabels( lp.rr.names )
		ax.set_xlabel( r"$\mathrm{Time}$" )
		ax.set_ylabel( r"$\mathrm{PR}(t)$" )
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

def return_time( clim , event , ofile , be_is_median = False , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.probabilities
	========================
	
	Plot return time (RtF,RtC) along time
	
	Arguments
	---------
	clim      : NSSEA.Climatology
		Climatology with stats computed
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	be_is_median : boolean
		If we assume than the true best estimate is the median of bootstrap (usefull for Bayesian)
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot return time" , end = "\r" )
	
	## Compute quantiles and best estimate
	##====================================
	qRt = clim.stats[:,1:,:,:].loc[:,:,["RtF","RtC"],:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample" ).assign_coords( quantile = ["ql","qu","med"] )
	if not be_is_median: qRt.loc["med",:,:,:] = clim.stats.loc[:,"be",["RtF","RtC"],:]
	
	## Some parameters
	##================
	lp = np.log10
	pdf = mpdf.PdfPages( ofile )
	
	for m in qRt.models:
		
		fig = plt.figure( figsize = (8,6) )
		
		ax = fig.add_subplot( 1 , 1 , 1 )
		
		ax.plot( qRt.time , lp(qRt.loc["med",:,"RtF",m]) , color = "red" , label = r"Rt$^F$" )
		ax.fill_between( qRt.time , lp(qRt.loc["ql",:,"RtF",m]) , lp(qRt.loc["qu",:,"RtF",m]) , color = "red" , alpha = 0.5 )
		
		ax.plot( qRt.time , lp(qRt.loc["med",:,"RtC",m]) , color = "blue" , label = r"Rt$^C$" )
		ax.fill_between( qRt.time , lp(qRt.loc["ql",:,"RtC",m]) , lp(qRt.loc["qu",:,"RtC",m]) , color = "blue" , alpha = 0.5 )
		
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_ylim(ylim)
		
		imax = int(lp(float(qRt.loc["qu",:,:,m].max()))) + 1
		
		ax.set_yticks( range(imax) )
		ax.set_yticklabels( [ 10**i for i in range(imax) ] )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( "Return time" )
		ax.legend()
		ax.set_title( m.values )
		
		pdf.savefig(fig)
		plt.close(fig)
	
	pdf.close()
	if verbose: print( "Plot return time (Done)" )
##}}}

