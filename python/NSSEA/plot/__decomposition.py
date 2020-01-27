
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


###############
## Functions ##
###############

def decomposition( lX , X , event , ofile , ci = 0.05 , verbose = False ): ##{{{
	"""
	NSSEA.plot.decomposition
	========================
	
	Plot the GAM decomposition of covariates
	
	Arguments
	---------
	Xd        : pandas.DataFrame
		Covariates before decomposition
	X         : xarray
		Covariates afer decomposition (NSSEA.Climatology.X)
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose : print( "Plot decomposition" , end = "\r" )
	
	Xl = X[:,1:,:,:].quantile( ci / 2.      , dim = "sample" )
	Xu = X[:,1:,:,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	pdf = mpdf.PdfPages( ofile )
	
	## Remove multi
	models = X.models.values.tolist()
	if "multi" in models:
		models = models[:-1]
	
	ymin = min( float(X.loc[:,:,["all","nat"],:].min()) , min([ float(lx.min()) for lx in lX]) )
	ymax = max( float(X.loc[:,:,["all","nat"],:].max()) , max([ float(lx.max()) for lx in lX]) )
	
	yminAnt = float(X.loc[:,:,"ant",:].min())
	ymaxAnt = float(X.loc[:,:,"ant",:].max())
	
	ylabel = r"${}$".format( "\mathrm{" + event.name_variable +"}\ \mathrm{(" + event.unit_variable + ")}" )
	
	## Loop
	for i,m in enumerate(X.models):
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( lX[i].index , lX[i].values.ravel() , color = "grey" , linestyle = "" , marker = "." , alpha = 0.3 , label = "" )
		ax.plot( X.time , X.loc[:,"be","all",m] , color = "red" , linestyle = "-" , marker = "" , label = r"$\mathrm{ALL}$" )
		ax.fill_between( X.time , Xl.loc[:,"all",m] , Xu.loc[:,"all",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (ymin,ymax) )
		ax.legend( loc = "upper left" )
		ax.set_xticks([])
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_ylabel( ylabel )
		
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( X.time , X.loc[:,"be","nat",m] , color = "blue" , linestyle = "-" , marker = "" , label = r"$\mathrm{NAT}$" )
		ax.fill_between( X.time , Xl.loc[:,"nat",m] , Xu.loc[:,"nat",m] , color = "blue" , alpha = 0.5 )
		ax.set_ylim( (ymin,ymax) )
		ax.set_xticks([])
		ax.legend( loc = "upper left" )
		ax.set_ylabel( ylabel )
		
		ax = fig.add_subplot( nrow , ncol , 3 )
		ax.plot( X.time , X.loc[:,"be","ant",m] , color = "green" , linestyle = "-" , marker = "" , label = r"$\mathrm{ANT}$" )
		ax.fill_between( X.time , Xl.loc[:,"ant",m] , Xu.loc[:,"ant",m] , color = "green" , alpha = 0.5 )
		ax.set_ylim( (yminAnt,ymaxAnt) )
		ax.legend( loc = "upper left" )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( ylabel )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( xlim , [0,0] , color = "black" , linestyle = "-" , marker = "" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose : print( "Plot decomposition (Done)" )
##}}}

