
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

def	ns_params( clim , ofile , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.ns_params
	====================
	
	Plot boxplot of non-stationary parameters 
	
	Arguments
	---------
	clim      : NSSEA.Climatology
		Climatology
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot ns_params" , end = "\r" )
	## ns params
	ns_params = clim.ns_params - clim.ns_params[:,0,:]
	n_ns_params = clim.n_ns_params
	
	## Extract quantile
	ns_q = ns_params[:,1:,:].quantile( [ ci / 2. , 1 - ci / 2. ] , dim = ["sample"] ) 
	ns_q = ns_q.assign_coords( quantile = ["l","u"] )
	
	
	law = clim.ns_law
	lf = []
	for p in law.lparams:
		for _ in range(law.lparams[p].n_params):
			lf.append(law.lparams[p].link)
	
	pdf = mpdf.PdfPages( ofile )
	for m in ns_params.models:
		
		fig = plt.figure( figsize = (7,7) )
		ax = fig.add_subplot( 1 , 1 , 1 )
		
		for i,p in enumerate(ns_params.ns_params):
			
			xl = i - 0.3
			xr = i + 0.3
			
			ax.hlines( 0 , xl , xr , color = "black" )
			val_be = lf[i](ns_params.loc[p,"be",m].values)
			ax.fill_between( [xl,xr] , lf[i](ns_q.loc["l",p,m].values) - val_be , lf[i](ns_q.loc["u",p,m]) - val_be , color = "red" , alpha = 0.5 )
			ax.text( i - 0.3 , 0 , round( float(lf[i](clim.ns_params.loc[p,"be",m])) , 2 ) )
		
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_xlim( (-0.5,n_ns_params-0.5) )
		ax.set_xticks( range(n_ns_params) )
		ax.set_xticklabels( law.get_params_names(True) )
		ax.set_xlabel( "Parameters" )
		ax.set_ylabel( "Anomalies parameters" )
		
		fig.set_tight_layout(True)
		pdf.savefig(fig)
		plt.close(fig)
	
	pdf.close()
	if verbose: print( "Plot ns_params (Done)" )
##}}}

def	ns_params_comparison( clim , clim2 , ofile , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.ns_params_comparison
	===============================
	
	Plot boxplot of two set of non-stationary parameters for comparison
	
	Arguments
	---------
	clim      : NSSEA.Climatology
		Climatology
	clim2      : NSSEA.Climatology
		Climatology
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot ns_params_comparison" , end = "\r" )
	## ns params
	ns_params  = clim.ns_params - clim.ns_params[:,0,:]
	ns_params2 = clim2.ns_params - clim2.ns_params[:,0,:]
	n_ns_params = clim.n_ns_params
	
	## Extract quantile
	ns_q = ns_params[:,1:,:].quantile( [ ci / 2. , 1 - ci / 2. ] , dim = ["sample"] ) 
	ns_q = ns_q.assign_coords( quantile = ["l","u"] )
	ns_q2 = ns_params2[:,1:,:].quantile( [ ci / 2. , 1 - ci / 2. ] , dim = ["sample"] ) 
	ns_q2 = ns_q2.assign_coords( quantile = ["l","u"] )
	
	
	law = clim.ns_law
	law2 = clim2.ns_law
	lf = []
	for p in law.lparams:
		for _ in range(law.lparams[p].n_params):
			lf.append(law.lparams[p].link)
	
	lf2 = []
	for p in law2.lparams:
		for _ in range(law2.lparams[p].n_params):
			lf2.append(law2.lparams[p].link)
	
	pdf = mpdf.PdfPages( ofile )
	for m in ns_params.models:
		
		fig = plt.figure( figsize = (7,7) )
		ax = fig.add_subplot( 1 , 1 , 1 )
		
		for i,p in enumerate(ns_params.ns_params):
			
			xl = i - 0.3
			xr = i + 0.3
			
			ax.hlines( 0 , xl , xr , color = "black" )
			val_be = lf[i](ns_params.loc[p,"be",m].values)
			ax.fill_between( [xl,xr] , lf[i](ns_q.loc["l",p,m].values) - val_be , lf[i](ns_q.loc["u",p,m]) - val_be , color = "red" , alpha = 0.2 )
			ax.text( i - 0.3 , 0 , round( float(lf[i](clim.ns_params.loc[p,"be",m])) , 2 ) )
			
			ax.hlines( 0 , xl , xr , color = "black" )
			val_be2 = lf2[i](ns_params2.loc[p,"be",m].values)
			ax.fill_between( [xl+0.1,xr-0.1] , lf2[i](ns_q2.loc["l",p,m].values) - val_be2 , lf2[i](ns_q2.loc["u",p,m]) - val_be2 , color = "red" , alpha = 0.5 )
			ax.text( i + 0.2 , 0 , round( float(lf2[i](clim2.ns_params.loc[p,"be",m])) , 2 ) )
		
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_xlim( (-0.5,n_ns_params-0.5) )
		ax.set_xticks( range(n_ns_params) )
		ax.set_xticklabels( law.get_params_names(True) )
		
		ax.set_xlabel( "Parameters" )
		ax.set_ylabel( "Anomalies parameters" )
		
		fig.set_tight_layout(True)
		pdf.savefig(fig)
		plt.close(fig)
	
	pdf.close()
	if verbose: print( "Plot ns_params_comparison (Done)" )
##}}}

def ns_params_time( clim , ofile , time = None , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.ns_params_time
	=========================
	
	Plot non-stationary parameters along time
	
	Arguments
	---------
	clim      : NSSEA.Climatology
		Climatology
	ofile     : str
		output file
	time      : array
		Array of time where to plot ns_params
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot time ns_params" , end = "\r" )
	if time is None:
		time = clim.time
	
	l_params = [k for k in clim.ns_law.lparams]
	s_params = xr.DataArray( np.zeros( (time.size,clim.n_sample+1,2,3) ) , dims = ["time","sample","forcing","params"] , coords = [time,clim.X.sample,["all","nat"],l_params] )
	
	for m in clim.models:
		for s in s_params.sample:
			clim.ns_law.set_params(clim.ns_params.loc[:,s,m].values)
			for f in s_params.forcing:
				clim.ns_law.set_covariable( clim.X.loc[time,s,f,m].values , time )
				for p in l_params:
					s_params.loc[:,s,f,p] = clim.ns_law.lparams[p](time)
		
		qs_params = s_params[:,1:,:,:].quantile( [ ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample" ).assign_coords( quantile = ["ql","qu","me"] )
		
		xlim = [time.min(),time.max()]
		deltax = 0.05 * ( xlim[1] - xlim[0] )
		xlim[0] -= deltax
		xlim[1] += deltax
		
		pdf = mpdf.PdfPages( ofile )
		fig = plt.figure( figsize = (12,12) )
		
		
		for i,p in enumerate(qs_params.params):
		
			ax = fig.add_subplot( len(l_params) , 1 , i + 1 )
			ax.plot( time , qs_params.loc["me",:,"all",p] , color = "red" )
			ax.fill_between( time , qs_params.loc["ql",:,"all",p] , qs_params.loc["qu",:,"all",p] , color = "red" , alpha = 0.5 )
			ax.plot( time , qs_params.loc["me",:,"nat",p] , color = "blue" )
			ax.fill_between( time , qs_params.loc["ql",:,"nat",p] , qs_params.loc["qu",:,"nat",p] , color = "blue" , alpha = 0.5 )
			xticks = ax.get_xticks()
			ax.set_xticks([])
			ax.set_xlim( xlim )
			ax.set_ylabel(str(p.values))
			if i == 0: ax.set_title(m)
		
		ax.set_xticks(xticks)
		ax.set_xlim(xlim)
		ax.set_xlabel("Time")
		
		
		fig.set_tight_layout(True)
		pdf.savefig(fig)
		plt.close(fig)
	
	pdf.close()
	if verbose: print( "Plot time ns_params (Done)" )
##}}}

