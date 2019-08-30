
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
	
	if verbose: print( "Plot ns_params" , end = "\r" )
	## ns params
	ns_params = clim.ns_params - clim.ns_params[:,0,:]
	n_ns_params = clim.n_ns_params
	
	## Extract quantile
	ns_q = ns_params[:,1:,:].quantile( [ ci / 2. , 1 - ci / 2. ] , dim = ["sample"] ) 
	ns_q = ns_q.assign_coords( quantile = ["l","u"] )
	
	
	ns_law = clim.ns_law
	ns_law_args = clim.ns_law_args
	law = ns_law(**ns_law_args)
	lf = law.link_fct_by_params()
	
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
		ax.set_xticklabels( ns_params.ns_params.values.tolist() )
		ax.set_xlabel( "Parameters" )
		ax.set_ylabel( "Anomalies parameters" )
		
		fig.set_tight_layout(True)
		pdf.savefig(fig)
		plt.close(fig)
	
	pdf.close()
	if verbose: print( "Plot ns_params (Done)" )
##}}}

def	ns_params_comparison( clim , clim2 , ofile , ci = 0.05 , verbose = False ):##{{{
	
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
	
	
	ns_law = clim.ns_law
	ns_law_args = clim.ns_law_args
	law = ns_law(**ns_law_args)
	lf = law.link_fct_by_params()
	
	ns_law2 = clim2.ns_law
	ns_law_args2 = clim2.ns_law_args
	law2 = ns_law2(**ns_law_args2)
	lf2 = law2.link_fct_by_params()
	
	
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
		ax.set_xticklabels( ns_params.ns_params.values.tolist() )
		
		ax.set_xlabel( "Parameters" )
		ax.set_ylabel( "Anomalies parameters" )
		
		fig.set_tight_layout(True)
		pdf.savefig(fig)
		plt.close(fig)
	
	pdf.close()
	if verbose: print( "Plot ns_params_comparison (Done)" )
##}}}


