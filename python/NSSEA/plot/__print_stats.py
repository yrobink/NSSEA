
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
import texttable as tt



###############
## Functions ##
###############

def print_time_stats( clim , time , model = "multi" , digit = 3 , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.print_time_stats
	===========================
	Print in a string a tabular summarizing statistics (FAR,RR,dI,p1,p0,I1,I0) at a time.
	
	Parameters
	----------
	S       : xr.DataArray
		Statistics from coffee (coffee.stats)
	time    : time type
		A time compatible with S.time
	model   : string
		The model chosen (default is multi)
	digit   : float
		number of digits (default is 3)
	ci      : float
		Level of confidence interval, default is 0.05 (95%)
	verbose : bool
		Print begin / end of execution
	
	Return
	------
	tab : string
		A tabular of statistics
	"""
	if verbose: print( "Print time stats" , end = "\r" )
	try:
		S = clim.stats
		S = S.loc[time,:,:,model].copy()
	except:
		return ""
	q0 = ci /2
	qm = 0.5
	q1 = 1 - ci / 2
	Sq = S[1:,:].quantile( [ q0,qm,q1 ] , dim = "sample" ).assign_coords( quantile = ["q0","qm","q1"] )
	
	tab = tt.Texttable()
	tab.set_precision(digit)
	tab.set_cols_align(["l", "r", "r", "r", "r"])
	tab.set_cols_dtype( ["t" ,"e" , "e","e" , "e"])
	
	tab.header( ["Stats {}".format(time),"Best estimate","Quantile {}".format(q0),"Median","Quantile {}".format(q1)])
	tab.add_row( [ "FAR" , float(1-1/S.loc["be","PR"]) , float(1-1/Sq.loc["q0","PR"]) ,float(1-1/Sq.loc["qm","PR"]) , float(1-1/Sq.loc["q1","PR"]) ] )
	for s in ["PR","dI","pF","pC","IF","IC"]:
		tab.add_row( [ s , float(S.loc["be",s]) , float(Sq.loc["q0",s]) , float(Sq.loc["qm",s]), float(Sq.loc["q1",s]) ] )
	
	if verbose: print( "Print time stats (Done)" )
	return tab.draw() + "\n"
##}}}

def print_relative_time_stats( clim , time , time_rel , model = "multi" , digit = 3 , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.print_relative_time_stats
	====================================
	Print in a string a tabular summarizing statistics (FAR,RR,dI,p1,p0,I1,I0) at a time relative to a second time time_rel.
	
	Parameters
	----------
	S        : xr.DataArray
		Statistics from coffee (coffee.stats)
	time     : time type
		A time compatible with S.time
	time_rel : time type
		A time compatible with S.time
	model    : string
		The model chosen (default is multi)
	digit    : float
		number of digits (default is 3)
	ci       : float
		Level of confidence interval, default is 0.05 (95%)
	verbose  : bool
		Print begin / end of execution
	
	Return
	------
	tab : string
		A tabular of statistics
	"""
	if verbose: print( "Print relative time stats" , end = "\r" )
	try:
		S = clim.stats
		Sr = S.loc[time,:,:,model].copy()
		d  = ["pC","pF","PR"]
		i  = ["IC","IF","dI"]
		Sr.loc[:,d] = Sr.loc[:,d] / S.loc[time_rel,:,d,model]
		Sr.loc[:,i] = Sr.loc[:,i] - S.loc[time_rel,:,i,model]
		S = Sr
	except:
		return ""
	
	
	q0 = ci /2
	qm = 0.5
	q1 = 1 - ci / 2
	Sq = S[1:,:].quantile( [ q0,qm,q1 ] , dim = "sample" ).assign_coords( quantile = ["q0","qm","q1"] )
	
	tab = tt.Texttable()
	tab.set_precision(digit)
	tab.set_cols_align(["l", "r", "r", "r", "r"])
	tab.set_cols_dtype( ["t" ,"e" , "e","e" , "e"])
	
	tab.header( ["Stats {} / {}".format(time,time_rel),"Best estimate","Quantile {}".format(q0),"Median","Quantile {}".format(q1)])
	s = "PR"
	tab.add_row( [ "FAR" , float(1-1/S.loc["be",s]) , float(1-1/Sq.loc["q0",s]) ,float(1-1/Sq.loc["qm",s]) , float(1-1/Sq.loc["q1",s]) ] )
	for s in ["PR","dI","pF","pC","IF","IC"]:
		tab.add_row( [ s , float(S.loc["be",s]) , float(Sq.loc["q0",s]) , float(Sq.loc["qm",s]), float(Sq.loc["q1",s]) ] )
	
	if verbose: print( "Print relative time stats (Done)" )
	return tab.draw() + "\n"
##}}}

def write_package_tabular( clim , event , ofile , model = "multi" , time_future = 2040 , digit = 3 , ci = 0.05 , verbose = False ):##{{{
	if verbose: print( "Write package tabular" , end = "\r" )
	
	with open( ofile , "w" ) as f:
		f.write( str(event) + "\n\n" )
		f.write( print_time_stats( clim , event.time  , model = model , digit = digit , ci = ci , verbose = False ) + "\n" )
		f.write( print_time_stats( clim , time_future , model = model , digit = digit , ci = ci , verbose = False ) + "\n" )
		f.write( print_relative_time_stats( clim , time_future , event.time , model = model , digit = digit , ci = ci , verbose = False ) + "\n" )
	
	if verbose: print( "Write package tabular (Done)" , end = "\n" )
##}}}



