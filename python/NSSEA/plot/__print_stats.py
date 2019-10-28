
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

def print_time_stats( S , time , model = "multi" , digit = 3 , ci = 0.05 , verbose = False ):##{{{
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
	s = "rr"
	tab.add_row( [ "FAR" , float(1-1/S.loc["be",s]) , float(1-1/Sq.loc["q0",s]) ,float(1-1/Sq.loc["qm",s]) , float(1-1/Sq.loc["q1",s]) ] )
	for name,s in zip(["RR","dI","p1","p0","I1","I0"],["rr","di","pall","pnat","iall","inat"]):
		tab.add_row( [ name , float(S.loc["be",s]) , float(Sq.loc["q0",s]) , float(Sq.loc["qm",s]), float(Sq.loc["q1",s]) ] )
	
	if verbose: print( "Print time stats (Done)" )
	return tab.draw() + "\n"
##}}}

def print_relative_time_stats( S , time , time_rel , model = "multi" , digit = 3 , ci = 0.05 , verbose = False ):##{{{
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
		Sr = S.loc[time,:,:,model].copy()
		d  = ["pnat","pall","rr"]
		i  = ["inat","iall","di"]
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
	s = "rr"
	tab.add_row( [ "FAR" , float(1-1/S.loc["be",s]) , float(1-1/Sq.loc["q0",s]) ,float(1-1/Sq.loc["qm",s]) , float(1-1/Sq.loc["q1",s]) ] )
	for name,s in zip(["RR","dI","p1","p0","I1","I0"],["rr","di","pall","pnat","iall","inat"]):
		tab.add_row( [ name , float(S.loc["be",s]) , float(Sq.loc["q0",s]) , float(Sq.loc["qm",s]), float(Sq.loc["q1",s]) ] )
	
	if verbose: print( "Print relative time stats (Done)" )
	return tab.draw() + "\n"
##}}}


