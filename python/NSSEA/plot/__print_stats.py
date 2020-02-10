
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
	Print in a string a tabular summarizing statistics (FAR,PR,dI,pF,pC,RtF,RtC,IF,IC) at a time.
	
	FAR : Fraction of Attribuable Risk
	PR  : Probability Ratio
	dI  : Difference in intensity between factual/counter world
	pF  : Probability of event in factual world
	pC  : Probability of event in counter factual world
	RtF : Return time of event in factual world
	RtC : Return time of event in counter factual world
	IF  : Intensity of event in factual world
	IC  : Intensity of event in counter factual world
	
	Parameters
	----------
	clim    : NSSEA.Climatology
		Climatology fitted
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
	
	## Add FAR and return time
	FAR = (1 - 1 / S.loc[:,"PR"]).assign_coords( stats = "FAR" )
	Rt  = ( 1. / S.loc[:,["pF","pC"]] ).assign_coords( stats = ["RtF","RtC"] )
	S = xr.concat( (S,FAR,Rt) , dim = "stats" )
	
	## Find quantile
	q0 = ci /2
	qm = 0.5
	q1 = 1 - ci / 2
	Sq = S[1:,:].quantile( [ q0,qm,q1 ] , dim = "sample" ).assign_coords( quantile = ["q0","qm","q1"] )
	
	tab = tt.Texttable()
	tab.set_precision(digit)
	tab.set_cols_align(["l", "r", "r", "r", "r"])
	tab.set_cols_dtype( ["t" ,"e" , "e","e" , "e"])
	
	tab.header( ["Stats {}".format(time),"Best estimate","Quantile {}".format(q0),"Median","Quantile {}".format(q1)])
	for s in ["FAR","PR","dI","RtF","RtC","pF","pC","IF","IC"]:
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
		S = clim.stats.loc[[time,time_rel],:,:,model]
	except:
		return ""
	
	
	## Add FAR and return time
	FAR = (1 - 1 / S.loc[:,:,"PR"]).assign_coords( stats = "FAR" )
	Rt  = ( 1. / S.loc[:,:,["pF","pC"]] ).assign_coords( stats = ["RtF","RtC"] )
	S = xr.concat( (S,FAR,Rt) , dim = "stats" )
	
	div = ["pC","pF","PR","FAR"]
	sub = ["IC","IF","dI","RtF","RtC"]
	S.loc[:,:,div] = S.loc[:,:,div] / S.loc[time_rel,:,div]
	S.loc[:,:,sub] = S.loc[:,:,sub] - S.loc[time_rel,:,sub]
	
	q0 = ci /2
	qm = 0.5
	q1 = 1 - ci / 2
	Sq = S[:,1:,:].quantile( [ q0,qm,q1 ] , dim = "sample" ).assign_coords( quantile = ["q0","qm","q1"] )
	
	tab = tt.Texttable()
	tab.set_precision(digit)
	tab.set_cols_align(["l", "r", "r", "r", "r"])
	tab.set_cols_dtype( ["t" ,"e" , "e","e" , "e"])
	
	tab.header( ["Stats {} / {}".format(time,time_rel),"Best estimate","Quantile {}".format(q0),"Median","Quantile {}".format(q1)])
	for s in ["FAR","PR","dI","RtF","RtC","pF","pC","IF","IC"]:
		tab.add_row( [ s , float(S.loc[time,"be",s]) , float(Sq.loc["q0",time,s]) , float(Sq.loc["qm",time,s]), float(Sq.loc["q1",time,s]) ] )
	
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



