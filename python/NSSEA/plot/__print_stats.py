
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


try:
	import matplotlib.pyplot as plt
except:
	import matplotlib as mpl
	mpl.use("Qt5Agg")
	import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf



###############
## Functions ##
###############

def print_time_stats( S , time , model = "multi" , digit = 3 , ci = 0.05 ):
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
	
	return tab.draw() + "\n"


def print_relative_time_stats( S , time , time_rel , model = "multi" , digit = 3 , ci = 0.05 ):
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
	
	return tab.draw() + "\n"

#	Sq = S.quantile( [ ci / 2 , 1 - ci / 2 ] , dim = "sample" ).assign_coords( quantile = ["q0","q1"] )
#	out =  "#######################################################\n"
#	out += "## Relative values at time {}\n".format(time_rel)
#	out += "## Time : {}\n".format(time)
#	out += "## CI   : {}%\n".format(100*(1-ci))
#	s = "rr"
#	out += "## FAR  : {} ( {} \/ {} )\n".format( float( (1-1/S.loc["be",s]).round(digit)) , float( (1-1/Sq).loc["q0",s].round(digit)) , float((1-1/Sq).loc["q1",s].round(digit)) )
#	for name,s in zip(["RR","dI","p1","p0","I1","I0"],["rr","di","pall","pnat","iall","inat"]):
#		out += "## {}   : {} ( {} \/ {} )\n".format( name , float(S.loc["be",s].round(digit)) , float(Sq.loc["q0",s].round(digit)) , float(Sq.loc["q1",s].round(digit)) )
#	out += "#######################################################\n"
#	return out



