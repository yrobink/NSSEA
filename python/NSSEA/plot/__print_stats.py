
##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2020                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## This software is a computer program that is part of the NSSEA                ##
## (Non-Stationary Statistics for Extreme Attribution) This library makes it    ##
## possible to infer the probability of an (extreme) event in the factual /     ##
## counter-factual world (without anthropic forcing) to attribute it to climate ##
## change.                                                                      ##
##                                                                              ##
## This software is governed by the CeCILL-C license under French law and       ##
## abiding by the rules of distribution of free software.  You can  use,        ##
## modify and/ or redistribute the software under the terms of the CeCILL-C     ##
## license as circulated by CEA, CNRS and INRIA at the following URL            ##
## "http://www.cecill.info".                                                    ##
##                                                                              ##
## As a counterpart to the access to the source code and  rights to copy,       ##
## modify and redistribute granted by the license, users are provided only      ##
## with a limited warranty  and the software's author,  the holder of the       ##
## economic rights,  and the successive licensors  have only  limited           ##
## liability.                                                                   ##
##                                                                              ##
## In this respect, the user's attention is drawn to the risks associated       ##
## with loading,  using,  modifying and/or developing or reproducing the        ##
## software by the user in light of its specific status of free software,       ##
## that may mean  that it is complicated to manipulate,  and  that  also        ##
## therefore means  that it is reserved for developers  and  experienced        ##
## professionals having in-depth computer knowledge. Users are therefore        ##
## encouraged to load and test the software's suitability as regards their      ##
## requirements in conditions enabling the security of their systems and/or     ##
## data to be ensured and,  more generally, to use and operate it in the        ##
## same conditions as regards security.                                         ##
##                                                                              ##
## The fact that you are presently reading this means that you have had         ##
## knowledge of the CeCILL-C license and that you accept its terms.             ##
##                                                                              ##
##################################################################################
##################################################################################

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2020                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## Ce logiciel est un programme informatique faisant partie de la librairie     ##
## NSSEA (Non-Stationary Statistics for Extreme Attribution). Cette librairie   ##
## permet d'estimer la probabilité d'un evenement (extreme) dans le monde       ##
## factuel / contre factuel (sans forcage anthropogenique) et de l'attribuer au ##
## changement climatique.                                                       ##
##                                                                              ##
## Ce logiciel est régi par la licence CeCILL-C soumise au droit français et    ##
## respectant les principes de diffusion des logiciels libres. Vous pouvez      ##
## utiliser, modifier et/ou redistribuer ce programme sous les conditions       ##
## de la licence CeCILL-C telle que diffusée par le CEA, le CNRS et l'INRIA     ##
## sur le site "http://www.cecill.info".                                        ##
##                                                                              ##
## En contrepartie de l'accessibilité au code source et des droits de copie,    ##
## de modification et de redistribution accordés par cette licence, il n'est    ##
## offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,    ##
## seule une responsabilité restreinte pèse sur l'auteur du programme, le       ##
## titulaire des droits patrimoniaux et les concédants successifs.              ##
##                                                                              ##
## A cet égard  l'attention de l'utilisateur est attirée sur les risques        ##
## associés au chargement,  à l'utilisation,  à la modification et/ou au        ##
## développement et à la reproduction du logiciel par l'utilisateur étant       ##
## donné sa spécificité de logiciel libre, qui peut le rendre complexe à        ##
## manipuler et qui le réserve donc à des développeurs et des professionnels    ##
## avertis possédant  des  connaissances  informatiques approfondies.  Les      ##
## utilisateurs sont donc invités à charger  et  tester  l'adéquation  du       ##
## logiciel à leurs besoins dans des conditions permettant d'assurer la         ##
## sécurité de leurs systèmes et ou de leurs données et, plus généralement,     ##
## à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.           ##
##                                                                              ##
## Le fait que vous puissiez accéder à cet en-tête signifie que vous avez       ##
## pris connaissance de la licence CeCILL-C, et que vous en avez accepté les    ##
## termes.                                                                      ##
##                                                                              ##
##################################################################################
##################################################################################

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

def print_time_ns_params( clim , time , model = "multi" , digit = 3 , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.print_time_ns_params
	===============================
	Print in a string a tabular sumarizing ns_params
	
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
	if verbose: print( "Print time ns_params" , end = "\r" )
	## Parameters
	time = np.array([time]).squeeze()
	ql = ci / 2
	qu = 1 - ci / 2
	
	## NS params along time
	l_params = [k for k in clim.ns_law.lparams]
	s_params = xr.DataArray( np.zeros( (clim.n_sample+1,2,len(l_params)) ) , dims = ["sample","forcing","params"] , coords = [clim.X.sample,["all","nat"],l_params] )
	
	for s in s_params.sample:
		clim.ns_law.set_params(clim.ns_params.loc[:,s,model].values)
		for f in s_params.forcing:
			clim.ns_law.set_covariable( clim.X.loc[:,s,f,model].values , clim.time )
			for p in l_params:
				s_params.loc[s,f,p] = clim.ns_law.lparams[p](time).squeeze()
	
	qs_params = s_params[1:,:,:].quantile( [ ql , qu , 0.5 ] , dim = "sample" ).assign_coords( quantile = ["ql","qu","me"] )
	
	## Values of ns_params
	
	tab = tt.Texttable()
	tab.set_precision(digit)
	tab.set_cols_align(["l", "r", "r", "r", "r"])
	tab.set_cols_dtype( ["t" ,"e" , "e","e" , "e"])
	
	tab.header( ["Params {}".format(time),"Best estimate","Quantile {}".format(ql),"Median","Quantile {}".format(qu)])
	
	for p in l_params:
		for nf,f in zip(["F","C"],["all","nat"]):
			tab.add_row( [ "{} ({})".format(p,nf) , float(s_params.loc["be",f,p]) ] + [ float(qs_params.loc[q,f,p]) for q in ["ql","me","qu"] ] )
	
	tab.add_row( [ "" for _ in range(5) ] )
	
	## value of coef
	qns_params = clim.ns_params[:,1:,:].quantile( [ ql , qu , 0.5 ] , dim = "sample" ).assign_coords( quantile = ["ql","qu","me"] )
	
	for p in qns_params.ns_params:
		link = clim.ns_law.lparams[str(p.values)[:-1]].link
		tab.add_row( [ str(p.values) , link(float(clim.ns_params.loc[p,"be",model])) ] + [link(float(qns_params.loc[q,p,model])) for q in ["ql","me","qu"] ] )
	
	if verbose: print( "Print time ns_params (Done)" )
	return tab.draw() + "\n"
##}}}

def write_package_tabular( clim , event , ofile , model = "multi" , time_future = 2040 , digit = 3 , ci = 0.05 , verbose = False ):##{{{
	if verbose: print( "Write package tabular" , end = "\r" )
	
	with open( ofile , "w" ) as f:
		f.write( str(event) + "\n\n" )
		f.write( print_time_stats( clim , event.time  , model = model , digit = digit , ci = ci , verbose = False ) + "\n" )
		f.write( print_time_stats( clim , time_future , model = model , digit = digit , ci = ci , verbose = False ) + "\n" )
		f.write( print_time_ns_params( clim , event.time , model = model , digit = digit , ci = ci , verbose = False ) + "\n" )
		f.write( print_relative_time_stats( clim , time_future , event.time , model = model , digit = digit , ci = ci , verbose = False ) + "\n" )
	
	if verbose: print( "Write package tabular (Done)" , end = "\n" )
##}}}

