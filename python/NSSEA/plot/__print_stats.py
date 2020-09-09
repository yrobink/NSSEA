
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

from NSSEA.__tools import ProgressBar
from NSSEA.__nsstats import add_FAR
from NSSEA.__nsstats import add_return_time


###############
## Functions ##
###############

def summary_table( clim , t0 , model = "Multi_Synthesis" , t1 = None , digit = 3 , ci = 0.05 , output = None , verbose = False ):##{{{
	"""
	NSSEA.plot.summary_table
	========================
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
	t0      : time type
		The time to print the statistics
	model   : string
		The model chosen (default is Multi_Synthesis)
	t1      : time type or None
		If t1 is not None, two lines are added containing the statitstics PR(t1) / PR(t0) and dI(t1) - dI(t0)
	digit   : float
		number of digits (default is 3)
	ci      : float
		Level of confidence interval, default is 0.05 (95%)
	output  : string or None
		If None, the table is returned, else the table is written in the file output.
	verbose : bool
		Print begin / end of execution
	
	Return
	------
	tab : string
		A tabular of statistics
	"""
	
	pb = ProgressBar( 2 , "summary_table" , verbose = verbose )
	
	## Add FAR and Rt
	if "FAR" not in clim.statistics.stats:
		clim = add_FAR( clim , verbose = False )
	if "RtF" not in clim.statistics.stats:
		clim = add_return_time( clim , verbose = False )
	
	## Statistics
	S = clim.statistics.loc[t0,:,:,model]
	if t1 is not None:
		S1 = xr.zeros_like(S.loc[:,["PR","dI"]])
		S1.loc[:,"PR"] = clim.statistics.loc[t1,:,"PR",model] / S.loc[:,"PR"]
		S1.loc[:,"dI"] = clim.statistics.loc[t1,:,"dI",model] - S.loc[:,"dI"]
	
	pb.print()
	
	## Find quantile
	ql = ci /2
	qm = 0.5
	qu = 1 - ci / 2
	qS = S[1:,:].quantile( [ ql,qm,qu ] , dim = "sample" ).assign_coords( quantile = ["ql","qm","qu"] )
	
	if t1 is not None:
		qS1 = S1[1:,:].quantile( [ql,qm,qu] , dim = "sample" ).assign_coords( quantile = ["ql","qm","qu"] )
	
	pb.print()
	
	tab = tt.Texttable()
	tab.set_precision(digit)
	tab.set_cols_align(["l", "r", "r", "r", "r"])
	tab.set_cols_dtype( ["t" ,"e" , "e","e" , "e"])
	
	tab.header( ["Stats {}".format(t0),"Best estimate","Quantile {}".format(ql),"Median","Quantile {}".format(qu)] )
	for s in ["FAR","PR","dI","RtF","RtC","pF","pC","IF","IC"]:
		tab.add_row( [ s , float(S.loc["BE",s]) , float(qS.loc["ql",s]) , float(qS.loc["qm",s]), float(qS.loc["qu",s]) ] )
	
	if t1 is not None:
		for s in ["PR","dI"]:
			tab.add_row( [ "{} {}/{}".format(s,t1,t0) , float(S1.loc["BE",s]) , float(qS1.loc["ql",s]) , float(qS1.loc["qm",s]) , float(qS1.loc["qu",s]) ] )
	
	pb.end()
	
	## Output
	if output is None:
		return tab.draw() + "\n"
	
	with open( output , "w" ) as f:
		f.write( tab.draw() + "\n" )
	
##}}}

def summary_event( clim , event , model = "Multi_Synthesis" , t1 = None , digit = 3 , ci = 0.05 , output = None , verbose = False ): ##{{{
	"""
	NSSEA.plot.summary_event
	========================
	
	Very similar to NSSEA.plot.summary_table, but print the event and the statistics
	at t0, and if t1 is set, pass t1 to the summary table of t0 and print the table
	for time t1.
	
	Parameters
	----------
	clim    : NSSEA.Climatology
		Climatology fitted
	t0      : time type
		The time to print the statistics
	model   : string
		The model chosen (default is Multi_Synthesis)
	t1      : time type or None
		If t1 is not None, two lines are added containing the statitstics PR(t1) / PR(t0) and dI(t1) - dI(t0)
	digit   : float
		number of digits (default is 3)
	ci      : float
		Level of confidence interval, default is 0.05 (95%)
	output  : string or None
		If None, the table is returned, else the table is written in the file output.
	verbose : bool
		Print begin / end of execution
	
	Return
	------
	tab : string
		A tabular of statistics
	"""
	
	pb = ProgressBar( 1 , "summary_event" , verbose = verbose )
	
	out = str(event)
	out = out + "\n" + summary_table( clim , event.time , model , t1 , digit , ci )
	pb.print()
	
	if t1 is not None:
		out = out + "\n" + summary_table( clim , t1 , model , digit = digit , ci = ci )
	
	out = out + "\n"
	
	pb.end()
	if output is None:
		return out
	
	with open( output , "w" ) as f:
		f.write(out)
##}}}

