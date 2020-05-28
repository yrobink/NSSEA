
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

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from NSSEA.__nsstats import build_params_along_time

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

def ns_params_time( clim , ofile , ns_params = None , time = None , ci = 0.05 , verbose = False ):##{{{
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
	ns_params : xr.DataArray or None
		ns params along time, if None, computed with function NSSEA.build_params_along_time
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
	
	if ns_params is None:
		ns_params = build_params_along_time(clim)
	
	l_params = [k for k in clim.ns_law.lparams]
	qs_params = ns_params[:,1:,:,:,:].quantile( [ ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample" ).assign_coords( quantile = ["ql","qu","me"] )
	
	pdf = mpdf.PdfPages( ofile )
	for m in clim.models:
		
		
		xlim = [time.min(),time.max()]
		deltax = 0.05 * ( xlim[1] - xlim[0] )
		xlim[0] -= deltax
		xlim[1] += deltax
		
		fig = plt.figure( figsize = (12,12) )
		
		
		for i,p in enumerate(qs_params.params):
		
			ax = fig.add_subplot( len(l_params) , 1 , i + 1 )
			ax.plot( time , qs_params.loc["me",:,"all",p,m] , color = "red" )
			ax.fill_between( time , qs_params.loc["ql",:,"all",p,m] , qs_params.loc["qu",:,"all",p,m] , color = "red" , alpha = 0.5 )
			ax.plot( time , qs_params.loc["me",:,"nat",p,m] , color = "blue" )
			ax.fill_between( time , qs_params.loc["ql",:,"nat",p,m] , qs_params.loc["qu",:,"nat",p,m] , color = "blue" , alpha = 0.5 )
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


