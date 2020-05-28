
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

import sys
import numpy as np
import xarray as xr

from .__tools import ProgressBar


###############
## Functions ##
###############

def extremes_stats( clim , event , verbose = False , tol = sys.float_info.epsilon ):##{{{
	"""
	NSSEA.extremes_stats
	====================
	Compute extremes statistics and add it to a Climatology.
	
	Arguments
	---------
	clim : NSSEA.Climatology
		A clim variable
	event  : NSSEA.Event
		An event variable
	verbose: bool
		Print state of execution or not
	tol    : float
		Numerical tolerance, default is sys.float_info.epsilon
	
	Return
	------
	clim : NSSEA.Climatology
		A clim variable with clim.stats set
	
	Statistics computed
	-------------------
	The variable clim.stats is an xarray with dimensions (n_time,n_sample+1,n_stats,n_models), stats available are:
	
	pF: Probability of event.anom at time event.time in factual world
	pC: Probability of event.anom at time event.time in counter factual world
	PR: Probability ratio (pF / pC)
	IF: Event with same probability than the probability of event.anom at each time in factual world
	IC: Event with same probability than the probability of event.anom at each time in counter factual world
	dI: IF - IC
	
	"""
	## Usefull variables
	time           = clim.time
	n_time         = clim.n_time
	models         = clim.models
	n_models       = clim.n_models
	n_sample       = clim.n_sample
	n_stats        = 6
	upper_side     = event.side == "upper"
	event_time     = event.time
	
	
	## Output
	stats = xr.DataArray( np.zeros( (n_time,n_sample + 1,n_stats,n_models) ) , coords = [clim.X.time , clim.X.sample , ["pC","pF","PR","IC","IF","dI"] , clim.X.models ] , dims = ["time","sample","stats","models"] )
	
	## 
	law = clim.ns_law
	pb = ProgressBar( "Statistics" , n_models * (n_sample + 1) )
	for m in clim.X.models:
		for s in clim.X.sample:
			if verbose: pb.print()
			
			law.set_params( clim.ns_params.loc[:,s,m].values )
			
			## Find threshold
			law.set_covariable( clim.X.loc[:,s,"all",m].values , time )
			if event.type_event == "threshold":
				threshold = np.zeros(n_time) + np.mean( law.meant(event.reference) ) + event.anomaly
			else:
				threshold = np.zeros(n_time) + event.anomaly
			
			## Find pF
			stats.loc[:,s,"pF",m] = law.sf( threshold , time ) if upper_side else law.cdf( threshold , time )
			
			## Find probability of the event in factual world
			pf = np.zeros(n_time) + ( law.sf( np.array([threshold[0]]) , np.array([event.time]) ) if upper_side else law.cdf( np.array([threshold[0]]) , np.array([event.time]) ) )
			
			## I1
			stats.loc[:,s,"IF",m] = law.isf( pf , time ) if upper_side else law.icdf( pf , time )
			
			## Find pC
			law.set_covariable( clim.X.loc[:,s,"nat",m].values , time )
			stats.loc[:,s,"pC",m] = law.sf( threshold , time ) if upper_side else law.cdf( threshold , time )
			
			## I0
			stats.loc[:,s,"IC",m] = law.isf( pf , time ) if upper_side else law.icdf( pf , time )
	
	
	## RR
	stats.loc[:,:,"PR",:] = stats.loc[:,:,"pF",:] / stats.loc[:,:,"pC",:]
#	stats.loc[:,:,"PR",:] = stats.loc[:,:,"PR",:].where( stats.loc[:,:,"PR",:] > 0 , np.inf )
	
	
	## deltaI
	stats.loc[:,:,"dI",:] = stats.loc[:,:,"IF",:] - stats.loc[:,:,"IC",:]
	
	clim.stats = stats
	if verbose: pb.end()
	
	return clim
##}}}

def PR_correction( S , tol = 1e-10 ):##{{{
	
	## 
	idx_all = S.loc[:,:,"pF",:] < tol
	idx_nat = S.loc[:,:,"pC",:] < tol
	cidx_all = np.logical_not(idx_all)
	cidx_nat = np.logical_not(idx_nat)
	
	S.loc[:,:,"pF",:] = S.loc[:,:,"pF",:].where( cidx_all , np.nan )
	S.loc[:,:,"pC",:] = S.loc[:,:,"pC",:].where( cidx_nat , np.nan )
	
	## Remove all problematic values from RR
	idx = np.logical_and( cidx_all , cidx_nat )
	S.loc[:,:,"PR",:]   = S.loc[:,:,"PR",:].where( idx , np.nan )
	
	## pF > 0, pC > 0, Don't touch!!!
	
	## pF > 0, pC = 0
	idx = np.logical_not( np.logical_and( cidx_all , idx_nat ) )
	S.loc[:,:,"PR",:]   = S.loc[:,:,"PR",:].where( idx , np.Inf )
	
	## pF = 0, pC > 0
	idx = np.logical_not( np.logical_and( idx_all , cidx_nat ) )
	S.loc[:,:,"PR",:]   = S.loc[:,:,"PR",:].where( idx , 0. )
	
	## pF = 0, pC = 0, here we replace by 1
	idx = np.logical_not( np.logical_and( idx_all , idx_nat ) )
	S.loc[:,:,"PR",:] = S.loc[:,:,"PR",:].where( idx , 1 )
	
	## 
	S.loc[:,:,"pF",:] = S.loc[:,:,"pF",:].where( cidx_all , 0 )
	S.loc[:,:,"pC",:] = S.loc[:,:,"pC",:].where( cidx_nat , 0 )
	
	return S
##}}}

def stats_relative_event( stats , time_event ):##{{{
	"""
	NSSEA.stats_relative_event
	==========================
	Compute relative statistics at a fixed time
	
	Arguments
	---------
	stats     : xarray
		clim.stats
	time_event: Time of event
	
	Return
	------
	satstEvent: xarray
		Similar to clim.stats
	"""
	statsEvent = xr.zeros_like(stats)
	statsEvent[:,:,:3,:] = stats[:,:,:3,:] / stats.loc[time_event,:,["pF","pC","PR"],:]
	statsEvent[:,:,3:,:] = stats[:,:,3:,:] - stats.loc[time_event,:,["IF","IC","dI"],:]
	return statsEvent
##}}}

def build_params_along_time( clim , verbose = False ):##{{{
	"""
	NSSEA.extremes_stats
	====================
	Build trajectories of ns_params alon time
	
	Arguments
	---------
	clim : NSSEA.Climatology
		A clim variable
	verbose: bool
		Print state of execution or not
	
	Return
	------
	ns_params : xr.DataArray
		An array containing ns_params along time
	
	"""
	ns_law = clim.ns_law
	
	l_params = [k for k in clim.ns_law.lparams]
	xrdims   = ["time","sample","forcing","params","models"]
	xrcoords = [clim.time,clim.X.sample,["all","nat"],l_params,clim.models]
	s_params = xr.DataArray( np.zeros( (clim.n_time,clim.n_sample+1,2,len(l_params),clim.n_models) ) , dims = xrdims , coords = xrcoords )
	
	
	pb = ProgressBar( "build_params_along_time" , clim.n_models * (clim.n_sample + 1) )
	for m in clim.models:
		for s in s_params.sample:
			if verbose: pb.print()
			
			clim.ns_law.set_params( clim.ns_params.loc[:,s,m].values )
			for f in s_params.forcing:
				clim.ns_law.set_covariable( clim.X.loc[clim.time,s,f,m].values , clim.time )
				for p in l_params:
					s_params.loc[:,s,f,p,m] = clim.ns_law.lparams[p](clim.time)
	
	
	if verbose: pb.end()
	
	return s_params
##}}}

def add_return_time( climIn ):##{{{
	"""
	NSSEA.add_return_time
	=====================
	Add return time to statistics computed (require that NSSEA.extremes_stats has been previously called)
	A copy is returned.
	
	Arguments
	---------
	climIn : NSSEA.Climatology
		A clim variable
	
	Return
	------
	clim : NSSEA.Climatology
		A clim variable with return time
	
	"""
	clim = climIn.copy()
	
	xrdims   = ["time","sample","stats","models"]
	xrcoords = [clim.time,clim.stats.sample,["RtF","RtC"],clim.models]
	Rt = xr.DataArray( np.zeros( (clim.n_time,clim.n_sample+1,2,clim.n_models) ) , dims = xrdims , coords = xrcoords )
	
	Rt.loc[:,:,:,:,] = 1. / clim.stats.loc[:,:,["pF","pC"]].values
	
	clim.stats = xr.concat( (clim.stats,Rt) , dim = "stats" )
	
	return clim
##}}}

def add_FAR( climIn ):##{{{
	"""
	NSSEA.add_FAR
	=============
	Add FAR (Fraction of Attribuable Risk = 1 - 1 / PR) to statistics computed (require that NSSEA.extremes_stats has been previously called)
	A copy is returned.
	
	Arguments
	---------
	climIn : NSSEA.Climatology
		A clim variable
	
	Return
	------
	clim : NSSEA.Climatology
		A clim variable with FAR
	
	"""
	clim = climIn.copy()
	
	xrdims   = ["time","sample","stats","models"]
	xrcoords = [clim.time,clim.stats.sample,["FAR"],clim.models]
	FAR = xr.DataArray( np.zeros( (clim.n_time,clim.n_sample+1,1,clim.n_models) ) , dims = xrdims , coords = xrcoords )
	
	FAR.loc[:,:,"FAR",:] =  1. - 1. / clim.stats.loc[:,:,"PR",:].values
	
	clim.stats = xr.concat( (clim.stats,FAR) , dim = "stats" )
	
	return clim
##}}}


