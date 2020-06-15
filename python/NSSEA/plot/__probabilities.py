
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

from .__linkParams import LinkParams


###############
## Functions ##
###############


def split_into_valid_time( nan_values , ci ):##{{{
	time_valid    = []
	time_notvalid = []
	t = int(nan_values.time[0])
	on_valid      = bool(nan_values.loc[t] < ci)
	curr = [t]
	
	for t in nan_values.time[1:]:
		curr.append(int(t))
		is_valid = bool(nan_values.loc[t] < ci)
		if not is_valid == on_valid:
			if on_valid: time_valid.append(curr)
			else: time_notvalid.append(curr)
			on_valid = not on_valid
			curr = [int(t)]
	if on_valid: time_valid.append(curr)
	else: time_notvalid.append(curr)
	
	return time_valid,time_notvalid
##}}}

def probabilities( clim , event , ofile , be_is_median = False , ci = 0.05 , verbose = False ):##{{{
	stats = clim.stats
	
	## Find impossible values
	##=======================
	nan_idx    = np.logical_and( stats[:,1:,:,:].loc[:,:,"pF",:] == 0 , stats[:,1:,:,:].loc[:,:,"pC",:] == 0 )
	nan_values = nan_idx.sum( dim = "sample" ) / ( stats.sample.size - 1 )
	imp_values = ( stats[:,1:,:,:].loc[:,:,"pC",:] == 0 ).sum( dim = "sample" ) / ( stats.sample.size - 1 )
	
	## Find quantiles
	##===============
	qstats = stats[:,1:,:,:].quantile( [ci / 2 , 1 - ci / 2 , 0.5 , 0.5 , 0.5 ] , dim = "sample" ).assign_coords( quantile = [ "ql" , "qu" , "be" , "bel" , "beu" ] )
	if not be_is_median: qstats.loc["be",:,:,:] = stats[:,0,:,:]
	
	## Special case : PR
	##==================
	qstats.loc["qu",:,"PR",:] = qstats.loc["qu",:,"PR",:].where( nan_values < ci , np.inf )
	qstats.loc["ql",:,"PR",:] = qstats.loc["ql",:,"PR",:].where( nan_values < ci , 0      )
	if be_is_median:
		qstats.loc["be",:,"PR",:] = qstats.loc["be",:,"PR",:].where( nan_values < ci , 1 )
	
	## Split into continuous time period
	##==================================
	time_valid,time_notvalid = split_into_valid_time( nan_values , ci )
	
	## Main plot
	##==========
	lp = LinkParams()
	
	pdf = mpdf.PdfPages( ofile )
	
	for m in stats.models:
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_axes( [0.08,0.67,0.85,0.3] )
		ax.plot( stats.time , lp.fp(qstats.loc["be",:,"pF",m]) , color = "red"  , linestyle = "-" , marker = "" , label = r"$p_\mathrm{F}(t)$" )
		ax.plot( stats.time , lp.fp(qstats.loc["be",:,"pC",m]) , color = "blue" , linestyle = "-" , marker = "" , label = r"$p_\mathrm{C}(t)$" )
		ax.fill_between( stats.time , lp.fp(qstats.loc["ql",:,"pF",m]) , lp.fp(qstats.loc["qu",:,"pF",m]) , color = "red" , alpha = 0.5 )
		ax.fill_between( stats.time , lp.fp(qstats.loc["ql",:,"pC",m]) , lp.fp(qstats.loc["qu",:,"pC",m]) , color = "blue" , alpha = 0.5 )
		ax.legend( loc = "upper left" )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_xticks([])
		ax.set_ylabel( "Probabilities" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		ax = fig.add_axes( [0.08,0.50,0.85,0.15] )
		ax.plot( stats.time , lp.fp(nan_values.loc[:,m]) , color = "red"  , label = r"$p_\mathrm{F}=p_\mathrm{C}=0$" )
		ax.plot( stats.time , lp.fp(imp_values.loc[:,m]) , color = "blue" , label = r"$p_\mathrm{C}=0$" )
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_ylabel( "Probabilities" )
		xlim = ax.get_xlim()
		ax.hlines( lp.fp(ci) , xlim[0] , xlim[1] , color = "grey" , label = "Threshold" )
		ax.set_xlim(xlim)
		ax.set_xticks([])
		ax.legend( loc = "upper right" )
		
		ax = fig.add_axes( [0.08,0.08,0.85,0.4] )
		ax.plot( stats.time , lp.frr(qstats.loc["be",:,"PR",m]) , color = "red" , linestyle = "-" , marker = "" )
		for t in time_valid:
			ax.fill_between( t , lp.frr(qstats.loc["ql",t,"PR",m]) , lp.frr(qstats.loc["qu",t,"PR",m]) , color = "red" , alpha = 0.5 )
		for t in time_notvalid:
			ax.fill_between( t , lp.frr(qstats.loc["ql",t,"PR",m]) , lp.frr(qstats.loc["qu",t,"PR",m]) , color = "red" , alpha = 0.3 )
		ax.set_ylim( (lp.rr.values.min(),lp.rr.values.max()) )
		ax.set_yticks( lp.rr.values )
		ax.set_yticklabels( lp.rr.names )
		ax.set_xlabel( r"$\mathrm{Time}$" )
		ax.set_ylabel( r"$\mathrm{PR}(t)$" )
		ax2 = fig.add_axes( [0.08,0.08,0.85,0.4] , sharex = ax , frameon = False )
		ax2.yaxis.tick_right()
		ax2.set_yticks( lp.rr.values )
		ax2.set_yticklabels( lp.far.names )
		ax2.yaxis.set_label_position( "right" )
		ax2.set_ylabel( r"$\mathrm{FAR}(t)$" )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( xlim                    , lp.frr([1,1]) , linestyle = "-"  , marker = "" , color = "black" )
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
##}}}

def probabilities_not_zero( clim , event , ofile , ci = 0.05 , verbose = False ): ##{{{
	"""
	NSSEA.plot.probabilities_not_zero
	=================================
	
	Plot probabilities (pF,pC,PR) along time, but assume than pC,pF > 0
	
	Arguments
	---------
	clim      : NSSEA.Climatology
		Climatology with stats computed
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot probabilities" , end = "\r" )
	
	stats = clim.stats
	statsl = stats[:,1:,:3,:].quantile( ci / 2.      , dim = "sample" )
	statsu = stats[:,1:,:3,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	
	lp = LinkParams()
	
	pdf = mpdf.PdfPages( ofile )
	
	for m in stats.models:
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( stats.time , lp.fp(stats.loc[:,"be","pF",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.fp(statsl.loc[:,"pF",m]) , lp.fp(statsu.loc[:,"pF",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_title( "{}".format( str(m.values).replace("_"," ") ) )
		ax.set_xticks([])
		ax.set_ylabel( r"$p_\mathrm{F}(t)$" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( stats.time , lp.fp(stats.loc[:,"be","pC",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.fp(statsl.loc[:,"pC",m]) , lp.fp(statsu.loc[:,"pC",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.p.values.min(),lp.p.values.max()) )
		ax.set_yticks( lp.p.values )
		ax.set_yticklabels( lp.p.names )
		ax.set_xticks([])
		ax.set_ylabel( r"$p_\mathrm{C}(t)$" )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
		ax = fig.add_subplot( nrow , ncol , 3 )
		ax.plot( stats.time , lp.frr(stats.loc[:,"be","PR",m]) , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , lp.frr(statsl.loc[:,"PR",m]) , lp.frr(statsu.loc[:,"PR",m]) , color = "red" , alpha = 0.5 )
		ax.set_ylim( (lp.rr.values.min(),lp.rr.values.max()) )
		ax.set_yticks( lp.rr.values )
		ax.set_yticklabels( lp.rr.names )
		ax.set_xlabel( r"$\mathrm{Time}$" )
		ax.set_ylabel( r"$\mathrm{PR}(t)$" )
		ax2 = fig.add_subplot( nrow , ncol , 3 , sharex = ax , frameon = False )
		ax2.yaxis.tick_right()
		ax2.set_yticks( lp.rr.values )
		ax2.set_yticklabels( lp.far.names )
		ax2.yaxis.set_label_position( "right" )
		ax2.set_ylabel( r"$\mathrm{FAR}(t)$" )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( xlim                    , lp.frr([1,1]) , linestyle = "-"  , marker = "" , color = "black" )
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	if verbose: print( "Plot probabilities (Done)" )
##}}}

def return_time( clim , event , ofile , be_is_median = False , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.probabilities
	========================
	
	Plot return time (RtF,RtC) along time
	
	Arguments
	---------
	clim      : NSSEA.Climatology
		Climatology with stats computed
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	be_is_median : boolean
		If we assume than the true best estimate is the median of bootstrap (usefull for Bayesian)
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot return time" , end = "\r" )
	
	## Compute quantiles and best estimate
	##====================================
	qRt = clim.stats[:,1:,:,:].loc[:,:,["RtF","RtC"],:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample" ).assign_coords( quantile = ["ql","qu","med"] )
	if not be_is_median: qRt.loc["med",:,:,:] = clim.stats.loc[:,"be",["RtF","RtC"],:]
	
	## Some parameters
	##================
	lp = np.log10
	pdf = mpdf.PdfPages( ofile )
	
	for m in qRt.models:
		
		fig = plt.figure( figsize = (8,6) )
		
		ax = fig.add_subplot( 1 , 1 , 1 )
		
		ax.plot( qRt.time , lp(qRt.loc["med",:,"RtF",m]) , color = "red" , label = r"Rt$^F$" )
		ax.fill_between( qRt.time , lp(qRt.loc["ql",:,"RtF",m]) , lp(qRt.loc["qu",:,"RtF",m]) , color = "red" , alpha = 0.5 )
		
		ax.plot( qRt.time , lp(qRt.loc["med",:,"RtC",m]) , color = "blue" , label = r"Rt$^C$" )
		ax.fill_between( qRt.time , lp(qRt.loc["ql",:,"RtC",m]) , lp(qRt.loc["qu",:,"RtC",m]) , color = "blue" , alpha = 0.5 )
		
		
		imax = lp(float(qRt.loc["qu",:,:,m].max()))
		if np.isfinite(imax): imax = int(imax) + 1
		else: imax = 6
		
		ax.set_yticks( range(imax) )
		ax.set_yticklabels( [ 10**i for i in range(imax) ] )
		ylim = (0,imax)
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_ylim(ylim)
		
		
		ax.set_xlabel( "Time" )
		ax.set_ylabel( "Return time" )
		ax.legend()
		ax.set_title( m.values )
		
		pdf.savefig(fig)
		plt.close(fig)
	
	pdf.close()
	if verbose: print( "Plot return time (Done)" )
##}}}

