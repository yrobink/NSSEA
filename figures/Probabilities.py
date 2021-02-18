# -*- coding: utf-8 -*-

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

import sys,os
import pickle as pk
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as mplgrid

import numpy as np
import pandas as pd
import xarray as xr

import NSSEA as ns
import NSSEA.models as nsm
import NSSEA.plot as nsp


####################
## Paramètres mpl ##
####################

mpl.rcParams['font.size'] = 12
#plt.rc('text',usetex=True)
#plt.rcParams['text.latex.unicode'] = True


###############
## Fonctions ##
###############

####################
## Plot functions ##
####################

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


#############
## Classes ##
#############

##########
## main ##
##########

if __name__ == "__main__":
	
	## Test or not
	##============
	ci = 0.05
	be_is_median = True
	
	## Path
	##=====
	basepath = os.path.dirname(os.path.abspath(__file__))
	pathInp  = os.path.join( basepath , "../examples/output" )
	pathOut  = basepath
	assert(os.path.exists(pathInp))
	assert(os.path.exists(pathOut))
	
	## Load output
	##============
	climN = ns.Climatology.from_netcdf( os.path.join( pathInp , "Normal" , "HW03_climCXCB.nc" ) , nsm.Normal() )
	climG = ns.Climatology.from_netcdf( os.path.join( pathInp , "GEV"    , "HW19_climCXCB.nc" ) , nsm.GEV() )
	
	
	## Plot itself
	##============
	
	
	fs,nrow,ncol = 10,3,2
	fig = plt.figure( figsize = ( fs * 1.5 , 0.4 * fs * nrow ) )
	grid = mplgrid.GridSpec( nrows = nrow , ncols = ncol , height_ratios = [1,0.4,1] )
	
	title = ["2003 French heatwave (Gaussian fit)","2019 French heatwave (GEV fit)"]
	lPR = nsp.LinkPR()
	lp  = nsp.Linkp()
	
	for i,clim in zip(range(2),[climN,climG]):
		stats = clim.statistics
		event = clim.event
		
		## Find impossible values
		##=======================
		nan_idx    = np.logical_and( stats[:,1:,:,:].loc[:,:,"pF",:] == 0 , stats[:,1:,:,:].loc[:,:,"pC",:] == 0 )
		nan_values = nan_idx.sum( dim = "sample" ) / ( stats.sample.size - 1 )
		imp_values = ( stats[:,1:,:,:].loc[:,:,"pC",:] == 0 ).sum( dim = "sample" ) / ( stats.sample.size - 1 )
		
		## Find quantiles
		##===============
		qstats = stats[:,1:,:,:].quantile( [ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample" ).assign_coords( quantile = [ "ql" , "qu" , "BE" ] )
		if not clim.BE_is_median:
			qstats.loc["BE",:,:,:] = stats[:,0,:,:]
		
		## Special case : PR
		##==================
		qstats.loc["qu",:,"PR",:] = qstats.loc["qu",:,"PR",:].where( nan_values < ci , np.inf )
		qstats.loc["ql",:,"PR",:] = qstats.loc["ql",:,"PR",:].where( nan_values < ci , 0      )
		if clim.BE_is_median:
			qstats.loc["BE",:,"PR",:] = qstats.loc["BE",:,"PR",:].where( nan_values < ci , 1 )
		
		## Split into continuous time period
		##==================================
		time_validity = {}
		for m in clim.model:
			time_validity[m] = split_into_valid_time( nan_values.loc[:,m] , ci )
		
		time_valid,time_notvalid = time_validity[m]
		
		ax = fig.add_subplot(grid[0,i])
		ax.plot( stats.time , lp.transform(qstats.loc["BE",:,"pF",m]) , color = "red"  , linestyle = "-" , marker = "" , label = r"$p^\mathrm{F}_t$" )
		ax.plot( stats.time , lp.transform(qstats.loc["BE",:,"pC",m]) , color = "blue" , linestyle = "-" , marker = "" , label = r"$p^\mathrm{C}_t$" )
		ax.fill_between( stats.time , lp.transform(qstats.loc["ql",:,"pF",m]) , lp.transform(qstats.loc["qu",:,"pF",m]) , color = "red"  , alpha = 0.5 )
		ax.fill_between( stats.time , lp.transform(qstats.loc["ql",:,"pC",m]) , lp.transform(qstats.loc["qu",:,"pC",m]) , color = "blue" , alpha = 0.5 )
		ax.legend( loc = "upper left" )
		ax.set_ylim( (lp.min,lp.max) )
		ax.set_yticks( lp.ticks )
		ax.set_yticklabels( lp.labels )
		ax.set_title( title[i]  )
		ax.set_xticks([])
		ax.set_ylabel( "Probability" )
		
		ax2 = fig.add_subplot( grid[0,i] , sharex = ax , frameon = False )
		ax2.yaxis.tick_right()
		ax2.set_yticks( lp.ticks )
		ax2.set_yticklabels( lp.Rtlabels )
		ax2.yaxis.set_label_position( "right" )
		ax2.set_ylabel( "Return Time" , rotation = 270 )
		ax2.set_ylim( (lp.min,lp.max) )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		ax = fig.add_subplot(grid[1,i])
		ax.plot( stats.time , lp.transform(nan_values.loc[:,m]) , color = "red"  , label = r"$p^\mathrm{F}=p^\mathrm{C}=0$" )
		ax.plot( stats.time , lp.transform(imp_values.loc[:,m]) , color = "blue" , label = r"$p^\mathrm{C}=0$" )
		ax.plot( [event.time,event.time] , ylim          , linestyle = "--" , marker = "" , color = "black" )
		ax.set_ylim( (lp.min,lp.max) )
		ax.set_yticks( lp.ticks )
		ax.set_yticklabels( lp.labels )
		ax.set_ylabel( "Probability" )
		xlim = ax.get_xlim()
		ax.hlines( lp.transform(ci) , xlim[0] , xlim[1] , color = "grey" , label = "Threshold" )
		ax.set_xlim(xlim)
		ax.set_xticks([])
		ax.legend( loc = "upper right" )
		
		ax = fig.add_subplot(grid[2,i])
		ax.plot( stats.time , lPR.transform(qstats.loc["BE",:,"PR",m]) , color = "red" , linestyle = "-" , marker = "" )
		for t in time_valid:
			ax.fill_between( t , lPR.transform(qstats.loc["ql",t,"PR",m]) , lPR.transform(qstats.loc["qu",t,"PR",m]) , color = "red" , alpha = 0.5 )
		for t in time_notvalid:
			ax.fill_between( t , lPR.transform(qstats.loc["ql",t,"PR",m]) , lPR.transform(qstats.loc["qu",t,"PR",m]) , color = "red" , alpha = 0.3 )
		ax.set_ylim( (lPR.min,lPR.max) )
		ax.set_yticks( lPR.ticks )
		ax.set_yticklabels( lPR.labels )
		ax.set_xlabel( r"$\mathrm{Time}$" )
		ax.set_ylabel( r"$\mathrm{PR}_t$" )
		ax2 = fig.add_subplot( grid[2,i] , sharex = ax , frameon = False )
		ax2.yaxis.tick_right()
		ax2.set_yticks( lPR.ticks )
		ax2.set_yticklabels( lPR.FARlabels )
		ax2.yaxis.set_label_position( "right" )
		ax2.set_ylabel( r"$\mathrm{FAR}_t$" , rotation = 270 )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.hlines( lPR.transform(1) , xlim[0] , xlim[1] , linestyle = "-"  , color = "black" )
		ax.vlines( event.time       , ylim[0] , ylim[1] , linestyle = "--" , color = "black" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		
	
	plt.tight_layout()
	plt.savefig( os.path.join( pathOut , "Probabilities.png" ) )
	
	
	print("Done")


