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
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr

import NSSEA as ns
import NSSEA.models as nsm


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
	nrow,ncol = 3,2
	fs = 10
	fig = plt.figure( figsize = ( fs * 1.5 , 0.4 * fs * nrow ) )
	
	m = "Multi_Synthesis"
	title = ["2003 French heatwave (Gaussian fit)","2019 French heatwave (GEV fit)"]
	ylimI  = [(20,40),(22,40)]
	ymindI = -1
	ymaxdI = 15
	units = ["(°\mathrm{C})","(°\mathrm{C})"]
	
	axes = [ [1,3,5] , [2,4,6] ]
	
	for i,clim in zip(range(2),[climN,climG]):
		
		event  = clim.event
		ylabel = "\mathrm{(" + event.unit + ")}"
		stats  = clim.statistics
		statsl = stats[:,1:,[2,3,5],:].quantile( ci / 2.      , dim = "sample" )
		statsu = stats[:,1:,[2,3,5],:].quantile( 1. - ci / 2. , dim = "sample" )
		
		ax = fig.add_subplot( nrow , ncol , axes[i][0] )
		ax.plot( stats.time , stats.loc[:,"BE","IF",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"IF",m] , statsu.loc[:,"IF",m] , color = "red" , alpha = 0.5 )
		ax.set_title( title[i] )
		ax.set_xticks([])
		ax.set_ylabel( r"${}$".format( "\mathbf{I}^\mathrm{F}_t\ " + units[i] ) )
		xlim = ax.get_xlim()
		ax.plot( [event.time,event.time] , ylimI[i]       , linestyle = "--" , marker = "" , color = "black" )
		ax.hlines( stats.loc[event.time,"BE","IF",m] , xlim[0] , xlim[1] , color = "black" , linestyle = "--" )
		ax.set_xlim(xlim)
		ax.set_ylim( ylimI[i] )
		
		ax = fig.add_subplot( nrow , ncol , axes[i][1] )
		ax.plot( stats.time , stats.loc[:,"BE","IC",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"IC",m] , statsu.loc[:,"IC",m] , color = "red" , alpha = 0.5 )
		ax.set_xticks([])
		ax.set_ylabel( r"${}$".format( "\mathbf{I}^\mathrm{C}_t\ " + units[i] ) )
		xlim = ax.get_xlim()
		ax.plot( [event.time,event.time] , ylimI[i]       , linestyle = "--" , marker = "" , color = "black" )
		ax.hlines( stats.loc[event.time,"BE","IC",m] , xlim[0] , xlim[-1] , color = "black" , linestyle = "--" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylimI[i])
		
		ax = fig.add_subplot( nrow , ncol , axes[i][2] )
		ax.plot( stats.time , stats.loc[:,"BE","dI",m] , color = "red" , linestyle = "-" , marker = "" )
		ax.fill_between( stats.time , statsl.loc[:,"dI",m] , statsu.loc[:,"dI",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (ymindI,ymaxdI) )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( r"${}$".format( "\Delta\mathbf{I}_t\ " + ylabel ) )
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( [event.time,event.time] , ylim  , linestyle = "--" , marker = "" , color = "black" )
		ax.plot( xlim                    , [0,0] , linestyle = "-"  , marker = "" , color = "black" )
		ax.hlines( stats.loc[event.time,"BE","dI",m] , xlim[0] , xlim[-1] , color = "black" , linestyle = "--" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
	
	fig.set_tight_layout(True)
	plt.savefig( os.path.join( pathOut , "Intensities.png" ) )
	
	
	print("Done")


