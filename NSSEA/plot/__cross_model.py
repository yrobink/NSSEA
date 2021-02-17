
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
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from ..__tools import ProgressBar
from .__link import LinkPR
from .__link import Linkp


###############
## Functions ##
###############

def statistics_time( l_clim , ofile , time = None , labels = None , colors = None , ci = 0.05 , verbose = False ): ##{{{
	
	pb = ProgressBar( 7 , "plot.statistics_time" , verbose )
	
	## Into list
	if type(l_clim) is not list:
		l_clim = [l_clim]
	
	## time
	if time is None:
		time = l_clim[0].event.time
	
	## Find quantile
	l_qstats = [ clim.statistics[:,1:,:,:].loc[time,:,:,:].quantile( [ ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample" ).assign_coords( quantile = ["ql","qu","BE"] ) for clim in l_clim ]
	for i,clim in enumerate(l_clim):
		if not clim.BE_is_median:
			l_qstats[i].loc["BE",:,:] = clim.statistics.loc[time,"BE",:,:]
	pb.print()
	
	## Find lim
	var  = ["pC","pF","IC","IF","PR","dI"]
	xmin = l_clim[0].statistics.loc[time,:,var,:].min( dim = ["sample","model"] )
	xmax = l_clim[0].statistics.loc[time,:,var,:].min( dim = ["sample","model"] )
	for clim in l_clim:
		nxmin = clim.statistics.loc[time,:,var,:].min( dim = ["sample","model"] )
		nxmax = clim.statistics.loc[time,:,var,:].max( dim = ["sample","model"] )
		xmin  = xmin.where( xmin < nxmin , nxmin )
		xmax  = xmax.where( xmax > nxmax , nxmax )
	delta = 0.1 * (xmax - xmin)
	xmin -= delta
	xmax += delta
	
	## Link
	lp  = Linkp()
	lPR = LinkPR()
	
	## Matplotlib
	pltkwargs = { "vert" : False , "showmeans" : False , "showextrema" : False , "showmedians" : False }
	if labels is not None:
		ylabels = labels
	else:
		ylabels = []
		for clim in l_clim:
			ylabels = ylabels + [ " ".join(m.split("_") ) for m in clim.model ]
	
	if colors is None:
		colors = plt.cm.brg( np.linspace( 0 , 1 , len(l_clim) ) )
	
	fig = plt.figure( figsize = (20,12) )
	
	
	## Subplot 11
	ax  = fig.add_subplot(2,3,1)
	posb,pose = 0,0
	for color,qstats,clim in zip(colors,l_qstats,l_clim):
		pose = posb + clim.n_model
		vplot = ax.violinplot( lp.transform(clim.statistics[:,1:,:,:].loc[time,:,"pF",:].values) , positions = range(posb,pose) , **pltkwargs )
		for pc in vplot["bodies"]:
			pc.set_facecolor(color)
			pc.set_edgecolor(color)
			pc.set_alpha(0.5)
		for p,m in zip(range(posb,pose),clim.model):
			ax.vlines( lp.transform(qstats.loc[:,"pF",m]) , p - 0.3 , p + 0.3 , color = color )
		posb += clim.n_model
	ax.set_xticks(lp.ticks)
	ax.set_xticklabels(lp.labels)
	ax.set_xlim( lp.transform([xmin.loc[["pF","pC"]].min(),xmax.loc[["pF","pC"]].max()]) )
	xlim = ax.get_xlim()
	for i in range(pose-1):
		ax.hlines( i + 0.5 , xlim[0] , xlim[1] , color = "grey" )
	ax.set_xlim(xlim)
	ax.set_yticks(range(pose))
	ax.set_yticklabels( ylabels )
	ax.set_ylim((-0.5,pose-0.5))
	ax.set_xlabel( r"$p^\mathrm{F}_{" + str(time) + "}$" )
	pb.print()
	
	## Subplot 12
	ax  = fig.add_subplot(2,3,2)
	posb,pose = 0,0
	for color,qstats,clim in zip(colors,l_qstats,l_clim):
		pose = posb + clim.n_model
		vplot = ax.violinplot( clim.statistics[:,1:,:,:].loc[time,:,"IF",:].values , positions = range(posb,pose) , **pltkwargs )
		for pc in vplot["bodies"]:
			pc.set_facecolor(color)
			pc.set_edgecolor(color)
			pc.set_alpha(0.5)
		for p,m in zip(range(posb,pose),clim.model):
			ax.vlines( qstats.loc[:,"IF",m] , p - 0.3 , p + 0.3 , color = color )
		posb += clim.n_model
	ax.set_xlim( [xmin.loc[["IF","IC"]].min(),xmax.loc[["IF","IC"]].max()] )
	xlim = ax.get_xlim()
	for i in range(pose-1):
		ax.hlines( i + 0.5 , xlim[0] , xlim[1] , color = "grey" )
	ax.set_xlim(xlim)
	ax.set_yticks([])
	ax.set_ylim((-0.5,pose-0.5))
	ax.set_xlabel( r"$\mathbf{I}^\mathrm{F}_{" + str(time) + "}$" )
	pb.print()
	
	## Subplot 21
	ax  = fig.add_subplot(2,3,4)
	posb,pose = 0,0
	for color,qstats,clim in zip(colors,l_qstats,l_clim):
		pose = posb + clim.n_model
		vplot = ax.violinplot( lp.transform(clim.statistics[:,1:,:,:].loc[time,:,"pC",:].values) , positions = range(posb,pose) , **pltkwargs )
		for pc in vplot["bodies"]:
			pc.set_facecolor(color)
			pc.set_edgecolor(color)
			pc.set_alpha(0.5)
		for p,m in zip(range(posb,pose),clim.model):
			ax.vlines( lp.transform(qstats.loc[:,"pC",m]) , p - 0.3 , p + 0.3 , color = color )
		posb += clim.n_model
	ax.set_xticks(lp.ticks)
	ax.set_xticklabels(lp.labels)
	ax.set_xlim( lp.transform([xmin.loc[["pF","pC"]].min(),xmax.loc[["pF","pC"]].max()]) )
	xlim = ax.get_xlim()
	for i in range(pose-1):
		ax.hlines( i + 0.5 , xlim[0] , xlim[1] , color = "grey" )
	ax.set_xlim(xlim)
	ax.set_yticks(range(pose))
	ax.set_yticklabels( ylabels )
	ax.set_ylim((-0.5,pose-0.5))
	ax.set_xlabel( r"$p^\mathrm{C}_{" + str(time) + "}$" )
	pb.print()
	
	## Subplot 22
	ax  = fig.add_subplot(2,3,5)
	posb,pose = 0,0
	for color,qstats,clim in zip(colors,l_qstats,l_clim):
		pose = posb + clim.n_model
		vplot = ax.violinplot( clim.statistics[:,1:,:,:].loc[time,:,"IC",:].values , positions = range(posb,pose) , **pltkwargs )
		for pc in vplot["bodies"]:
			pc.set_facecolor(color)
			pc.set_edgecolor(color)
			pc.set_alpha(0.5)
		for p,m in zip(range(posb,pose),clim.model):
			ax.vlines( qstats.loc[:,"IC",m] , p - 0.3 , p + 0.3 , color = color )
		posb += clim.n_model
	ax.set_xlim( [xmin.loc[["IF","IC"]].min(),xmax.loc[["IF","IC"]].max()] )
	xlim = ax.get_xlim()
	for i in range(pose-1):
		ax.hlines( i + 0.5 , xlim[0] , xlim[1] , color = "grey" )
	ax.set_xlim(xlim)
	ax.set_yticks([])
	ax.set_ylim((-0.5,pose-0.5))
	ax.set_xlabel( r"$\mathbf{I}^\mathrm{C}_{" + str(time) + "}$" )
	pb.print()
	
	## Subplot 13
	ax  = fig.add_subplot(2,3,3)
	posb,pose = 0,0
	for color,qstats,clim in zip(colors,l_qstats,l_clim):
		pose = posb + clim.n_model
		vplot = ax.violinplot( lPR.transform(clim.statistics[:,1:,:,:].loc[time,:,"PR",:].values) , positions = range(posb,pose) , **pltkwargs )
		for pc in vplot["bodies"]:
			pc.set_facecolor(color)
			pc.set_edgecolor(color)
			pc.set_alpha(0.5)
		for p,m in zip(range(posb,pose),clim.model):
			ax.vlines( lPR.transform(qstats.loc[:,"PR",m]) , p - 0.3 , p + 0.3 , color = color )
		posb += clim.n_model
	ax.set_xticks(lPR.ticks)
	ax.set_xticklabels(lPR.labels)
#	ax.set_xlim( lPR.transform([xmin.loc["PR"],xmax.loc["PR"]]) )
	xlim = ax.get_xlim()
	for i in range(pose-1):
		ax.hlines( i + 0.5 , xlim[0] , xlim[1] , color = "grey" )
	ax.set_xlim(xlim)
	ax.set_yticks(range(pose))
	ax.set_yticklabels( [] )
	ax.set_ylim((-0.5,pose-0.5))
	ax.set_xlabel( r"$\mathrm{PR}_{" + str(time) + "}$" )
	pb.print()
	
	## Subplot 23
	ax  = fig.add_subplot(2,3,6)
	posb,pose = 0,0
	for color,qstats,clim in zip(colors,l_qstats,l_clim):
		pose = posb + clim.n_model
		vplot = ax.violinplot( clim.statistics[:,1:,:,:].loc[time,:,"dI",:].values , positions = range(posb,pose) , **pltkwargs )
		for pc in vplot["bodies"]:
			pc.set_facecolor(color)
			pc.set_edgecolor(color)
			pc.set_alpha(0.5)
		for p,m in zip(range(posb,pose),clim.model):
			ax.vlines( qstats.loc[:,"dI",m] , p - 0.3 , p + 0.3 , color = color )
		posb += clim.n_model
	xlim = ax.get_xlim()
	for i in range(pose-1):
		ax.hlines( i + 0.5 , xlim[0] , xlim[1] , color = "grey" )
	ax.set_xlim(xlim)
	ax.set_yticks([])
	ax.set_ylim((-0.5,pose-0.5))
	ax.set_xlabel( r"$\Delta\mathbf{I}_{" + str(time) + "}$" )
	pb.print()
	
	fig.set_tight_layout(True)
	plt.savefig(ofile)
	
	pb.end()
##}}}

