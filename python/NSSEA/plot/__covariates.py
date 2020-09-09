
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
#mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from NSSEA.__tools import ProgressBar


###############
## Functions ##
###############

def GAM_decomposition( clim , lX , event , ofile , ci = 0.05 , verbose = False ): ##{{{
	"""
	NSSEA.plot.GAM_decomposition
	============================
	
	Plot the GAM decomposition of covariates
	
	Arguments
	---------
	Xd        : pandas.DataFrame
		Covariates before decomposition
	X         : xarray
		Covariates afer decomposition (NSSEA.Climatology.X)
	event     : NSSEA.Event
		Event variable
	ofile     : str
		output file
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	pb = ProgressBar( clim.n_model , "GAM_decomposition" , verbose )
	
	X = clim.X.copy()
	XA = ( X.loc[:,:,"F",:] - X.loc[:,:,"C",:] ).assign_coords( forcing = "A" )
	X = xr.concat( [X,XA] , dim = "forcing" )
	
	Xl = X[:,1:,:,:].quantile( ci / 2.      , dim = "sample" )
	Xu = X[:,1:,:,:].quantile( 1. - ci / 2. , dim = "sample" )
	
	pdf = mpdf.PdfPages( ofile )
	
	## Remove multi
	models = clim.model
	if "Multi_Synthesis" in models:
		models = models[:-1]
	
	ymin = min( float(X.loc[:,:,["F","C"],:].min()) , min([ float(lx.min()) for lx in lX]) )
	ymax = max( float(X.loc[:,:,["F","C"],:].max()) , max([ float(lx.max()) for lx in lX]) )
	
	yminAnt = float(X.loc[:,:,"A",:].min())
	ymaxAnt = float(X.loc[:,:,"A",:].max())
	
	ylabel = r"${}$".format( "\mathrm{" + event.name_variable +"}\ \mathrm{(" + event.unit_variable + ")}" )
	
	## Loop
	for i,m in enumerate(models):
		pb.print()
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( lX[i].index , lX[i].values.ravel() , color = "grey" , linestyle = "" , marker = "." , alpha = 0.3 , label = "" )
		ax.plot( X.time , X.loc[:,"BE","F",m] , color = "red" , linestyle = "-" , marker = "" , label = r"$\mathrm{Factual}$" )
		ax.fill_between( X.time , Xl.loc[:,"F",m] , Xu.loc[:,"F",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (ymin,ymax) )
		ax.legend( loc = "upper left" )
		ax.set_xticks([])
		ax.set_title( "{}".format( str(m).replace("_"," ") ) )
		ax.set_ylabel( ylabel )
		
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( X.time , X.loc[:,"BE","C",m] , color = "blue" , linestyle = "-" , marker = "" , label = r"$\mathrm{Counter-factual}$" )
		ax.fill_between( X.time , Xl.loc[:,"C",m] , Xu.loc[:,"C",m] , color = "blue" , alpha = 0.5 )
		ax.set_ylim( (ymin,ymax) )
		ax.set_xticks([])
		ax.legend( loc = "upper left" )
		ax.set_ylabel( ylabel )
		
		ax = fig.add_subplot( nrow , ncol , 3 )
		ax.plot( X.time , X.loc[:,"BE","A",m] , color = "green" , linestyle = "-" , marker = "" , label = r"$\mathrm{Anthropic}$" )
		ax.fill_between( X.time , Xl.loc[:,"A",m] , Xu.loc[:,"A",m] , color = "green" , alpha = 0.5 )
		ax.set_ylim( (yminAnt,ymaxAnt) )
		ax.legend( loc = "upper left" )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( ylabel )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( xlim , [0,0] , color = "black" , linestyle = "-" , marker = "" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	if "Multi_Synthesis" in clim.model:
		pb.print()
		nrow,ncol = 3,1
		fs = 10
		fig = plt.figure( figsize = ( fs * ncol , 0.4 * fs * nrow ) )
		m = "Multi_Synthesis"
		
		ax = fig.add_subplot( nrow , ncol , 1 )
		ax.plot( X.time , X.loc[:,"BE","F",m] , color = "red" , linestyle = "-" , marker = "" , label = r"$\mathrm{Factual}$" )
		ax.fill_between( X.time , Xl.loc[:,"F",m] , Xu.loc[:,"F",m] , color = "red" , alpha = 0.5 )
		ax.set_ylim( (ymin,ymax) )
		ax.legend( loc = "upper left" )
		ax.set_xticks([])
		ax.set_title( "{}".format( str(m).replace("_"," ") ) )
		ax.set_ylabel( ylabel )
		
		ax = fig.add_subplot( nrow , ncol , 2 )
		ax.plot( X.time , X.loc[:,"BE","C",m] , color = "blue" , linestyle = "-" , marker = "" , label = r"$\mathrm{Counter-factual}$" )
		ax.fill_between( X.time , Xl.loc[:,"C",m] , Xu.loc[:,"C",m] , color = "blue" , alpha = 0.5 )
		ax.set_ylim( (ymin,ymax) )
		ax.set_xticks([])
		ax.legend( loc = "upper left" )
		ax.set_ylabel( ylabel )
		
		ax = fig.add_subplot( nrow , ncol , 3 )
		ax.plot( X.time , X.loc[:,"BE","A",m] , color = "green" , linestyle = "-" , marker = "" , label = r"$\mathrm{Anthropic}$" )
		ax.fill_between( X.time , Xl.loc[:,"A",m] , Xu.loc[:,"A",m] , color = "green" , alpha = 0.5 )
		ax.set_ylim( (yminAnt,ymaxAnt) )
		ax.legend( loc = "upper left" )
		ax.set_xlabel( "Time" )
		ax.set_ylabel( ylabel )
		
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		ax.plot( xlim , [0,0] , color = "black" , linestyle = "-" , marker = "" )
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
	
	pdf.close()
	
	pb.end()

##}}}



