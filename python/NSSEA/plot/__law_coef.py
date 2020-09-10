
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
import numpy  as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from ..__tools import ProgressBar

#############
## Classes ##
#############

def law_coef( clim , ofile , ci = 0.05 , verbose = False ):##{{{
	"""
	NSSEA.plot.law_coef
	====================
	
	Plot a violin plot of the coefficients 
	
	Arguments
	---------
	clim      : [NSSEA.Climatology] Climatology
	ofile     : [str] output file
	ci        : [float] Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : [bool] Print (or not) state of execution
	"""
	
	pb = ProgressBar( clim.n_model , "plot.law_coef" , verbose )
	
	## Quantile
	
	qcoef = clim.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = "sample" ).assign_coords( quantile = ["ql","qu","BE"] )
	if not clim.BE_is_median:
		qcoef.loc["BE",:,:] = clim.law_coef.loc[:,"BE",:]
	qcoef.loc[["ql","qu"],:,:] = qcoef.loc[["ql","qu"],:,:] - qcoef.loc["BE",:,:]
	
	## mpl parameter
	ymin = float( (clim.law_coef - qcoef.loc["BE",:,:]).min())
	ymax = float( (clim.law_coef - qcoef.loc["BE",:,:]).max())
	delta = 0.1 * (ymax-ymin)
	ylim = (ymin-delta,ymax+delta)
	
	kwargs = { "positions" : range(clim.n_coef) , "showmeans" : False , "showextrema" : False , "showmedians" : False }
	pdf = mpdf.PdfPages( ofile )
	for m in clim.model:
		fig = plt.figure( figsize = ( 14 , 10 ) )
		ax = fig.add_subplot(1,1,1)
		
		## violin plot
		vplot = ax.violinplot( (clim.law_coef - qcoef.loc["BE",:,:])[:,1:,:].loc[:,:,m].values.T , **kwargs )
		
		## Change color
		for pc in vplot["bodies"]:
			pc.set_facecolor("red")
			pc.set_edgecolor("red")
			pc.set_alpha(0.5)
		
		## add quantiles
		for i in range(clim.n_coef):
			for q in ["ql","qu"]:
				ax.hlines( qcoef[:,i,:].loc[q,m] , i - 0.3 , i + 0.3 , color = "red" )
		ax.hlines( 0 , -0.5 , clim.n_coef-0.5 , color = "black" )
		for i in range(clim.n_coef-1):
			ax.vlines( i + 0.5 , ylim[0] , ylim[1] , color = "grey" )
		
		## some params
		ax.set_xlim((-0.5,clim.n_coef-0.5))
		ax.set_xticks(range(clim.n_coef))
		xticks = [ "{}".format(p) + "{}".format( "-" if np.sign(q) < 0 else "+" ) + r"${}$".format(float(np.sign(q)) * round(float(q),2)) for p,q in zip(clim.ns_law.get_params_names(True),qcoef.loc["BE",:,m]) ]
		ax.set_xticklabels( xticks , fontsize = 15 )
		for item in ax.get_yticklabels():
			item.set_fontsize(15)
		ax.set_ylim(ylim)
		
		ax.set_title( " ".join(m.split("_")) , fontsize = 20 )
		
		fig.set_tight_layout(True)
		pdf.savefig( fig )
		plt.close(fig)
		pb.print()
	
	pdf.close()
	
	pb.end()
##}}}



