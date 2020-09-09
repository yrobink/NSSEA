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

import os
import numpy  as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
#mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

from .__linkParams    import LinkParams
from .__probabilities import probabilities
from .__ns_params     import ns_params
from .__intensities   import intensities
from .__stats_event   import stats_event
from .__stats_event   import stats_relative


###############
## Functions ##
###############


def plot_classic_packages( clim , event , path , suffix = "" , be_is_median = False , ci = 0.05 , verbose = False ):
	"""
	NSSEA.plot.plot_classic_packages
	================================
	
	Just a function which call:
		- NSSEA.plot.probabilities
		- NSSEA.plot.ns_params
		- NSSEA.plot.intensities
		- NSSEA.plot.stats_event
		- NSSEA.plot.stats_relative
	
	
	Arguments
	---------
	clim    : NSSEA.Climatology
		A clim variable
	event     : NSSEA.Event
		Event variable
	suffix    : str
		suffix for the name of files
	ci        : float
		Size of confidence interval, default is 0.05 (95% confidence)
	verbose   : bool
		Print (or not) state of execution
	"""
	
	if verbose: print( "Plot classic_packages ({}): 0/5".format(suffix) , end = "\r" )
	probabilities(  clim , event , ofile = os.path.join( path , "Probabilities" + suffix + ".pdf"                       ) , ci = ci , be_is_median = be_is_median , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 1/5".format(suffix) , end = "\r" )
	intensities(    clim , event , ofile = os.path.join( path , "Intensities"   + suffix + ".pdf"                       ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 2/5".format(suffix) , end = "\r" )
	stats_event(    clim , event , ofile = os.path.join( path , "StatsEvent"    + suffix + "_{}.pdf".format(event.time) ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 3/5".format(suffix) , end = "\r" )
	stats_relative( clim , event , ofile = os.path.join( path , "StatsRelative" + suffix + "_{}.pdf".format(event.time) ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 4/5".format(suffix) , end = "\r" )
	ns_params(      clim ,         ofile = os.path.join( path , "ns_params"     + suffix + ".pdf"                       ) , ci = ci , verbose = False )
	if verbose: print( "Plot classic_packages ({}): 5/5".format(suffix) , end = "\n" )
	

