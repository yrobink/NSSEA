
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
import sys


#############
## Classes ##
#############

class LinkPR:
	"""
	NSSEA.plot.Linkp
	================
	
	Link function to plot Probability Ratio and FAR. The transformation used is:
	y_tmp = arctan(log(PR)) / (pi / 2)
	y     = sign(y) * (abs(y))**e
	"""
	def __init__( self , e = 3. ):
		"""
		Constructor
		===========
		
		Parameters
		----------
		e : [float] exponent in the transformation. An exponent lower than 1
		            focuses on values close to 1, an exponent larger than 1 on
		            values far of 1.
		"""
		self.e          = e
		self.true_ticks = np.array([0,1e-6, 1/1000, 1/100,1/10, 1/5, 1, 5, 10, 100, 1000,1e6,np.Inf])
		self.labels     = [ r"${}$".format(s) for s in ["0", "10^{-6}","10^{-3}","10^{-2}","10^{-1}", "1/5", "1", "5", "10", "10^2", "10^3" , "10^6", "\infty"] ]
		self.FARlabels  = [ r"${}$".format(s) for s in ["-\infty","~","-999" ,"-99","-9", "-4" , "0" , "0.8" , "0.9" , "0.99" , "0.999" , "~","1"] ]
		self.ticks      = self.transform(self.true_ticks)
		self.min        = self.ticks.min()
		self.max        = self.ticks.max()
		
	def transform( self , x ):
		x = np.array([x]).squeeze()
		y = np.arctan( np.log(np.where( x > 0 , x , sys.float_info.epsilon ) ) ) / (np.pi/2)
		return np.sign(y) * np.power( np.abs(y) , self.e )


class Linkp:
	"""
	NSSEA.plot.Linkp
	================
	
	Link function to plot probabilities and return time. The transformation used
	is:
	( 1 + arctan( log(p) ) / (pi / 2) )**e
	"""
	def __init__( self , e = 2. / 3. ):
		"""
		Constructor
		===========
		
		Parameters
		----------
		e : [float] exponent in the transformation. An exponent lower than 1
		            focuses on low values, an exponent larger than 1 on large
		            values.
		"""
		self.e          = e
		self.true_ticks = np.array( [ 0 , 1e-6 , 1e-3	  ,  1e-2   ,   1e-1  , 1/5 , 1/3 , 1/2 , 1 ] )
		self.labels     = [ r"${}$".format(s) for s in ["0","10^{-6}","10^{-3}","10^{-2}","10^{-1}","1/5","1/3","1/2","1"] ]
		self.Rtlabels   = [ r"${}$".format(s) for s in ["\infty","10^6","10^3","10^2","10","5","3","2","1"] ]
		self.ticks      = self.transform(self.true_ticks)
		self.min        = self.ticks.min()
		self.max        = self.ticks.max()
	
	def transform( self , x ):
		x = np.array([x]).squeeze()
		y = np.arctan( np.log(np.where( x > 0 , x , sys.float_info.epsilon ) ) ) / (np.pi/2)
		return np.power( 1. + y , self.e )


