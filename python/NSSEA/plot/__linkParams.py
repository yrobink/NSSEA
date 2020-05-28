
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

class LinkParams_old:##{{{
	
	class A:
		def __init__(self):
			self.values = None
			self.names  = None
	
	def __init__( self ):
		self._ticks     = np.array([1e-300, 1/100,1/10, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 10, 100, 1e300])
		self.rr         = LinkParams.A()
		self.rr.values  = self.frr(self._ticks)
		self.rr.names   = [ r"${}$".format(x) for x in ["0", "10^{-2}","10^{-1}", "1/3", "1/2", "2/3", "1", "1.5", "2", "3", "10", "10^2", "\infty"] ]
		self.far        = LinkParams.A()
		self.far.values = self.frr(1 - 1. / self._ticks)
		self.far.names  = [ r"${}$".format(x) for x in ["-\infty","-99","-9", "-2" , "-1" , "-0.5" , "0" , "1/3" , "1/2" , "2/3" , "0.9" , "0.99" , "1"] ]
		self.p          = LinkParams.A()
		self.p.values   = self.fp([ 1e-300 , 1e-5	  ,  1e-2   ,   1e-1  , 1/5 , 1/3 , 1/2 , 1 ])
		self.p.names    = [ r"${}$".format(x) for x in ["0","10^{-5}","10^{-2}","10^{-1}","1/5","1/3","1/2","1"] ]
	
	def _log_no_warn( self , x ):
		x = np.array(x)
		x[ np.logical_not( x > 0 ) ] = sys.float_info.epsilon
		return np.log(x)
	
	def frr( self , x ):
		return np.arctan( self._log_no_warn(x) ) / (np.pi / 2.)
	
	def fp( self , x ):
		return np.power( 1. + self.frr(x) , 1. / 1.2 )
##}}}

class LinkParams:##{{{
	
	def _add_true_extreme( self , X ):##{{{
		for i,x in enumerate(X):
			if x and i > 0:
				X[i-1] = True
				break
		for i,x in enumerate(X[::-1]):
			if x and i > 0:
				X[X.size-i] = True
				break
		return X
	##}}}
	
	class A:
		def __init__(self):
			self.values = None
			self.names  = None
	
	def __init__( self , rr_min = 1e-300 , rr_max = 1e300 , p_min = 1e-300 , p_max = 1 ):##{{{
		self._rrticks = np.array([1e-300, 1/100,1/10, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 10, 100, 1e300])
		self._rrlab   = np.array( ["0", "10^{-2}","10^{-1}", "1/3", "1/2", "2/3", "1", "1.5", "2", "3", "10", "10^2", "\infty"] )
		self._farlab  = np.array( ["-\infty","-99","-9", "-2" , "-1" , "-0.5" , "0" , "1/3" , "1/2" , "2/3" , "0.9" , "0.99" , "1"] )
		self._rr_idx = self._add_true_extreme( np.logical_not( np.logical_or( self._rrticks < rr_min , self._rrticks > rr_max ) ) )
		self.rrticks = self._rrticks[self._rr_idx]
		self.rrlab   = self._rrlab[self._rr_idx]
		self.farlab  = self._farlab[self._rr_idx]
		
		self._pticks = np.array( [ 1e-300 , 1e-5	  ,  1e-2   ,   1e-1  , 1/5 , 1/3 , 1/2 , 1 ] )
		self._plab   = np.array( ["0","10^{-5}","10^{-2}","10^{-1}","1/5","1/3","1/2","1"] )
		self._p_idx = self._add_true_extreme( np.logical_not( np.logical_or( self._pticks < p_min , self._pticks > p_max ) ) )
		self.pticks = self._pticks[self._p_idx]
		self.plab   = self._plab[self._p_idx]
		
		
		self.rr         = LinkParams.A()
		self.rr.values  = self.frr(self.rrticks)
		self.rr.names   = [ r"${}$".format(x) for x in self.rrlab ]
		self.far        = LinkParams.A()
		self.far.values = self.frr(1 - 1. / self.rrticks)
		self.far.names  = [ r"${}$".format(x) for x in self.farlab ]
		self.p          = LinkParams.A()
		self.p.values   = self.fp(self.pticks)
		self.p.names    = [ r"${}$".format(x) for x in self.plab ]
	##}}}
	
	def _log_no_warn( self , x ):
		x = np.array(x)
		x[ np.logical_not( x > 0 ) ] = sys.float_info.epsilon
		return np.log(x)
	
	def frr( self , x ):
		return np.arctan( self._log_no_warn(x) ) / (np.pi / 2.)
	
	def fp( self , x ):
		return np.power( 1. + self.frr(x) , 1. / 1.2 )
##}}}

