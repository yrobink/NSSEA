
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
import numpy as np
import scipy.linalg as scl


class ProgressBar: ##{{{
	"""
	NSSEA.ProgressBar
	=================
	
	Class used to print a progress bar in screen during execution
	
	
	Example
	-------
	>> pb = ProgressBar( "My task" , 100 )
	>> for i in range(100):
	>>     pb.print()
	>> pb.end()
	"""
	def __init__( self , n_step , message , verbose = True ):
		"""
		Constructor

		Arguments
		---------
		n_step  : Lenght of loop where progress bar will be inserted
		message : Message printed
		verbose : If we print or not
		"""
		self.message = message
		self.size_mess = len(message)
		self.n_step  = n_step
		self.step    = 0
		self.verbose = verbose
		self.print()

	def print( self ):
		"""
		Method which print on the screen
		"""
		if self.verbose:
			n_char = os.get_terminal_size()[0] - self.size_mess - 14
			n_char1 = int(n_char * self.step / self.n_step)
			n_char2 = n_char - n_char1
			output = "{} ({}) [{}{}]".format( self.message , "{0:{fill}{align}{n}}%".format(round(100 * self.step / self.n_step,2),fill=" ",align=">",n=5 ) , "#"*n_char1 , " "*n_char2 )
			print( output  , end = "\r" ) 
		self.step += 1
	
	def end(self):
		"""
		Method printed the final message afer loop
		"""
		if self.verbose:
			print("",end="\n")
##}}}


###############
## Functions ##
###############

def matrix_squareroot( M , disp = False ):##{{{
	"""
	NSSEA.matrix_squareroot
	=======================
	Method which compute the square root of a matrix (in fact just call scipy.linalg.sqrtm), but if disp == False, never print warning
	
	Arguments
	---------
	M   : np.array
		A matrix
	disp: bool
		disp error (or not)
	
	Return
	------
	Mp : np.array
		The square root of M
	"""
	Mh = scl.sqrtm( M , disp = disp )
	if not disp:
		Mh = Mh[0]
	return np.real(Mh)
##}}}

def matrix_positive_part( M ):##{{{
	"""
	NSSEA.matrix_positive_part
	==========================
	Return the positive part of a matrix
	
	Arguments
	---------
	M  : np.array
		A matrix
	
	Return
	------
	Mp : np.array
		The positive part of M
	
	"""
	lbda,v = np.linalg.eig(M)
	lbda   = np.real(lbda)
	v      = np.real(v)
	lbda[lbda<0] = 0
	return v @ np.diag(lbda) @ v.T
##}}}

def barycenter_covariance( lcov , weights = None , maxit = 50 , tol = 1e-3 , verbose = False ):##{{{
	"""
	NSSEA.barycenter_covariance
	===========================
	Compute the barycenter (in optimal transport sense) of covariance matrices
	
	Arguments
	---------
	lcov   : list[np.array]
		A list of covariance matrix
	weights: array
		Weights of each covariance matrices, if None 1/size is used
	maxit  : integer
		Max number of iterate. Default is 50
	tol    : float
		Numerical tolerance, default is 1e-3
	verbose : bool
		Print (or not) state of execution
	Return
	------
	cov   : np.array
		A covariance matrix barycenter of lcov
	"""
	
	n_cov = len(lcov)
	
	if weights is None:
		weights = np.ones( n_cov )
	weights = np.array(weights) / np.sum(weights)
	
	def brower( S , lcov , weights ):
		root_S = matrix_squareroot(S)
		Sn = np.zeros( lcov[0].shape )
		
		for i in range(n_cov):
			Sn += weights[i] * matrix_squareroot( root_S @ lcov[i] @ root_S )
		
		return Sn
	
	cov  = np.identity(lcov[0].shape[0])
	diff = 1. + tol
	nit  = 0
	
	while diff > tol and nit < maxit:
		if verbose: print( "Optimal barycenter: {} (<{}), {} (>{})                                        ".format(nit,maxit,round(diff,4),tol) , end = "\r" )
		covn = brower( cov , lcov , weights )
		diff = np.linalg.norm(cov - covn)
		cov  = covn
		nit += 1 
	
	if verbose: print( "Optimal barycenter (Done)                                                    " )
	return cov
##}}}


