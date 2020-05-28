
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

import numpy        as np
import xarray       as xr

from .__tools import matrix_positive_part
from .__tools import matrix_squareroot
from .__tools import barycenter_covariance


#############
## Classes ##
#############

class MultiModelParams:##{{{
	"""
	NSSEA.MultiModelParams
	=============
	Class infering multimodel parameters. Use NSSEA.infer_multi_model to build it
	
	
	Attributes
	----------
	mean   : array
		Multi model mean
	cov    : array
		Multi model covariance matrix
	std    : array
		Square root of multimodel covariance matrix
	"""
	
	def __init__( self ):##{{{
		self.mean = None
		self._cov  = None
		self.std  = None
	##}}}
	
	def _fit_classic( self , mm_matrix ):##{{{
		n_params,n_sample,n_models = mm_matrix.shape
		
		cov_S = np.zeros( (n_params,n_params) )
		for i in range(n_models):
			cov_S += np.cov( mm_matrix[:,1:,i] )
		
		SSM     = np.cov( mm_matrix[:,0,:] ) * ( n_models - 1 )
		cov_CMU = matrix_positive_part( SSM / ( n_models - 1 ) - cov_S / n_models )
		self.cov  = ( n_models + 1 ) / n_models * cov_CMU + cov_S / n_models**2
	##}}}
	
	def _fit_empirical( self , mm_matrix ):##{{{
		n_params,n_sample,n_models = mm_matrix.shape
		self.cov = np.cov( mm_matrix.reshape( (n_params,n_sample*n_models) ) )
	##}}}
	
	def _fit_barycenter( self , mm_matrix , verbose ):##{{{
		n_params,n_sample,n_models = mm_matrix.shape
		lcov = [ np.cov( mm_matrix[:,:,i] ) for i in range(n_models) ]
		self.cov = barycenter_covariance( lcov , verbose = verbose )
	##}}}
	
	def fit( self , mm_matrix , method , verbose = False ):##{{{
		"""
		Fit Multi model parameters
		
		Parameters
		----------
		mm_matrix: array
			Big matrix containing sample to infer multi model parameters
		method   : str
			Method used, "classic" or "optimal"
		verbose  : bool
			Print (or not) state of execution
		"""
		self.mean = np.mean( mm_matrix[:,0,:] , axis = 1 )
		if method == "empirical":
			self._fit_empirical(mm_matrix)
		elif method == "barycenter":
			self._fit_barycenter(mm_matrix,verbose)
		else:
			self._fit_classic(mm_matrix)
	##}}}
	
	def rvs(self):##{{{
		"""
		Return a random sample from multi model
		"""
		return self.mean + self.std @ np.random.normal(size = self.mean.size)
	##}}}
	
	## Properties {{{
	
	@property
	def n_mm_params(self):
		return None if self.mean is None else self.mean.size
	
	@property
	def cov(self):
		return self._cov
	
	@cov.setter
	def cov( self , _cov ):
		self._cov = _cov
		self.std = matrix_squareroot(self._cov)
	
	##}}}
	
	def copy(self):##{{{
		mmp = MultiModelParams()
		mmp.mean = None if self.mean is None else self.mean.copy()
		if self._cov is not None:
			mmp.cov = self.cov.copy()
		return mmp
	##}}}
##}}}


###############
## Functions ##
###############

def infer_multi_model( climIn , mm_method = "classic" , verbose = False ):
	"""
	NSSEA.infer_multi_model
	=======================
	Infer multimodel mean and covariance
	
	Arguments
	---------
	climIn : NSSEA.Climatology
		clim variable
	mm_method: str
		Multi model method, currently "classic" (A. Ribes method) or "optimal" (Optimal transport)
	verbose  : bool
		Print (or not) state of execution
	
	Return
	------
	clim: NSSEA.Climatology
		A COPY of the input clim, where clim.mm_params is set, and clim.X contains multi model sample. The input clim IS NOT MODIFIED.
	
	
	"""
	if verbose: print( "Multi model" , end = "\r" if mm_method == "classic" else "\n" )
	## Parameters
	##===========
	clim      = climIn.copy()
	n_time      = clim.n_time
	n_ns_params = clim.n_ns_params
	n_sample    = clim.n_sample
	n_models    = clim.n_models
	sample      = clim.X.sample.values.tolist()
	n_mm_params = 2 * n_time + n_ns_params
	
	## Big matrix
	##===========
	mm_matrix                        = np.zeros( (n_mm_params,n_sample + 1,n_models) )
	mm_matrix[:n_time,:,:]           = clim.X.loc[:,:,"all",:].values
	mm_matrix[n_time:(2*n_time),:,:] = clim.X.loc[:,:,"nat",:].values
	mm_matrix[(2*n_time):,:,:]       = clim.ns_params.values
	
	## Multi model parameters inference
	##=================================
	clim.mm_params.fit( mm_matrix , mm_method , verbose )
	
	## Generate sample
	##================
	mm_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,3,1) )    , coords = [ clim.X.time , sample , clim.X.forcing , ["multi"] ] , dims = ["time","sample","forcing","models"] )
	mm_params = xr.DataArray( np.zeros( (n_ns_params,n_sample + 1,1) ) , coords = [ clim.ns_params.ns_params , sample , ["multi"] ]       , dims = ["ns_params","sample","models"] )
	
	mm_sample.loc[:,"be","all","multi"] = clim.mm_params.mean[:n_time]
	mm_sample.loc[:,"be","nat","multi"] = clim.mm_params.mean[n_time:(2*n_time)]
	mm_params.loc[:,"be","multi"]       = clim.mm_params.mean[(2*n_time):]
	
	for s in sample[1:]:
		draw = clim.mm_params.rvs()
		mm_sample.loc[:,s,"all","multi"] = draw[:n_time]
		mm_sample.loc[:,s,"nat","multi"] = draw[n_time:(2*n_time)]
		mm_params.loc[:,s,"multi"]       = draw[(2*n_time):]
	
	mm_sample.loc[:,:,"ant","multi"] = mm_sample.loc[:,:,"all","multi"] - mm_sample.loc[:,:,"nat","multi"]
	
	## Add multimodel to clim
	##=======================
	clim.X         = xr.concat( [clim.X , mm_sample] , "models" )
	clim.ns_params = xr.concat( [clim.ns_params,mm_params] , "models" )
	clim.models.append( "multi" )
	
	if verbose: print( "Multi model (Done)" )
	
	return clim

