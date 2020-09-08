
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
from .__tools import ProgressBar


#############
## Classes ##
#############

class MultiModel:##{{{
	"""
	NSSEA.MultiModel
	================
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
	
	def _fit( self , mm_matrix ):##{{{
		n_params,n_sample,n_models = mm_matrix.shape
		
		cov_S = np.zeros( (n_params,n_params) )
		for i in range(n_models):
			cov_S += np.cov( mm_matrix[:,1:,i] )
		
		SSM     = np.cov( mm_matrix[:,0,:] ) * ( n_models - 1 )
		cov_CMU = matrix_positive_part( SSM / ( n_models - 1 ) - cov_S / n_models )
		self.cov  = ( n_models + 1 ) / n_models * cov_CMU + cov_S / n_models**2
	##}}}
	
	def fit( self , mm_matrix ):##{{{
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
		self._fit(mm_matrix)
	##}}}
	
	def rvs(self):##{{{
		"""
		Return a random sample from multi model
		"""
		return self.mean + self.std @ np.random.normal(size = self.mean.size)
	##}}}
	
	## Properties {{{
	
	@property
	def n_mm_coef(self):
		return None if self.mean is None else self.mean.size
	
	@property
	def cov(self):
		return self._cov
	
	@cov.setter
	def cov( self , _cov ):
		self._cov = _cov
		self.std = matrix_squareroot(self._cov)
	
	##}}}
	
##}}}


###############
## Functions ##
###############

def infer_multi_model( clim , verbose = False ):
	"""
	NSSEA.infer_multi_model
	=======================
	Infer multi-model synthesis. A new model called "Multi_Synthesis" is added
	to "clim", synthesis of the model. The parameters are given
	in "clim.synthesis".
	
	Arguments
	---------
	clim : [NSSEA.Climatology] Clim variable
	verbose  : [bool] Print (or not) state of execution
	
	Return
	------
	clim: [NSSEA.Climatology] The clim with the multi model synthesis with
	      name "Multi_Synthesis"
	
	"""
	
	pb = ProgressBar( 3 , "infer_multi_model" , verbose )
	
	## Parameters
	##===========
	n_time    = clim.n_time
	n_coef    = clim.n_coef
	n_sample  = clim.n_sample
	n_model   = clim.n_model
	sample    = clim.sample
	n_mm_coef = 2 * n_time + n_coef
	
	## Big matrix
	##===========
	mm_data                        = np.zeros( (n_mm_coef,n_sample + 1,n_model) )
	mm_data[:n_time,:,:]           = clim.X.loc[:,:,"F",:].values
	mm_data[n_time:(2*n_time),:,:] = clim.X.loc[:,:,"C",:].values
	mm_data[(2*n_time):,:,:]       = clim.law_coef.values
	pb.print()
	
	## Multi model parameters inference
	##=================================
	mmodel = MultiModel()
	mmodel.fit( mm_data )
	pb.print()
	
	## Generate sample
	##================
	name = "Multi_Synthesis"
	mm_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,2,1) ) , coords = [ clim.time , sample , clim.data.forcing , [name] ] , dims = ["time","sample","forcing","model"] )
	mm_params = xr.DataArray( np.zeros( (n_coef,n_sample + 1,1) )   , coords = [ clim.law_coef.coef.values , sample , [name] ]     , dims = ["coef","sample","model"] )
	
	mm_sample.loc[:,"BE","F",name] = mmodel.mean[:n_time]
	mm_sample.loc[:,"BE","C",name] = mmodel.mean[n_time:(2*n_time)]
	mm_params.loc[:,"BE",name]     = mmodel.mean[(2*n_time):]
	
	for s in sample[1:]:
		draw = mmodel.rvs()
		mm_sample.loc[:,s,"F",name] = draw[:n_time]
		mm_sample.loc[:,s,"C",name] = draw[n_time:(2*n_time)]
		mm_params.loc[:,s,name]     = draw[(2*n_time):]
	pb.print()
	
	
	## Add multimodel to clim
	##=======================
	X        = xr.concat( [clim.X , mm_sample] , "model" )
	law_coef = xr.concat( [clim.law_coef,mm_params] , "model" )
	clim.data = xr.Dataset( { "X" : X , "law_coef" : law_coef } )
	
	## Add multimodel to xarray, and add to clim
	##==========================================
	index = [ "{}F".format(t) for t in clim.time ] + [ "{}C".format(t) for t in clim.time ] + clim.data.coef.values.tolist()
	dmm_mean = xr.DataArray( mmodel.mean , dims = ["coef"] , coords = [index] )
	dmm_cov  = xr.DataArray( mmodel.cov  , dims = ["coef","coef"] , coords = [index,index] )
	clim.synthesis = xr.Dataset( { "mean" : dmm_mean , "cov" : dmm_cov } )
	
	pb.end()
	
	return clim

