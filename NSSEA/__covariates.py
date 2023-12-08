
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
import pandas as pd
import xarray as xr
import scipy.stats as sc
import netCDF4 as nc4
import pygam as pg
import statsmodels.gam.api as gamapi

from .__tools import ProgressBar


##########################
## Energy Balance Model ##
##########################

class EBM: ##{{{
	"""
	NSSEA.EBM
	=========
	
	An EBM model to approximate natural forcing, 13 differents parameters are possible.
	Note that the param "0" is the mean of output of other samples
	
	"""
	def __init__( self ):
		"""
		Constructor
		"""
		self.file_path = os.path.join( os.path.dirname(os.path.realpath(__file__)) , "data" )
		self.file_EBM = "EBM_param_DA_test.nc"
		
		with nc4.Dataset( os.path.join( self.file_path , self.file_EBM ) , "r" ) as File:
			self.sigma = np.array(File.variables["ebm_sigmaforcing"][:])
			self.sigma_names = File.variables["ebm_sigmaforcing"].forcing_name.split( " " )
			self.model_value = np.array(File.variables["ebm_param"][:])
			self.model_names = File.variables["ebm_param"].model_name.split(" ")
			self.model_param = File.variables["ebm_param"].param_name.split(" ")
			self.forcing = np.array( File.variables["ebm_forcing"][:] )
			self.forcing_names = File.variables["ebm_forcing"].forcing_name.split( " " )
			self.year = self.forcing[:,0]
	
	def draw_sample( self , t , n_sample , fix_first = None ):
		"""
		Draw many sample with EBM parameters
		
		Parameters
		----------
		t        : array
			Time axis (in years)
		n_sample : integer
			How many sample ?
		fix_first: None or integer 
			Choose if the first sample returned is fix to a particular parameter
		
		Return
		------
		out      : np.array[ shape = (time.size,n_sample) ]
			EBM computed
		"""
		ebm = np.zeros( (t.size,13) )
		ebm[:,1:] = self.predict( t , range(12) )
		ebm[:,0] = np.mean( ebm[:,1:] , axis = 1 )
		sample = np.random.choice( 13 , n_sample , replace = True )
		out = ebm[:,sample]
		if fix_first is not None:
			out[:,0] = ebm[:,fix_first]
		return pd.DataFrame( out , index = t )
	
	def predict( self , t , I ):
		"""
		Predict EBM
		
		Parameters
		----------
		t : array
			Time axis (in years)
		I : array of integer
			Integer between 0 and 12, repetition and cut are possible
		
		Return
		------
		out      : np.array[ shape = (t.size,I.size) ]
			EBM computed
		"""
		I = np.array([I]).ravel()
		res = np.zeros( (t.size,I.size) )
		for i in range(I.size):
			res[:,i] = self._make_predict(t,I[i])
		if I.size == 1:
			res = np.ravel(res)
		return res
	
	
	def _make_predict( self , t , i ):
		res = self._hmodel( self.forcing[:,3] , self.model_value[i,3] , self.model_value[i,4] ,  self.model_value[i,1] , self.model_value[i,2] )
		output = np.zeros( (t.size) )
		for j,dt in enumerate(t):
			output[j] = res[np.argwhere( dt == self.year ).ravel(),0]
		return output
	
	
	def _hmodel( self , forcing , c , c0 , lamb , gamm ):
		N = forcing.size
		dt = 1.
		res = np.zeros( (N+1,3) )
		
		for i in range(2,N+1):
			res[i,0] = res[i-1,0] + (dt / c) * ( forcing[i-1] - lamb * res[i-1,0] - gamm * ( res[i-1,0] - res[i-1,1] ) )
			res[i,1] = res[i-1,1] + (dt / c0 ) * gamm * ( res[i-1,0] - res[i-1,1] )
		res[:,2] = gamm * ( res[:,0] - res[:,1] )
		
		return np.delete( res , 0 , 0 )
##}}}


#####################################################
## Factual / counter factual forcing decomposition ##
#####################################################

class GAM_FC:##{{{
	def __init__( self , dof ):
		self.dof   = dof
		self.tol   = 1e-2
		self.model = None

	@property
	def edof(self):
		return self.model.statistics_["edof"]
	
	@property
	def coef_(self):
		return self.model.coef_
	
	@coef_.setter
	def coef_( self , coef ):
		self.model.coef_ = coef
	
	@property
	def cov_(self):
		return self.model.statistics_["cov"]
	
	def fit( self , X , Y ):
		lam_up    = 1e2
		lam_lo    = 1e-2
		diff      = 1. + self.tol
		n_splines = int(self.dof + 2)
		nit       = 0
		while diff > self.tol:
			lam = ( lam_up + lam_lo ) / 2.
			
			self.model = pg.LinearGAM( pg.s( 0 , n_splines = n_splines , penalties = "auto" , lam = lam ) + pg.l( 1 , penalties = None ) )
			self.model.fit( X , Y )
			
			if self.edof < self.dof:
				lam_up = lam
			else:
				lam_lo = lam
			diff = np.abs( self.dof - self.edof )
			nit += 1
			if nit % 100 == 0:
				lam_up    = 1e2
				lam_lo    = 1e-2
				n_splines += 1
	
	def predict( self , X ):
		return self.model.predict(X)
	
	def error_distribution(self):
		return sc.multivariate_normal( mean = self.coef_ , cov = self.cov_ , allow_singular = True )

##}}}

def covariates_FC_GAM( clim , lX , XN , dof = 7 , method = "pygam" , verbose = False ):##{{{
	"""
	NSSEA.covariates_FC_GAM
	=======================
	
	This function add to the climatology "clim" the factual / counter factual
	decomposition of the models "lX" with the following GAM model:
	
	X ~ X0 + XN + spline(clim.time) + normal_distribution
	
	Factual and counter covariates are of the form:
	XF ~ X0 + XN + spline(clime.time)
	XC ~ X0 + XN
	
	The GAM model is fitted with "dof" degree of freedom, i.e. if "dof = 7", we
	have one dof for X0, one dof for XN, and 5 dof for the spline term.
	
	Parameters
	==========
	clim : [NSSEA.climatology] Climatology to add the covariate.
	lX   : [list(pandas.DataFrame)] List of models, the column name must be the 
	       name of the model, and defined in the climatology.
	XN   : [pandas.DataFrame] Natural forcing.
	dof  : [integer] Degree of freedom
	method : [str] "statsmodels" or "pygam", select the package used to solve
	         GAM model
	verbose : [bool] If we print the progress of the fit or not.
	
	"""
	if method == "pygam":
		return _covariates_FC_GAM_pygam( clim , lX , XN , dof , verbose )
	else:
		return _covariates_FC_GAM_statsmodels( clim , lX , XN , dof , verbose )
##}}}

def _covariates_FC_GAM_pygam( clim , lX , XN , dof = 7 , verbose = False ):##{{{
	"""
	NSSEA.covariates_FC_GAM_pygam
	=============================
	
	Same parameters that NSSEA.covariates_FC_GAM, use pygam package.
	
	"""
	## Parameters
	models   = clim.model
	time     = clim.time
	n_model  = clim.n_model
	n_time   = clim.n_time
	samples  = clim.sample
	n_sample = clim.n_sample
	
	## verbose
	pb = ProgressBar( n_model * n_sample , "covariates_FC_GAM" , verbose )
	
	## Define output
	dX = xr.DataArray( np.zeros( (n_time,n_sample + 1,2,n_model) ) , coords = [time , samples , ["F","C"] , models ] , dims = ["time","sample","forcing","model"] )
	
	## Define others prediction variables
	time_C   = np.repeat( time[0] , n_time )
	
	## Main loop
	for X in lX:
		model = X.columns[0]
		
		xn = XN.values[:,0]
		XF = np.stack( (time  ,xn) , -1 )
		XC = np.stack( (time_C,xn) , -1 )
		
		## GAM decomposition
		gam_model = GAM_FC( dof )
		gam_model.fit( np.stack( (X.index,XN.loc[X.index,0].values) , -1 ) , X.values )
		
		## prediction
		dX.loc[:,"BE","F",model] = gam_model.predict( XF )
		dX.loc[:,"BE","C",model] = gam_model.predict( XC )
		
		## Distribution of GAM coefficients
		gam_law = gam_model.error_distribution()
		coefs_  = gam_law.rvs(n_sample)
		
		for i,s in enumerate(samples[1:]):
			pb.print()
			
			xn = XN.values[:,i+1]
			XF = np.stack( (time  ,xn) , -1 )
			XC = np.stack( (time_C,xn) , -1 )
			
			## Perturbation
			gam_model.coef_ = coefs_[i,:]
			
			## Final decomposition
			dX.loc[:,s,"F",model] = gam_model.predict( XF )
			dX.loc[:,s,"C",model] = gam_model.predict( XC )
	
	clim.X = dX
	pb.end()
	return clim
##}}}

def _covariates_FC_GAM_statsmodels( clim , lX , XN , dof = 7 , verbose = False ):##{{{
	"""
	NSSEA.covariates_FC_GAM_statsmodels
	===================================
	
	Same parameters that NSSEA.covariates_FC_GAM, use statsmodels package.
	
	"""
	## Parameters
	models   = clim.model
	time     = clim.time
	n_model  = clim.n_model
	n_time   = clim.n_time
	samples  = clim.sample
	n_sample = clim.n_sample
	
	## verbose
	pb = ProgressBar( n_model , "covariates_FC_GAM" , verbose )
	
	## Define output
	dX = xr.DataArray( np.zeros( (n_time,n_sample + 1,2,n_model) ) , coords = [time , samples , ["F","C"] , models ] , dims = ["time","sample","forcing","model"] )
	
	## Main loop
	for X in lX:
		pb.print()
		
		## All data in a dataframe
		model = X.columns[0]
		Xg    = X.groupby(int).aggregate(np.mean)
		dataf = pd.DataFrame( np.array([Xg.index,np.repeat(Xg.index[0],Xg.size),XN.loc[Xg.index,0].values.squeeze(),Xg.values.squeeze()]).T , columns = ["timeF","timeC","XN","X"] )
		
		## Define GAM model and fit
		bs      = gamapi.BSplines( dataf["timeF"] , df = dof - 1 , degree = 3 )
		gam_bs  = gamapi.GLMGam.from_formula( "X ~ 1 + XN" , data = dataf , smoother = bs )
		
		res_fit = gam_bs.fit()
		alpha,_,_ = gam_bs.select_penweight()
		gam_bs  = gamapi.GLMGam.from_formula( 'X ~ 1 + ebm' , data = dataf , smoother = bs , alpha = alpha )
		res_fit = gam_bs.fit()
		
		## Build design matrices and coefs
		timeF   = np.unique(dataf.timeF.values)
		timeC   = np.repeat( timeF[0] , timeF.size )
		designF = np.hstack( (np.ones((timeF.size,1)),dataf["XN"].values.reshape(-1,1),gam_bs.smoother.transform(timeF)) )
		designC = np.hstack( (np.ones((timeF.size,1)),dataf["XN"].values.reshape(-1,1),gam_bs.smoother.transform(timeC)) )
		coef_   = res_fit.params.values
		cov_    = res_fit.cov_params().values
		coefs   = np.vstack( ( coef_ , np.random.multivariate_normal( mean = coef_ , cov = cov_ , size = n_sample ) ) )
		
		dX.loc[:,:,"F",model] = designF @ coefs.T
		dX.loc[:,:,"C",model] = designC @ coefs.T
		
	
	clim.X = dX
	pb.end()
	return clim
##}}}


