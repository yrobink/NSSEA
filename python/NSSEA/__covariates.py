
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import os
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
import pygam as pg

from .__tools import ProgressBar
from .__tools import matrix_positive_part
from .__tools import matrix_squareroot


#############
## Classes ##
#############

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


###############
## Functions ##
###############

def fit_gam_with_fix_dof( X , Y , dof ):##{{{
	lam_up    = 1e2
	lam_lo    = 1e-2
	tol       = 1e-2
	diff      = 1. + tol
	n_splines = int(dof + 2)
	nit       = 0
	while diff > tol:
		lam = ( lam_up + lam_lo ) / 2.
		
		gam_model = pg.LinearGAM( pg.s( 0 , n_splines = n_splines , penalties = "auto" , lam = lam ) + pg.l( 1 , penalties = None ) )
		gam_model.fit( X , Y )
		current_dof = gam_model.statistics_["edof"]
		if current_dof < dof:
			lam_up = lam
		else:
			lam_lo = lam
		diff = np.abs( dof - current_dof )
		nit += 1
		if nit % 100 == 0:
			lam_up    = 1e2
			lam_lo    = 1e-2
			n_splines += 1
	return gam_model
##}}}

def gam_decomposition( lX , Xnat , dof = 7 , verbose = False ): ##{{{
	"""
	NSSEA.gam_decomposition
	=======================
	Perform the decomposition anthropic/natural forcing with GAM
	
	arguments
	---------
	"""
	models   = [ lx.columns[0] for lx in lX]
	n_models = len(models)
	n_sample = Xnat.shape[1] - 1
	time     = np.unique( lX[0].index )
	n_time   = time.size
	time_l   = np.repeat( time[0] , n_time )
	Xa       = np.repeat( 0. , n_time )
	
	sample = ["be"] + [ "S{}".format(i) for i in range(n_sample) ]
	X = xr.DataArray( np.zeros( (n_time,n_sample + 1,3,n_models) ) , coords = [time , sample , ["all","nat","ant"] , models ] , dims = ["time","sample","forcing","models"] )
	
	spl_pen = "auto"
	lin_pen = None
	
	
	
	pb = ProgressBar( "GAM decomposition" , n_models * n_sample )
	for i in range(n_models):
		
		Xl    = Xnat.values[:,0]
		x_all = np.stack( (time  ,Xl) , -1 )
		x_nat = np.stack( (time_l,Xl) , -1 )
		
		## GAM decomposition
		gam_model = fit_gam_with_fix_dof( np.stack( (lX[i].index,Xnat.loc[lX[i].index,0].values) , -1 ) , lX[i].values , dof )
		
		## prediction
		X.values[:,0,0,i] = gam_model.predict( x_all )
		X.values[:,0,1,i] = gam_model.predict( x_nat )
		
		mean_coef = gam_model.coef_
		cov_coef  = gam_model.statistics_["cov"]
		
		for j in range(n_sample):
			if verbose: pb.print()
			
			Xl    = Xnat.values[:,j+1]
			x_all = np.stack( (time  ,Xl) , -1 )
			x_nat = np.stack( (time_l,Xl) , -1 )
			
			## Perturbation
			gam_model.coef_ = np.random.multivariate_normal( mean = mean_coef , cov = cov_coef , size = 1 ).ravel()
			
			## Final decomposition
			X.values[:,j+1,0,i] = gam_model.predict( x_all )
			X.values[:,j+1,1,i] = gam_model.predict( x_nat )
			
	
	X.loc[:,:,"ant",:] = X.loc[:,:,"all",:] - X.loc[:,:,"nat",:]
	
	if verbose: pb.end()
	
	return X
	
#	if time_center is not None:
#		X_event = X.loc[time_center,:,"all",:]
#		X_center = X - X_event
#	
#	
#	return XSplitted( X , X_event , X_center )
##}}}



#########
## Old ##
#########

class XSplitted: ##{{{
	def __init__( self , X , X_event , X_center ):
		self.X        = X
		self.X_event  = X_event
		self.X_center = X_center
	
	def has_center(self):
		return self.X_center is not None
##}}}

def gam_decomposition_classic( lX , Enat , Sigma = None , time_center = None , n_splines = None , gam_lam = None , verbose = False ): ##{{{
	"""
	NSSEA.gam_decomposition
	=======================
	Perform the decomposition anthropic/natural forcing with GAM
	
	arguments
	---------
	"""
	models   = [ lx.columns[0] for lx in lX]
	n_models = len(models)
	n_sample = Enat.shape[1] - 1
	time     = np.unique( lX[0].index )
	n_time   = time.size
	time_l   = np.repeat( time[0] , n_time )
	Xa       = np.repeat( 0. , n_time )
	
	sample = ["be"] + [ "S{}".format(i) for i in range(n_sample) ]
	X = xr.DataArray( np.zeros( (n_time,n_sample + 1,3,n_models) ) , coords = [time , sample , ["all","nat","ant"] , models ] , dims = ["time","sample","forcing","models"] )
	
	spl_pen = "auto"
	lin_pen = None
	
	if n_splines is None:
		n_splines = 8
	if gam_lam is None:
		gam_lam = 0.6
	
	
	pb = ProgressBar( "GAM decomposition" , n_models * n_sample )
	for i in range(n_models):
		
		Xl    = Enat.values[:,0]
		x_all = np.stack( (time  ,Xl) , -1 )
		x_nat = np.stack( (time_l,Xl) , -1 )
		
		## GAM decomposition
		gam_model = pg.LinearGAM( pg.s( 0 , n_splines = n_splines , penalties = spl_pen , lam = gam_lam ) + pg.l( 1 , penalties = lin_pen ) )
#		gam_model = pg.LinearGAM( pg.s( 0 , n_splines = gam_dof - 2 , penalties = spl_pen , lam = 0.9 ) + pg.l( 1 , penalties = lin_pen ) )
#		gam_model = pg.LinearGAM( pg.s( 0 , n_splines = gam_dof - 2 , penalties = spl_pen ) + pg.l( 1 , penalties = lin_pen ) )
		gam_model.fit( np.stack( (lX[i].index,Enat.loc[lX[i].index,0].values) , -1 ) , lX[i].values )
		
		X.values[:,0,0,i] = gam_model.predict( x_all )
		X.values[:,0,1,i] = gam_model.predict( x_nat )
		
		mean_coef = gam_model.coef_
		cov_coef  = gam_model.statistics_["cov"]
		
		for j in range(n_sample):
			if verbose: pb.print()
			
			Xl    = Enat.values[:,j+1]
			x_all = np.stack( (time  ,Xl) , -1 )
			x_nat = np.stack( (time_l,Xl) , -1 )
			
			## Perturbation
			gam_model.coef_ = np.random.multivariate_normal( mean = mean_coef , cov = cov_coef , size = 1 ).ravel()
			
			## Final decomposition
			X.values[:,j+1,0,i] = gam_model.predict( x_all )
			X.values[:,j+1,1,i] = gam_model.predict( x_nat )
			
	
	X.loc[:,:,"ant",:] = X.loc[:,:,"all",:] - X.loc[:,:,"nat",:]
	
	if time_center is not None:
		X_event = X.loc[time_center,:,"all",:]
		X_center = X - X_event
	
	if verbose: pb.end()
	
	return XSplitted( X , X_event , X_center )
##}}}

def gam_decomposition_old_old_old( Xd , Enat , Sigma = None , time_center = None , gam_dof = 7 , verbose = False ): ##{{{
	"""
	NSSEA.gam_decomposition
	=======================
	Perform the decomposition anthropic/natural forcing with GAM
	
	arguments
	---------
	"""
	models   = Xd.columns.to_list()
	n_models = Xd.shape[1]
	n_sample = Enat.shape[1] - 1
	time     = Xd.index.values
	n_time   = time.size
	time_l   = np.repeat( time[0] , n_time )
	Eant     = np.repeat( 0. , n_time )
	
	sample = ["be"] + [ "S{}".format(i) for i in range(n_sample) ]
	X = xr.DataArray( np.zeros( (n_time,n_sample + 1,3,n_models) ) , coords = [time , sample , ["all","nat","ant"] , models ] , dims = ["time","sample","forcing","models"] )
	
	pb = ProgressBar( "GAM decomposition" , n_models * n_sample )
	for i in range(n_models):
		gam_model = pg.LinearGAM( pg.s( 0 , n_splines = gam_dof - 2 , penalties = None ) + pg.l( 1 , penalties = None ) )
		gam_model.fit( np.stack( (time,Enat.values[:,0]) , -1 ) , Xd.values[:,i] )
		
		X.values[:,0,0,i] = gam_model.predict( np.stack( (time  ,Enat.values[:,0]) , -1 ) )
		X.values[:,0,1,i] = gam_model.predict( np.stack( (time_l,Enat.values[:,0]) , -1 ) )
		X.values[:,0,2,i] = gam_model.predict( np.stack( (time,Eant) , -1 ) )
		
		
		
		for j in range(n_sample):
			if verbose: pb.print()
		
			Xl = Enat.values[:,j+1]
			mVt = np.stack( (time  ,Xl) , -1 )
			mVl = np.stack( (time_l,Xl) , -1 )
			
			## GAM decomposition
			gam_model = pg.LinearGAM( pg.s( 0 , n_splines = gam_dof - 2 , penalties = None ) + pg.l(1, penalties = None ) )
			gam_model.fit( mVt , Xd.values[:,i] )
			
			## Coefficients of decomposition
			int_coef = gam_model.coef_[-1]
			lin_coef = gam_model.coef_[-2]
			spl_coef = gam_model.coef_[:-2]
			
			spl_mat  = gam_model._modelmat( mVt ).todense()[:,:-2]
			proj_mat = spl_mat @ np.linalg.inv( spl_mat.T @ spl_mat ) @ spl_mat.T
			
			
			## Noise of linear term
			sigma_lin = np.sqrt( ( Xl.transpose() @ Sigma @ Xl ) / ( Xl.transpose() @ Xl )**2 )
			noise_lin = np.random.normal( loc = 0 , scale = sigma_lin )
			
			## Noise of spline term
			std_spl   = matrix_squareroot( matrix_positive_part( proj_mat.transpose() @ Sigma @ proj_mat ) )
			noise_spl = np.ravel( std_spl @ np.random.normal( loc = 0 , scale = 1 , size = time.size ).reshape( (n_time,1) ) )
			noise_spl = noise_spl - noise_spl[0]
			
			## Final decomposition
			gam_model.coef_[-2] += noise_lin
			X.values[:,j+1,0,i] = gam_model.predict( mVt ) + noise_spl
			X.values[:,j+1,1,i] = gam_model.predict( mVl )
			X.values[:,j+1,2,i] = gam_model.predict( np.stack( (time,Eant) , -1 ) ) + noise_spl
			
	
	if time_center is not None:
		X_event = X.loc[time_center,:,"all",:]
		X_center = X - X_event
	
	if verbose: pb.end()
	
	return XSplitted( X , X_event , X_center )
##}}}

