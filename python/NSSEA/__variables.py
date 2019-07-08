# -*- coding: utf-8 -*-

#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import numpy   as np
import xarray  as xr
import netCDF4 as nc


#############
## Classes ##
#############

class Event: ##{{{
	"""
	NSSEA.Event
	===========
	
	Event variable containing information about event considered
	
	Attributes
	----------
	
	name     : str
		Name of event
	dir_name : str
		Directory of output
	time     : time_index
		Time of event
	anom     : double
		Anomaly of event
	ref_anom : array
		Time period to considered as reference for anomaly
	var      : str
		Name of variable
	unit     : str
		Unit of variable
	side     : str
		"upper" or "lower" extremes event
	def_type : str
		If is is anomaly bellow a threshold ("threshold") or a hard value ("hard_value")
	"""
	
	def __init__( self , name , dir_name , time , anom , ref_anom , var , unit , side , def_type = "threshold" ):
		"""
		Constructor of Event
		
		Arguments
		---------
		
		name     : str
			Name of event
		dir_name : str
			Directory of output
		time     : time_index
			Time of event
		anom     : double
			Anomaly of event
		ref_anom : array
			Time period to considered as reference for anomaly
		var      : str
			Name of variable
		side     : str
			"upper" or "lower" extremes event
		"""
		self.name     = name
		self.dir_name = dir_name
		self.def_type = def_type
		self.time     = time
		self.anom     = anom
		self.ref_anom = ref_anom
		self.var      = var
		self.unit     = unit
		self.side     = side
	
	def __repr__(self):
		return self.__str__()
	
	def __str__(self):
		return "Event:\nname: {},\ntime: {},\nanom: {},\nvar : {},\nside: {}\n".format(self.name,self.time,self.anom,self.var,self.side)
##}}}

class Coffee: ##{{{
	"""
	NSSEA.Coffee
	============
	
	Build the coffee variable containing results of execution of NSSEA
	
	Attributes
	----------
	
	time       : array
		Vector of time values
	n_sample   : integer
		Number of sample drawn for confidence interval
	models     : array of str
		Names of models
	ns_law     : NSModel
		Statistical non stationary model from library NSModel
	ns_law_args: dict or None
		Arguments pass to NSModel
	"""
	
	def __init__( self , time , n_sample , models , ns_law , ns_law_args = None ):
		"""
		Constructor of the coffee variable
		
		Arguments
		---------
		
		time     : array
			Vector of time values
		n_sample : integer
			Number of sample drawn for confidence interval
		models   : array of str
			Names of models
		ns_law   : NSModel
			Statistical non stationary model from library NSModel
		"""
		self.X           = None
		self.time        = time
		self.n_time      = len(time)
		self.n_sample    = n_sample
		self.models      = models
		self.n_models    = len(models)
		
		self.ns_law      = ns_law
		self.ns_law_args = ns_law.default_arg(ns_law_args)
		self.ns_params   = None
		self.n_ns_params = None
		
		self.stats       = None
		self.n_stats     = None
		
		self.mm_params   = None
		self.n_mm_params = None
	
	def copy(self):
		c             = Coffee( self.time.copy() , self.n_sample , self.models , self.ns_law , self.ns_law_args )
		
		c.X           = self.X.copy()         if self.X         is not None else None
		c.ns_params   = self.ns_params.copy() if self.ns_params is not None else None
		c.mm_params   = self.mm_params.copy() if self.mm_params is not None else None
		c.stats       = self.stats.copy()     if self.stats     is not None else None
		
		c.n_models    = self.n_models
		c.n_ns_params = self.n_ns_params
		c.n_mm_params = self.n_mm_params
		c.n_stats     = self.n_stats
		return c
##}}}

class CXParams: ##{{{
	"""
	NSSEA.CXParams
	==============
	
	Parameters of the CX constraints
	
	Attributes
	----------
	
	centering: bool
		If we need or not to center covariates to observed covariates
	ref      : array
		Time period of reference period
	trust    : bool
		If we assume that the covariance matrix of observed covariate has the same scale that covariates
	"""
	def __init__( self , centering , ref , trust ):
		"""
		Constructor
		
		Arguments
		---------
		
		centering: bool
			If we need or not to center covariates to observed covariates
		ref      : array
			Time period of reference period
		trust    : bool
			If we assume that the covariance matrix of observed covariate has the same scale that covariates
		"""
		self.centering = centering
		self.ref       = ref
		self.trust     = trust
##}}}


def coffee2netcdf( coffee , event , ofile , with_cx = False , with_co = False ):##{{{
	with nc.Dataset( ofile , "w" , format = "NETCDF4" ) as ncFile:
	
		## Create dimensions
		dim_time     = ncFile.createDimension( "time"     , coffee.X.time.size              )
		dim_sample   = ncFile.createDimension( "sample"   , coffee.X.sample.size            )
		dim_forcing  = ncFile.createDimension( "forcing"  , coffee.X.forcing.size           )
		dim_models   = ncFile.createDimension( "models"   , coffee.X.models.size            )
		dim_ns_param = ncFile.createDimension( "ns_param" , coffee.ns_params.ns_params.size )
		dim_stat     = ncFile.createDimension( "stat"     , coffee.stats.stats.size         )
		
		## Set dimensions as variables
		nc_time     = ncFile.createVariable( "time"     , coffee.X.time.dtype    , ("time",)     )
		nc_sample   = ncFile.createVariable( "sample"   , coffee.X.sample.dtype  , ("sample",)   )
		nc_forcing  = ncFile.createVariable( "forcing"  , coffee.X.forcing.dtype , ("forcing",)  )
		nc_models   = ncFile.createVariable( "models"   , str                    , ("models",)   )
		nc_ns_param = ncFile.createVariable( "ns_param" , str                    , ("ns_param",) )
		nc_stat     = ncFile.createVariable( "stat"     , str                    , ("stat",)     )
		
		## Set dimensions values
		nc_time[:]     = coffee.X.time.values
		nc_sample[:]   = coffee.X.sample.values
		nc_forcing[:]  = coffee.X.forcing.values
		nc_models[:]   = coffee.X.models.values
		nc_ns_param[:] = coffee.ns_params.ns_params.values
		nc_stat[:]     = coffee.stats.stats.values
		
		## Variables
		nc_X         = ncFile.createVariable( "X"         , coffee.X.dtype         , ("time","sample","forcing","models") )
		nc_ns_params = ncFile.createVariable( "ns_params" , coffee.ns_params.dtype , ("ns_param","sample","models")       )
		nc_stats     = ncFile.createVariable( "stats"     , coffee.stats.dtype     , ("time","sample","stat","models")    )
		
		## Set variables values
		nc_X[:]         = coffee.X.values
		nc_ns_params[:] = coffee.ns_params.values
		nc_stats[:]     = coffee.stats.values
		
		## Attributes
		ncFile.event_name = event.name
		ncFile.event_time = str(event.time)
		ncFile.event_anom = event.anom
		ncFile.event_var  = event.var
		ncFile.event_unit = event.unit
		ncFile.event_side = event.side
		
		ncFile.cx = str(with_cx)
		ncFile.co = str(with_co)
##}}}

def netcdf2coffee( ifile , ns_law ):##{{{
	with nc.Dataset( ifile , "r" ) as ncFile:
		
		## Extract dimensions
		time     = ncFile.variables["time"][:]
		sample   = ncFile.variables["sample"][:]
		forcing  = ncFile.variables["forcing"][:]
		models   = ncFile.variables["models"][:]
		ns_param = ncFile.variables["ns_param"][:]
		stats    = ncFile.variables["stat"][:]
		
		## 
		coffee = Coffee( time , sample.size - 1 , models , ns_law )
		coffee.n_ns_params = ns_param.size
		coffee.n_stats     = stats.size
		coffee.X         = xr.DataArray( ncFile.variables["X"][:]         , coords = [time,sample,forcing,models] , dims = ["time","sample","forcing","models"] )
		coffee.ns_params = xr.DataArray( ncFile.variables["ns_params"][:] , coords = [ns_param,sample,models]     , dims = ["ns_params","sample","models"]      )
		coffee.stats     = xr.DataArray( ncFile.variables["stats"][:]     , coords = [time,sample,stats,models]   , dims = ["time","sample","stats","models"]   )
	
	return coffee
##}}}


