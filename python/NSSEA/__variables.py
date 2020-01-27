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
	"""
	
	def __init__( self , name_event , time , anomaly , reference , type_event = "threshold" , side = "upper" , name_variable = "variable" , unit_variable = "U" ):
		"""
		Constructor of Event
		
		Arguments
		---------
		
		name_event    : str
			Name of event
		time          : time_index
			Time when event occured
		anomaly       : double
			Anomaly of event
		reference     : array
			Time period to considered as reference for anomaly
		type_event    : "threshold" or "hard"
			If we compute probabilities as a threshold beyond mean, or anomaly is used as a hard value.
		side          : str
			"upper" or "lower" extremes event
		name_variable : str
			Name of variable (temperature, precipitation, etc)
		unit_variable : str
			Unit of the variable
		"""
		self.name_event    = name_event
		self.time          = time
		self.anomaly       = anomaly
		self.reference     = reference
		self.type_event    = type_event if type_event in ["threshold","hard"] else "threshold"
		self.side          = side if side in ["upper","lower"] else "upper"
		self.name_variable = name_variable
		self.unit_variable = unit_variable
	
	def __repr__(self):
		return self.__str__()
	
	def __str__(self):
		out = ""
		out += "Event     : {},\n".format(self.name_event)
		out += "variable  : {} ({}),\n".format(self.name_variable,self.unit_variable)
		out += "time      : {},\n".format(self.time)
		out += "anomaly   : {},\n".format(self.anomaly)
		out += "reference : {}-{},\n".format(self.reference.min(),self.reference.max())
		out += "side      : {}\n".format(self.side)
		return out
##}}}

class Climatology: ##{{{
	"""
	NSSEA.Climatology
	============
	
	Build the clim variable containing results of execution of NSSEA
	
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
		Constructor of the clim variable
		
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
		c             = Climatology( self.time.copy() , self.n_sample , self.models , self.ns_law , self.ns_law_args )
		
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


def clim2netcdf( clim , event , ofile , with_cx = False , with_co = False ):##{{{
	with nc.Dataset( ofile , "w" , format = "NETCDF4" ) as ncFile:
	
		## Create dimensions
		dim_time     = ncFile.createDimension( "time"     , clim.X.time.size              )
		dim_sample   = ncFile.createDimension( "sample"   , clim.X.sample.size            )
		dim_forcing  = ncFile.createDimension( "forcing"  , clim.X.forcing.size           )
		dim_models   = ncFile.createDimension( "models"   , clim.X.models.size            )
		dim_ns_param = ncFile.createDimension( "ns_param" , clim.ns_params.ns_params.size )
		dim_stat     = ncFile.createDimension( "stat"     , clim.stats.stats.size         )
		
		## Set dimensions as variables
		nc_time     = ncFile.createVariable( "time"     , clim.X.time.dtype    , ("time",)     )
		nc_sample   = ncFile.createVariable( "sample"   , clim.X.sample.dtype  , ("sample",)   )
		nc_forcing  = ncFile.createVariable( "forcing"  , clim.X.forcing.dtype , ("forcing",)  )
		nc_models   = ncFile.createVariable( "models"   , str                    , ("models",)   )
		nc_ns_param = ncFile.createVariable( "ns_param" , str                    , ("ns_param",) )
		nc_stat     = ncFile.createVariable( "stat"     , str                    , ("stat",)     )
		
		## Set dimensions values
		nc_time[:]     = clim.X.time.values
		nc_sample[:]   = clim.X.sample.values
		nc_forcing[:]  = clim.X.forcing.values
		nc_models[:]   = clim.X.models.values
		nc_ns_param[:] = clim.ns_params.ns_params.values
		nc_stat[:]     = clim.stats.stats.values
		
		## Variables
		nc_X         = ncFile.createVariable( "X"         , clim.X.dtype         , ("time","sample","forcing","models") )
		nc_ns_params = ncFile.createVariable( "ns_params" , clim.ns_params.dtype , ("ns_param","sample","models")       )
		nc_stats     = ncFile.createVariable( "stats"     , clim.stats.dtype     , ("time","sample","stat","models")    )
		
		## Set variables values
		nc_X[:]         = clim.X.values
		nc_ns_params[:] = clim.ns_params.values
		nc_stats[:]     = clim.stats.values
		
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

def netcdf2clim( ifile , ns_law ):##{{{
	with nc.Dataset( ifile , "r" ) as ncFile:
		
		## Extract dimensions
		time     = ncFile.variables["time"][:]
		sample   = ncFile.variables["sample"][:]
		forcing  = ncFile.variables["forcing"][:]
		models   = ncFile.variables["models"][:]
		ns_param = ncFile.variables["ns_param"][:]
		stats    = ncFile.variables["stat"][:]
		
		## 
		clim = Climatology( time , sample.size - 1 , models , ns_law )
		clim.n_ns_params = ns_param.size
		clim.n_stats     = stats.size
		clim.X         = xr.DataArray( ncFile.variables["X"][:]         , coords = [time,sample,forcing,models] , dims = ["time","sample","forcing","models"] )
		clim.ns_params = xr.DataArray( ncFile.variables["ns_params"][:] , coords = [ns_param,sample,models]     , dims = ["ns_params","sample","models"]      )
		clim.stats     = xr.DataArray( ncFile.variables["stats"][:]     , coords = [time,sample,stats,models]   , dims = ["time","sample","stats","models"]   )
	
	return clim
##}}}


