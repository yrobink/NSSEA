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

from .__multi_model import MultiModelParams


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
	
	def __init__( self , time , models , ns_law ):##{{{
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
		self.models      = models
		
		self.ns_law      = ns_law
		self.ns_params   = None
		self.stats       = None
		self.mm_params   = MultiModelParams()
	##}}}
	
	## Properties {{{
	
	@property
	def n_time(self):
		return self.time.size
	
	@property
	def n_models(self):
		return len(self.models)
	
	@property
	def n_sample(self):
		return None if self.X is None else self.X.shape[1] - 1
	
	@property
	def n_ns_params(self):
		return None if self.ns_params is None else self.ns_params.shape[0]
	
	@property
	def n_stats(self):
		return None if self.stats is None else self.stats.shape[2]
	
	@property
	def n_mm_params(self):
		return self.mm_params.n_mm_params
	##}}}
	
	def keep_models( self , models ):##{{{
		models = models if type(models) is list else [models]
		if not np.all( [m in self.models for m in models] ):
			return
		self.models = models
		if self.X         is not None: self.X         = self.X.loc[:,:,:,models]
		if self.ns_params is not None: self.ns_params = self.ns_params.loc[:,:,models]
		if self.stats     is not None: self.stats     = self.stats.loc[:,:,:,models]
	##}}}
	
	def remove_models( self , models ):##{{{
		models_keep = [ m for m in self.models if m not in models ]
		self.keep_models(models_keep)
	##}}}
	
	def copy(self):##{{{
		c           = Climatology( self.time.copy() , self.models , self.ns_law )
		c.X         = self.X.copy()         if self.X         is not None else None
		c.ns_params = self.ns_params.copy() if self.ns_params is not None else None
		c.mm_params = self.mm_params.copy() if self.mm_params is not None else None
		c.stats     = self.stats.copy()     if self.stats     is not None else None
		
		return c
	##}}}
##}}}


def to_netcdf( clim , event , ofile , constraints = None ):##{{{
	with nc.Dataset( ofile , "w" , format = "NETCDF4" ) as ncFile:
	
		## Create dimensions
		dim_time     = ncFile.createDimension( "time"     , clim.X.time.size              )
		dim_sample   = ncFile.createDimension( "sample"   , clim.X.sample.size            )
		dim_forcing  = ncFile.createDimension( "forcing"  , clim.X.forcing.size           )
		dim_models   = ncFile.createDimension( "models"   , clim.X.models.size            )
		dim_ns_param = ncFile.createDimension( "ns_param" , clim.ns_params.ns_params.size )
		dim_stat     = ncFile.createDimension( "stat"     , clim.stats.stats.size         )
		dim_ref      = ncFile.createDimension( "ref"      , event.reference.size          )
		dim_multimod = ncFile.createDimension( "mm_size"  , clim.n_mm_params              )
		
		
		## Set dimensions as variables
		nc_time      = ncFile.createVariable( "time"      , clim.X.time.dtype     , ("time",)     )
		nc_sample    = ncFile.createVariable( "sample"    , str                   , ("sample",)   )
		nc_forcing   = ncFile.createVariable( "forcing"   , str                   , ("forcing",)  )
		nc_models    = ncFile.createVariable( "models"    , str                   , ("models",)   )
		nc_ns_param  = ncFile.createVariable( "ns_param"  , str                   , ("ns_param",) )
		nc_stat      = ncFile.createVariable( "stat"      , str                   , ("stat",)     )
		nc_reference = ncFile.createVariable( "reference" , event.reference.dtype , ("ref",)      )
		
		
		## Set dimensions values
		nc_time[:]      = clim.X.time.values
		nc_sample[:]    = clim.X.sample.values
		nc_forcing[:]   = clim.X.forcing.values
		nc_models[:]    = clim.X.models.values
		nc_ns_param[:]  = clim.ns_params.ns_params.values
		nc_stat[:]      = clim.stats.stats.values
		nc_reference[:] = event.reference
		
		## Variables
		nc_X         = ncFile.createVariable( "X"         , clim.X.dtype              , ("time","sample","forcing","models") )
		nc_ns_params = ncFile.createVariable( "ns_params" , clim.ns_params.dtype      , ("ns_param","sample","models")       )
		nc_stats     = ncFile.createVariable( "stats"     , clim.stats.dtype          , ("time","sample","stat","models")    )
		nc_mm_mean   = ncFile.createVariable( "mm_mean"   , clim.mm_params.mean.dtype , ("mm_size")                          )
		nc_mm_cov    = ncFile.createVariable( "mm_cov"    , clim.mm_params.cov.dtype  , ("mm_size","mm_size")                )
		
		## Set variables values
		nc_X[:]         = clim.X.values
		nc_ns_params[:] = clim.ns_params.values
		nc_stats[:]     = clim.stats.values
		nc_mm_mean[:]   = clim.mm_params.mean
		nc_mm_cov[:]    = clim.mm_params.cov
		
		## Attributes for event
		ncFile.event_name     = event.name_event
		ncFile.event_time     = str(event.time)
		ncFile.event_anomaly  = event.anomaly
		ncFile.event_variable = event.name_variable
		ncFile.event_unit     = event.unit_variable
		ncFile.event_side     = event.side
		ncFile.event_type     = event.type_event
		
		## Attributes for constraints
		constraints = constraints if constraints is not None else "No_constraints"
		ncFile.constraints = constraints
		
		## Attributes for law
		ncFile.ns_law = str(clim.ns_law)
		
##}}}

def from_netcdf( ifile , ns_law ):##{{{
	with nc.Dataset( ifile , "r" ) as ncFile:
		
		## Extract dimensions
		time      = np.ma.getdata( ncFile.variables["time"][:] )
		sample    = ncFile.variables["sample"][:]
		forcing   = ncFile.variables["forcing"][:]
		models    = ncFile.variables["models"][:]
		ns_param  = ncFile.variables["ns_param"][:]
		stats     = ncFile.variables["stat"][:]
		reference = np.ma.getdata( ncFile.variables["reference"][:] )
		
		
		##  Set climatology
		clim = Climatology( time , models , ns_law )
		clim.X         = xr.DataArray( np.ma.getdata( ncFile.variables["X"][:]         ) , coords = [time,sample,forcing,models] , dims = ["time","sample","forcing","models"] )
		clim.ns_params = xr.DataArray( np.ma.getdata( ncFile.variables["ns_params"][:] ) , coords = [ns_param,sample,models]     , dims = ["ns_params","sample","models"]      )
		clim.stats     = xr.DataArray( np.ma.getdata( ncFile.variables["stats"][:]     ) , coords = [time,sample,stats,models]   , dims = ["time","sample","stats","models"]   )
		clim.mm_params.mean = np.ma.getdata( ncFile.variables["mm_mean"][:] )
		clim.mm_params.cov  = np.ma.getdata( ncFile.variables["mm_cov"][:]  )
		
		## Set event
		event = Event( ncFile.event_name , reference.dtype.type(ncFile.event_time) , clim.X.dtype.type(ncFile.event_anomaly) , reference , ncFile.event_type , ncFile.event_side , ncFile.event_variable , ncFile.event_unit ) 
	
	return clim,event
##}}}


