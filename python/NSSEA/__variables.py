# -*- coding: utf-8 -*-

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

import numpy   as np
import xarray  as xr
import netCDF4 as nc
import SDFC.tools as sdt

from .models.__Normal import Normal
from .models.__GEV    import GEV
from .models.__GEVMin import GEVMin


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

class Climatology2: ##{{{
	
	def __init__( self , time , models , n_sample , ns_law ): ##{{{
		samples = ["BE"] + [ 'S{0:{fill}{align}{n}}'.format(i,fill="0",align=">",n=int(np.floor(np.log10(n_sample))+1)) for i in range(n_sample)]
		self.data   = xr.Dataset( { "time" : time , "model" : models , "sample" : samples } )
		self.ns_law = ns_law
		self.be_is_median = False
	##}}}
	
	def _add_variable( self , name , variable ):##{{{
		if name in self.data.variables:
			self.data[name] = variable
		else:
			self.data = self.data.assign( { name : variable } )
	##}}}
	
	def copy(self): ##{{{
		clim = Climatology2( self.time , self.model , self.n_sample , self.ns_law )
		clim.data = self.data.copy()
		try:
			clim.synthesis = self.synthesis.copy()
		except:
			pass
		return clim
	##}}}
	
	def keep_models( self , models ): ##{{{
		if type(models) is not list: models = [models]
		self.data = self.data.sel( model = models , drop = False )
	##}}}
	
	def remove_models( self , models ):##{{{
		if type(models) is not list: models = [models]
		kept = [ m for m in self.model if m not in models ]
		self.keep_models(keps)
	##}}}
	
	## Generic properties {{{
	
	@property
	def time(self):
		return self.data.time.values
	
	@property
	def n_time(self):
		return self.data.time.size
	
	@property
	def n_model(self):
		return self.data.model.size
	
	@property
	def model(self):
		return self.data.model.values
	
	@property
	def sample(self):
		return self.data.sample.values
	
	@property
	def n_sample(self):
		return self.data.sample.size -1
	
	@property
	def n_coef(self):
		try:
			return self.data.coef.size
		except:
			return 0
	
	##}}}
	
	## Covariate properties {{{
	
	@property
	def X(self):
		try:
			return self.data.X
		except:
			return None
	
	@X.setter
	def X( self , X_ ):
		self._add_variable( "X" , X_ )
	
	##}}}
	
	## law params properties {{{
	
	@property
	def law_coef(self):
		try:
			return self.data.law_coef
		except:
			return None
	
	@law_coef.setter
	def law_coef( self , law_coef_ ):
		self._add_variable( "law_coef" , law_coef_ )
	
	##}}}
	
	## statistics properties {{{
	
	@property
	def statistics(self):
		try:
			return self.data.statistics
		except:
			return None
	
	@statistics.setter
	def statistics( self , stats ):
		self._add_variable( "statistics" , stats )
	
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
		nclaw = clim.ns_law.to_netcdf()
		for p in nclaw:
			ncFile.setncattr_string( str(p) , str(nclaw[p]) )
		
##}}}

def from_netcdf( ifile , ns_law = None ):##{{{
	with nc.Dataset( ifile , "r" ) as ncFile:
		
		## Extract dimensions
		##===================
		time      = np.ma.getdata( ncFile.variables["time"][:] )
		sample    = ncFile.variables["sample"][:]
		forcing   = ncFile.variables["forcing"][:]
		models    = ncFile.variables["models"][:]
		ns_param  = ncFile.variables["ns_param"][:]
		stats     = ncFile.variables["stat"][:]
		reference = np.ma.getdata( ncFile.variables["reference"][:] )
		
		if ns_law is None:
			## Extract ns_law attributes
			##==========================
			ns_law_attr = []
			for p in ncFile.__dict__:
				if "ns_law" in p: ns_law_attr.append(p)
			
			## Transform ns_law attributes to ns_law params
			##=============================================
			ns_law_kwargs = {}
			for p in ns_law_attr:
				if "ns_law_param" in p:
					pn,pk = p.split("_")[-2:]
					if pk == "cst":
						ns_law_kwargs[ pn + "_cst" ] = ncFile.__dict__[p] == str(True)
					if pk == "link":
						if ncFile.__dict__[p] == str(sdt.ExpLink()):
							ns_law_kwargs[ "l_" + pn ] = sdt.ExpLink()
						else:
							ns_law_kwargs[ "l_" + pn ] = sdt.IdLink()
			
			ns_law = None
			if ncFile.__dict__["ns_law_name"] == "Normal":
				ns_law = Normal( **ns_law_kwargs )
			elif ncFile.__dict__["ns_law_name"] == "GEV":
				ns_law = GEV( **ns_law_kwargs )
			elif ncFile.__dict__["ns_law_name"] == "GEVMin":
				ns_law = GEVMin( **ns_law_kwargs )
		
		
		##  Set climatology
		##=================
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


