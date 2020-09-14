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
	
	def __init__( self , name , time , reference , anomaly = None , type_ = "threshold" , side = "upper" , variable = "variable" , unit = "U" ):
		"""
		Constructor of Event
		
		Arguments
		---------
		
		name          : [str] Name of event
		time          : [time_index] Time when event occured
		reference     : [array] Time period to considered as reference for anomaly
		anomaly       : [float] Anomaly of event
		type          : ["threshold" or "hard"] If we compute probabilities as a threshold beyond mean, or anomaly is used as a hard value.
		side          : [str] "upper" or "lower" extremes event
		name_variable : [str] Name of variable (temperature, precipitation, etc)
		unit          : [str] Unit of the variable
		"""
		self.name      = name
		self.time      = time
		self.anomaly   = anomaly
		self.reference = reference
		self.type      = type_ if type_ in ["threshold","hard"] else "threshold"
		self.side      = side if side in ["upper","lower"] else "upper"
		self.variable  = variable
		self.unit      = unit
	
	def __repr__(self):
		return self.__str__()
	
	def __str__(self):
		out = ""
		out += "Event     : {},\n".format(self.name)
		out += "variable  : {} ({}),\n".format(self.variable,self.unit)
		out += "time      : {},\n".format(self.time)
		out += "anomaly   : {},\n".format(self.anomaly)
		out += "reference : {}-{},\n".format(self.reference.min(),self.reference.max())
		out += "side      : {}\n".format(self.side)
		return out
##}}}

class Climatology: ##{{{
	
	def __init__( self , event , time , models , n_sample , ns_law ): ##{{{
		samples = ["BE"] + [ 'S{0:{fill}{align}{n}}'.format(i,fill="0",align=">",n=int(np.floor(np.log10(n_sample))+1)) for i in range(n_sample)]
		self.event  = event
		self.data   = xr.Dataset( { "time" : time , "model" : models , "sample" : samples , "anomaly_period" : event.reference } )
		self.ns_law = ns_law
		self.BE_is_median = False
	##}}}
	
	def _add_variable( self , name , variable ):##{{{
		if name in self.data.variables:
			self.data[name] = variable
		else:
			self.data = self.data.assign( { name : variable } )
	##}}}
	
	def copy(self): ##{{{
		clim = Climatology( self.event , self.time , self.model , self.n_sample , self.ns_law )
		clim.data = self.data.copy(deep=True)
		return clim
	##}}}
	
	def to_netcdf( self , ofile ): ##{{{
		self.data.attrs["BE_is_median"]   = str(self.BE_is_median)
		self.data.attrs["event.name"]     = self.event.name
		self.data.attrs["event.time"]     = self.event.time
		self.data.attrs["event.anomaly"]  = self.event.anomaly
		self.data.attrs["event.type"]     = self.event.type
		self.data.attrs["event.side"]     = self.event.side
		self.data.attrs["event.variable"] = self.event.variable
		self.data.attrs["event.unit"]     = self.event.unit
		self.data.to_netcdf(ofile)
	
	##}}}
	
	def from_netcdf( ifile , ns_law ):##{{{
		data = xr.open_dataset(ifile)
		event = Event( data.attrs["event.name"] , data.attrs["event.time"] , data.anomaly_period.values , data.attrs["event.anomaly"] , data.attrs["event.type"] , data.attrs["event.side"] , data.attrs["event.variable"] , data.attrs["event.unit"] )
		clim = Climatology( event , data.time.values , data.model.values , data.sample.size , ns_law )
		clim.data = data
		return clim
	##}}}
	
	def from_netcdf_v03( ifile , ns_law , BE_is_median ):##{{{
		## Load data
		data = xr.open_dataset( ifile )
		
		## Build dimensions
		sample = data.sample.values
		anomaly_period = data.reference.values
		coef    = data.ns_param.values.tolist()
		forcing = ["F","C"]
		time    = data.time.values
		model   = data.models.values.tolist()
		if "multi" in model:
			model[model.index("multi")] = "Multi_Synthesis"
		mm_coef = [ "{}F".format(t) for t in time ] + [ "{}C".format(t) for t in time ] + coef
		stats   = ["pC","pF","IC","IF","PR","dI"]
		
		## Build event
		event = Event( data.event_name , np.array([data.event_time] , dtype = time.dtype )[0] , anomaly_period , data.event_anomaly , data.event_type , data.event_side , data.event_variable , data.event_unit )
		
		## And now define climatology
		clim          = Climatology( event , time , model , sample.size - 1 , ns_law )
		clim.BE_is_median = BE_is_median
		clim.X        = xr.DataArray( data.X[:,:,:2,:].values , dims = ["time","sample","forcing","model"] , coords = [time,clim.sample,forcing,model] )
		clim.law_coef = xr.DataArray( data.ns_params.values , dims = ["coef","sample","model"] , coords = [coef,clim.sample,model] )
		
		mm_mean  = xr.DataArray( data.mm_mean.values , dims = ["mm_coef"] , coords = [mm_coef] )
		mm_cov   = xr.DataArray( data.mm_cov.values  , dims = ["mm_coef","mm_coef"] , coords = [mm_coef,mm_coef] )
		clim.data = clim.data.assign( { "mm_mean" : mm_mean , "mm_cov" : mm_cov } )
		
		clim.statistics = xr.DataArray( data.stats.loc[:,:,stats,:].values , dims = ["time","sample","stats","model"] , coords = [time,clim.sample,stats,model] )
		
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

