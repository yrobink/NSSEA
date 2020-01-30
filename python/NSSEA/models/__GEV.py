# -*- coding: utf-8 -*-

###############
## Libraries ##
###############

import numpy             as np
import scipy.stats       as sc
import scipy.interpolate as sci
import scipy.special     as scs
import SDFC              as sd
import SDFC.tools        as sdt

from .__AbstractModel import AbstractModel


#############
## Classes ##
#############

class GEV(AbstractModel):
	def __init__( self , loc_cst = False , scale_cst = False , shape_cst = True , **kwargs ):
		l_scale = kwargs.get("l_scale")
		if l_scale is None: l_scale = sdt.ExpLink()
		lparams = []
		lparams.append( { "name" : "loc"   , "is_cst" :   loc_cst , "link" : kwargs.get("l_loc")   , "name_tex" : r"\mu"    } )
		lparams.append( { "name" : "scale" , "is_cst" : scale_cst , "link" : l_scale               , "name_tex" : r"\sigma" } )
		lparams.append( { "name" : "shape" , "is_cst" : shape_cst , "link" : kwargs.get("l_shape") , "name_tex" : r"\xi"    } )
		AbstractModel.__init__( self , "GEV" , sc.genextreme , sd.GEV , lparams , **kwargs )
	
	def loct( self , t ):
		return self.lparams["loc"](t)
	
	def scalet( self , t ):
		return self.lparams["scale"](t)
	
	def shapet( self , t ):
		return self.lparams["shape"](t)
	
	def meant( self , t ):##{{{
		shapet = self.shapet(t)
		idx = np.abs(shapet) > 1e-8
		cst = np.zeros(shapet) + np.euler_gamma
		cst[idx] = ( scs.gamma( 1 - shapet[idx] ) - 1 ) / shapet[idx]
		return self._loct(t) + self._scalet(t) * cst
	##}}}
	
	def mediant( self , t ):##{{{
		return self.loct(t) + self.scalet(t) * ( np.pow( np.log(2) , - self.shapet(t) ) - 1. ) / self.shapet(t)
	##}}}

	def _get_sckwargs( self , t ):##{{{
		sckwargs = AbstractModel._get_sckwargs( self , t )
		sckwargs["c"] = - sckwargs["shape"]
		del sckwargs["shape"]
		return sckwargs
	##}}}

