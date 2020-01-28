# -*- coding: utf-8 -*-


###############
## Libraries ##
###############

import numpy             as np
import scipy.stats       as sc
import scipy.interpolate as sci
import SDFC              as sd
import SDFC.tools        as sdt

from .__AbstractModel import AbstractModel


#############
## Classes ##
#############

class Normal(AbstractModel):
	def __init__( self , loc_cst = False , scale_cst = False , **kwargs ):
		lparams = []
		lparams.append( { "name" : "loc"   , "is_cst" :   loc_cst , "link" : kwargs.get("l_loc")   , "name_tex" : "\mu"    } )
		lparams.append( { "name" : "scale" , "is_cst" : scale_cst , "link" : kwargs.get("l_scale") , "name_tex" : "\sigma" } )
		AbstractModel.__init__( self , sc.norm , sd.Normal , lparams , **kwargs )
	
	def loct( self , t ):
		return self.lparams["loc"](t)
	
	def scalet( self , t ):
		return self.lparams["scale"](t)
	
	def meant( self , t ):
		return self.loct(t)
	
	def mediant( self , t ):
		return self.loct(t)



