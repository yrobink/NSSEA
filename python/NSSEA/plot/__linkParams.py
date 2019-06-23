
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############

import numpy  as np
import sys


#############
## Classes ##
#############

class LinkParams:##{{{
	
	class A:
		def __init__(self):
			self.values = None
			self.names  = None
	
	def __init__( self ):
		self._ticks     = np.array([1e-300, 1/100,1/10, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 10, 100, 1e300])
		self.rr         = LinkParams.A()
		self.rr.values  = self.frr(self._ticks)
		self.rr.names   = [ r"${}$".format(x) for x in ["0", "10^{-2}","10^{-1}", "1/3", "1/2", "2/3", "1", "1.5", "2", "3", "10", "10^2", "\infty"] ]
		self.far        = LinkParams.A()
		self.far.values = self.frr(1 - 1. / self._ticks)
		self.far.names  = [ r"${}$".format(x) for x in ["-\infty","-99","-9", "-2" , "-1" , "-0.5" , "0" , "1/3" , "1/2" , "2/3" , "0.9" , "0.99" , "1"] ]
		self.p          = LinkParams.A()
		self.p.values   = self.fp([ 1e-300 , 1e-5	  ,  1e-2   ,   1e-1  , 1/5 , 1/3 , 1/2 , 1 ])
		self.p.names    = [ r"${}$".format(x) for x in ["0","10^{-5}","10^{-2}","10^{-1}","1/5","1/3","1/2","1"] ]
	
	def _log_no_warn( self , x ):
		x = np.array(x)
		x[ np.logical_not( x > 0 ) ] = sys.float_info.epsilon
		return np.log(x)
	
	def frr( self , x ):
		return np.arctan( self._log_no_warn(x) ) / (np.pi / 2.)
	
	def fp( self , x ):
		return np.power( 1. + self.frr(x) , 1. / 1.2 )
##}}}


