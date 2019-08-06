
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

class LinkParams_old:##{{{
	
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

class LinkParams:##{{{
	
	def _add_true_extreme( self , X ):##{{{
		for i,x in enumerate(X):
			if x and i > 0:
				X[i-1] = True
				break
		for i,x in enumerate(X[::-1]):
			if x and i > 0:
				X[X.size-i] = True
				break
		return X
	##}}}
	
	class A:
		def __init__(self):
			self.values = None
			self.names  = None
	
	def __init__( self , rr_min = 1e-300 , rr_max = 1e300 , p_min = 1e-300 , p_max = 1 ):##{{{
		self._rrticks = np.array([1e-300, 1/100,1/10, 1/3, 1/2, 2/3, 1, 1.5, 2, 3, 10, 100, 1e300])
		self._rrlab   = np.array( ["0", "10^{-2}","10^{-1}", "1/3", "1/2", "2/3", "1", "1.5", "2", "3", "10", "10^2", "\infty"] )
		self._farlab  = np.array( ["-\infty","-99","-9", "-2" , "-1" , "-0.5" , "0" , "1/3" , "1/2" , "2/3" , "0.9" , "0.99" , "1"] )
		self._rr_idx = self._add_true_extreme( np.logical_not( np.logical_or( self._rrticks < rr_min , self._rrticks > rr_max ) ) )
		self.rrticks = self._rrticks[self._rr_idx]
		self.rrlab   = self._rrlab[self._rr_idx]
		self.farlab  = self._farlab[self._rr_idx]
		
		self._pticks = np.array( [ 1e-300 , 1e-5	  ,  1e-2   ,   1e-1  , 1/5 , 1/3 , 1/2 , 1 ] )
		self._plab   = np.array( ["0","10^{-5}","10^{-2}","10^{-1}","1/5","1/3","1/2","1"] )
		self._p_idx = self._add_true_extreme( np.logical_not( np.logical_or( self._pticks < p_min , self._pticks > p_max ) ) )
		self.pticks = self._pticks[self._p_idx]
		self.plab   = self._plab[self._p_idx]
		
		
		self.rr         = LinkParams.A()
		self.rr.values  = self.frr(self.rrticks)
		self.rr.names   = [ r"${}$".format(x) for x in self.rrlab ]
		self.far        = LinkParams.A()
		self.far.values = self.frr(1 - 1. / self.rrticks)
		self.far.names  = [ r"${}$".format(x) for x in self.farlab ]
		self.p          = LinkParams.A()
		self.p.values   = self.fp(self.pticks)
		self.p.names    = [ r"${}$".format(x) for x in self.plab ]
	##}}}
	
	def _log_no_warn( self , x ):
		x = np.array(x)
		x[ np.logical_not( x > 0 ) ] = sys.float_info.epsilon
		return np.log(x)
	
	def frr( self , x ):
		return np.arctan( self._log_no_warn(x) ) / (np.pi / 2.)
	
	def fp( self , x ):
		return np.power( 1. + self.frr(x) , 1. / 1.2 )
##}}}

