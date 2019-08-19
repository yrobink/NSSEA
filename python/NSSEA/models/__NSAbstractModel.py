# -*- coding: utf-8 -*-

###############
## Libraries ##
###############

#############
## Classes ##
#############

class NSAbstractModel:
	"""
	NSSEA.models.NSAbstractModel
	============================
	Main class to describe non-stationary model class, this class must only use to be herit!
	"""
	
	#################
	## Constructor ##
	#################
	
	def __init__( self ):
		pass
	
	
	###############
	## Accessors ##
	###############
	
	#############
	## Methods ##
	#############
	
	def fit( self , Y , X ):
		pass
	
	def set_covariable( self , X , t = None ):
		pass
	
	def check( self , Y , X , t = None ):
		pass
	
	def rvs( self , t ):
		pass
	
	def cdf( self , Y , t ):
		pass
	
	def icdf( self , q , t ):
		pass
	
	def sf( self , Y , t ):
		pass
	
	def isf( self , q , t ):
		pass

