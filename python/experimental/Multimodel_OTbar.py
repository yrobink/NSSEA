#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#################
## Yoann Robin ##
#################

################
## Librairies ##
################

import sys,os

import numpy as np
import pickle as pk
import texttable as tt
import scipy.stats as sc
import scipy.optimize as sco
import scipy.linalg as scl
import sklearn.datasets as skd

import matplotlib.pyplot as plt



###########
## Class ##
###########

def matrix_squareroot( M , disp = False ):##{{{
	Mh = scl.sqrtm( M , disp = disp )
	if not disp:
		Mh = Mh[0]
	return np.real(Mh)


class Barycenter:
	
	def __init__( self , maxit = 50 , tol = 1e-3 , verbose = False ):
		self._maxit   = maxit
		self._tol     = tol
		self._verbose = verbose
		
		self.weights = None
		
	def fit( self , lmean , lcov , weights = None , init_cov = None ):
		
		n_law = len(lmean)
		n_dim = lmean[0].size
		
		if weights is None: weights = np.ones( n_law )
		self.weights = np.array(weights) / np.sum(weights)
		
		## Fit mean
		self.mean = np.array( lmean ).T @ self.weights
		
		## Fit cov
		self.cov  = np.identity(n_dim) if init_cov is None else init_cov
		
		diff = 1. + self._tol
		nit  = 0
	
		while diff > self._tol and nit < self._maxit:
#			if self._verbose: print( "Optimal barycenter: {} (<{}), {} (>{})                                   ".format( nit,maxit,round(diff,4),tol) , end = "\r" )
			covn = self.brower( lcov )
			diff = np.linalg.norm(self.cov - covn)
			self.cov  = covn
			nit += 1 
		
	def brower( self , lcov ):
		root_S = matrix_squareroot(self.cov)
		Sn = np.zeros( lcov[0].shape )
		
		for i,cov in enumerate(lcov):
			Sn += self.weights[i] * matrix_squareroot( root_S @ cov @ root_S )
		
		return Sn



###############
## Fonctions ##
###############

def likelihood_normal( mean , cov , x ):
	z = x - mean
#	return (z**2).sum()
#	return np.log(np.linalg.det(cov)) + z @ np.linalg.solve( cov , z )
	icov = np.linalg.pinv(cov)
	l,_  = np.linalg.eig(cov)
	return np.log(l[l>0].prod()) + z @ icov @ z

def likelihood_im_normal( mean , cov , y ):
	loc   = mean[0]
	scale = mean[1]
	if not scale > 0 : return np.inf
	
	z = y - loc
	return np.sum( z**2 / scale**2 + np.log(scale) )

def optim_function( w , lmean , lcov , x , y , p = 0.2 ):
	w = w / np.sum(w)
	bar = Barycenter()
	bar.fit( lmean , lcov , weights = w )
	return p * likelihood_normal( bar.mean , bar.cov , x ) + ( 1 - p ) * likelihood_im_normal( bar.mean , bar.cov , y )


##########
## main ##
##########


if __name__ == "__main__":
	
	## Data
	n_law,n_dim = 5,2
	R = 10
	t = np.linspace( 0 , 2 * np.pi * ( 1 - 1 / n_law ) , n_law )
	lmean = [ R * np.array( [np.cos(t[i]) , np.sin(t[i]) ] ) + np.array([0,15]) for i in range(n_law) ]
	lcov  = [ np.array( [np.random.uniform(1e-3,5),1,1,np.random.uniform(1e-3,5)] ).reshape((2,2)) * skd.make_spd_matrix(n_dim) for _ in range(n_law) ]
	
	## Obs
	x_true = np.array( [0,5] , dtype = np.float )
	x = np.random.multivariate_normal( mean = x_true , cov = 2 * np.identity(2) )
	y = np.random.normal( loc = x_true[0] , scale = x_true[1] , size = 1000 )
	
	
	## Barycenter
	winit = np.ones(n_law)
	winit,_,_,_ = scl.lstsq( np.array(lmean).T , x )
	winit /= np.sum(winit)
	
	optim = sco.minimize( optim_function , x0 = winit , args = (lmean,lcov,x,y) , method = "L-BFGS-B" , bounds = [(0,None) for _ in range(n_law)] )
	w     = optim.x
	w /= np.sum(w)
	bar = Barycenter()
	bar.fit( lmean , lcov , weights = w )
	
	
	
	if True:
		## Histogram
		lmean.append( bar.mean )
		lcov.append( bar.cov )
		xmin,xmax = -15,15
		ymin,ymax = 0,30
		bins = [ np.linspace( xmin , xmax , 100 ) , np.linspace( ymin , ymax , 100 ) ]
		lH = []
		for m,S in zip(lmean,lcov):
			X = np.random.multivariate_normal( mean = m , cov = S , size = 10000 )
			H,_,_ = np.histogram2d( X[:,0] , X[:,1] , bins = bins )
			H /= np.sum(H)
			H[H==0] = np.nan
			lH.append(H)
			
		## Plot
		cmap = plt.cm.inferno
		fig = plt.figure()
		
		ax = fig.add_subplot( 1 , 1 , 1 )
		for H in lH[:-1]:
			ax.imshow( np.rot90(H) , extent = [xmin,xmax,ymin,ymax] , cmap = cmap )
			H = lH[-1]
			ax.imshow( np.rot90(H) , extent = [xmin,xmax,ymin,ymax] , cmap = plt.cm.jet )

	
	print("Done")
