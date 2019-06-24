#!/usr/bin/python3
# -*- coding: utf-8 -*-


###############
## Libraries ##
###############

#from distutils.core import setup, find_packages
from setuptools import setup, find_packages


#################
## Compilation ##
#################

list_packages = [
	"NSSEA",
	"NSSEA.plot",
	"NSSEA.models",
]

setup(
	name = "NSSEA" ,
	version = "0.2.4" ,
	description = "",
	author = "Yoann Robin" ,
	author_email = "yoann.robin.k@gmail.com" ,
	license = "CeCILL-C" ,
	platforms = [ "linux" , "macosx" ] ,
	requires = [ "numpy(>=1.15.0)" , "scipy(>=0.19)" , "xarray" , "pandas" , "matplotlib" , "pygam(>=0.8.0)" , "netCDF4" , "SDFC(>=0.3.0)" ],
	packages = list_packages,
	package_dir = { "NSSEA" : "NSSEA" },
	include_package_data = True
)




