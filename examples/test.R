
##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, Aurelien Ribes, 2020                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
## aurelien.ribes@meteo.fr                                                      ##
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
## Copyright Yoann Robin, Aurelien Ribes, 2020                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
## aurelien.ribes@meteo.fr                                                      ##
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

base::rm( list = base::ls() )


###############
## Libraries ##
###############

#library(devtools)
#library(roxygen2)
#roxygenize( "../R/NSSEA" )
#devtools::load_all( "../R/NSSEA" )
#
#library(abind)
library(ncdf4)


#######################
## Plotting function ##
#######################

PlotTools = R6::R6Class( "PlotTools" , ##{{{
	
	
	public = list(
	
	###############
	## Arguments ##
	###############
	
	os     = NULL,
	device = NULL,
	
	#################
	## Constructor ##
	#################
	
	initialize = function()
	{
		self$os = self$get_os()
		self$device = self$get_device()
	},
	
	
	#############
	## Methods ##
	#############
	
	get_os = function()
	{
		sysinf = base::Sys.info()
		if( !is.null(sysinf) )
		{
			os = sysinf['sysname']
			if( os == 'Darwin' ) os = "osx"
		}
		else
		{
			## mystery machine
			os = .Platform$OS.type
			if( base::grepl( "^darwin"   , R.version$os ) ) os = "osx"
			if( base::grepl( "linux-gnu" , R.version$os ) ) os = "linux"
		}
		invisible(tolower(os))
	},
	
	get_device = function()
	{
		if( self$os == "osx" )
			return(grDevices::quartz)
		else
			return(grDevices::X11)
	},
	
	figure = function( nrow = 1 , ncol = 1 , ratio = 12 / 9 , npix = 6 )
	{
		kwargs = list( width = ratio * npix , height = npix , pointsize = 8 )
		
		base::do.call( self$device , kwargs )
		
		graphics::par( mfrow = base::c( nrow , ncol ) )
	},
	
	wait = function()
	{
		while( base::names(grDevices::dev.cur()) != 'null device' ) base::Sys.sleep(1)
	}
	
	)
)
##}}}

plt = PlotTools$new()


###############
## Functions ##
###############

is_nc_file = function( file )##{{{
{
	return( tools::file_ext(file) == "nc" )
}
##}}}

path_to_model_name = function( path )##{{{
{
	path_wo_ext = tools::file_path_sans_ext(path)
	path_split  = base::unlist( base::strsplit( path_wo_ext , "_" ) )
	n_split     = base::length(path_split)
	mod         = base::paste( path_split[n_split-1] , "_" , path_split[n_split] , sep = "" )
	return(mod)
}
##}}}

read_data = function( path )##{{{
{
	l_files_X = base::list.files( base::file.path(pathInp,"X") )
	l_files_Y = base::list.files( base::file.path(pathInp,"Y") )
	
	
	## Read X names
	##=============
	l_models_X = list()
	i = 1
	for( fX in l_files_X )
	{
		if( !is_nc_file(fX) )
			next
		mod = path_to_model_name(fX)
		l_models_X[[i]] = mod
		i = i + 1
	}
	l_models_X = base::unlist(l_models_X)
	
	## Read Y names
	##=============
	l_models_Y = list()
	i = 1
	for( fY in l_files_Y )
	{
		if( !is_nc_file(fY) )
			next
		mod = path_to_model_name(fY)
		l_models_Y[[i]] = mod
		i = i + 1
	}
	l_models_Y = base::unlist(l_models_Y)
	
	## Merge its
	##==========
	condX = l_models_X %in% l_models_Y
	condY = l_models_Y %in% l_models_X
	
	l_models_X = l_models_X[condX]
	l_models_Y = l_models_Y[condY]
	models = base::sort(l_models_X)
	
	
	## And now read ncfiles
	##=====================
	lX = list()
	lY = list()
	i  = 1
	for( m in models )
	{
		## Read X
		fX     = base::file.path( pathInp , "X" , base::paste( "tas_mon_" , m , ".nc" , sep = "" ) )
		ncfile = nc_open(fX)
		X = matrix( ncvar_get( ncfile , "tas" ) , ncol = 1 )
		base::rownames(X) = as.integer( base::floor( ncvar_get( ncfile , "time" ) ) )
		base::colnames(X) = m
		lX[[i]] = X
		nc_close(ncfile)
		
		## Read Y
		fY     = base::file.path( pathInp , "Y" , base::paste( "tas_mon_" , m , ".nc" , sep = "" ) )
		ncfile = nc_open(fY)
		Y = matrix( ncvar_get( ncfile , "tas" ) , ncol = 1 )
		base::rownames(Y) = as.integer( base::floor( ncvar_get( ncfile , "time" ) ) )
		base::colnames(Y) = m
		lY[[i]] = Y
		nc_close(ncfile)
		
		i = i + 1
	}
	
	## Read Xo
	##========
	ncfile = nc_open( base::file.path( path , "Xo.nc" ) )
	Xo   = ncvar_get( ncfile , "temperature_anomaly" )
	base::rownames(Xo) = 1850 + base::floor( ncvar_get( ncfile , "time" ) / 365 )
	nc_close(ncfile)
	
	## Read Yo
	##========
	ncfile = nc_open( base::file.path( path , "Yo.nc" ) )
	Yo   = ncvar_get( ncfile , "temperature_anomaly" )
	base::rownames(Yo) = 1850 + base::floor( ncvar_get( ncfile , "time" ) / 365 )
	nc_close(ncfile)
	
	return( list( models = models , lX = lX , lY = lY , Xo = Xo , Yo = Yo ) )
}
##}}}


##########
## Main ##
##########

## Test or not
##============
is_test = FALSE
if( is_test )
	base::set.seed(42) ## Always the big answer when we set the seed

## Path
##=====
basepath = base::getwd()
pathInp  = base::file.path( basepath , "input/Normal" )
pathOut  = base::file.path( basepath , "output/Normal" )

## Read models
##============
data = read_data( pathInp )

## In anomaly
##===========
for( i in 1:length(data$models) )
{
	X = data$lX[[i]]
	data$lX[[i]] = X - base::mean( subset( X , base::rownames(X) %in% 1961:1990 ) )
	
	Y = data$lY[[i]]
	data$lY[[i]] = Y - base::mean( subset( Y , base::rownames(Y) %in% 1961:1990 ) )
}




#plt$figure( nrow = 2 , ncol = 1 )
#plot( rownames(data$Xo) , data$Xo , col = "red" )
#plot( rownames(data$Yo) , data$Yo , col = "blue" )
#
#plt$wait()

#print(pathInp)


base::cat("Done\n")

