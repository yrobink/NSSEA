
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

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
	
	os = NULL,
	
	
	#################
	## Constructor ##
	#################
	
	initialize = function()
	{
		self$os = self$get_os()
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
	
	new_screen = function( nrow = 1 , ncol = 1 , ratio = 12 / 9 )
	{
		if( self$os == "osx" )
		{
			grDevices::quartz( width = ratio * 6 * nrow , height = 6 * ncol )
		}
		if( self$os == "linux" )
		{
			grDevices::X11()
		}
		
		graphics::par( mfrow = base::c( nrow , ncol ) )
	},
	
	wait = function()
	{
		while( base::names(grDevices::dev.cur()) !='null device' ) base::Sys.sleep(1)
	}
	
	)
)
##}}}

plt = PlotTools$new()


###############
## Functions ##
###############

read_data = function()
{
	
	## Read Xo
	##========
	ncfile = nc_open( base::file.path( pathInp , "Xo.nc" ) )
	Xo   = ncvar_get( ncfile , "temperature_anomaly" )
	base::rownames(Xo) = 1850 + base::floor( ncvar_get( ncfile , "time" ) / 365 )
	nc_close(ncfile)
	
	## Read Yo
	##========
	ncfile = nc_open( base::file.path( pathInp , "Yo.nc" ) )
	Yo   = ncvar_get( ncfile , "temperature_anomaly" )
	base::rownames(Yo) = 1850 + base::floor( ncvar_get( ncfile , "time" ) / 365 )
	nc_close(ncfile)
	
	return( list( Xo = Xo , Yo = Yo ) )
}



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

data = read_data()

plt$new_screen()
plot( data$Xo , col = "red" )
points( data$Yo , col = "blue" )

plt$wait()

print(pathInp)


base::cat("Done\n")

