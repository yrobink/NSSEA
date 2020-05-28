
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

library(devtools)
library(roxygen2)
roxygenize( "../R/NSSEA" )
devtools::load_all( "../R/NSSEA" )

library(abind)


###############
## Functions ##
###############

#coffee2netcdf = function( coffee , event , ofile , with_cx = FALSE , with_co = FALSE )
#{
#	
#	library(ncdf4)
#	dim_time     = ncdim_def( "time"     , "" , coffee$time              )
#	dim_sample   = ncdim_def( "sample"   , "" , dimnames(coffee$X)$sample            )
#	dim_forcing  = ncdim_def( "forcing"  , "" , dimnames(coffee$X)$forcing           )
#	dim_models   = ncdim_def( "models"   , "" , dimnames(coffee$X)$models            )
#	dim_ns_param = ncdim_def( "ns_param" , "" , dimnames(coffee$ns_params)$params )
#	dim_stat     = ncdim_def( "stat"     , "" , dimnames(coffee$stats)$stats         )
#	
##	ncFile = nc_create( ofile )
##	nc_close(ncFile)
#}


##########
## Main ##
##########

## Test or not
is_test = FALSE
if( is_test )
	base::set.seed(42) ## Always the big answer when we set the seed


## Path
pathInput = "data"
pathOut   = "/home/robiny/Local/scratch/NSSEA_R"
#pathOutput = "/home/yrobin/Local/scratch/EUPHEME/R"

## Command line arguments
code      = "yrobin"


## Global parameters
n_sample    = if( is_test ) 10 else 100
ns_law      = NSSEA::NSGaussianModel
ns_law_args = list( link_fct_scale = SDFC::IdLinkFct$new() )
gam_dof     = 7
verbose     = TRUE
time        = 1850:2100
n_time      = length(time)

event       = NSSEA::make_event( "EHW03" , pathOut , 2003 , 5. , 1961:1990 , "tas" , "high" )
cx_params   = NSSEA::make_CXParams( centering = TRUE , ref = 1961:1990 , trust = FALSE )


## Load models and covariates
base::load( base::file.path( pathInput , "X_full.Rdata" ) )
base::load( base::file.path( pathInput , "Y_full.Rdata" ) )
if( is_test )
{
	X_full = X_full[1:3]
	Y_full = Y_full[1:3]
}
Y = Y_full
models   = names(X_full)
n_models = length(models)

## In anomaly

## Load observations
base::load( base::file.path( pathInput , "Xo.Rdata" ) )
base::load( base::file.path( pathInput , "Yo.Rdata" ) )


## Aggregate covariate
Xd = array( NA , dim = base::c( n_time , n_models ) , dimnames = list( time = time , models = models ) )
for( i in 1:n_models )
{
	xf = X_full[[ models[i] ]]
	agg = stats::aggregate( xf , by = list( time = names(xf) ) , base::mean )
	Xd[,i] = agg$x[ agg$time %in% time ]
}


## Define coffee variable from input
coffee = NSSEA::make_coffee( time , n_sample , models , ns_law , ns_law_args )


### Split
Enat = NSSEA::ebm_response( coffee$time , coffee$n_sample )
X_splitted = NSSEA::gam_decomposition( X_full , Enat , event$time , gam_dof , verbose = verbose , code = code )
coffee$X = if( is.null(X_splitted$X_center) ) X_splitted$X else X_splitted$X_center


## NS fit
coffee = NSSEA::nslaw_fit( Y , coffee , verbose = verbose , code = code )


### Multi model
coffee = NSSEA::infer_multi_model( coffee , verbose = verbose )


## Apply constraints
coffeeCX   = NSSEA::constraints_CX( coffee   , Xo , cx_params , verbose = verbose )
coffeeC0   = NSSEA::constraints_C0( coffee   , Yo , event , verbose = verbose )
coffeeCXC0 = NSSEA::constraints_C0( coffeeCX , Yo , event , verbose = verbose )


### Compute stats
coffee     = NSSEA::extremes_stats( coffee     , event , verbose = verbose , code = code )
coffeeCX   = NSSEA::extremes_stats( coffeeCX   , event , verbose = verbose , code = code )
coffeeC0   = NSSEA::extremes_stats( coffeeC0   , event , verbose = verbose , code = code )
coffeeCXC0 = NSSEA::extremes_stats( coffeeCXC0 , event , verbose = verbose , code = code )

#if( FALSE )
#{
## Plot 
NSSEA::plot_X_dec( X_full , X_splitted$X ,  ofile = base::file.path( event$dir_name , "Covariate_decomposition.pdf" ) , verbose = verbose )	# X-decomposition
NSSEA::plot_ns_params_fitted( Y , coffee ,  ofile = base::file.path( event$dir_name , "ns_params_fitted.pdf"        ) , verbose = verbose )	# NS-fit
NSSEA::plot_stats_probability( coffee ,     ofile = base::file.path( event$dir_name , "Stats_probability.pdf"       ) , verbose = verbose )
NSSEA::plot_stats_intensity( coffee ,       ofile = base::file.path( event$dir_name , "Stats_intensity.pdf"         ) , verbose = verbose )
NSSEA::plot_relative( coffee , event$time , ofile = base::file.path( event$dir_name , "Stats_relative.pdf"          ) , verbose = verbose )


## Multi-model
#NSSEA::plot_multimodel( coffee , ofile = base::file.path( event$dir_name , "Multi-model.pdf" ) ) # Multi-model

## Constraints
NSSEA::plot_constraints_CX( coffee , coffeeCX , Xo , ref_plot = event$ref_anom , ofile = base::file.path( event$dir_name , "CX.pdf" ) , verbose = verbose )
NSSEA::plot_stats_probability( coffeeCXC0 ,                                      ofile = base::file.path( event$dir_name , "Stats_probability_CXC0.pdf" ) , verbose = verbose )
NSSEA::plot_stats_intensity( coffeeCXC0 ,                                        ofile = base::file.path( event$dir_name , "Stats_intensity_CXC0.pdf"   ) , verbose = verbose )


## Stats relative to the event
#NSSEA::plot_stats_time( coffee$stats     , event$time , ofile = base::file.path( event$dir_name , base::paste0("Diags_",event$time,".pdf"     ) ) , verbose = verbose )
#NSSEA::plot_stats_time( coffeeCX$stats   , event$time , ofile = base::file.path( event$dir_name , base::paste0("Diags_",event$time,"_CX.pdf"  ) ) , verbose = verbose )
#NSSEA::plot_stats_time( coffeeC0$stats   , event$time , ofile = base::file.path( event$dir_name , base::paste0("Diags_",event$time,"_C0.pdf"  ) ) , verbose = verbose )
#NSSEA::plot_stats_time( coffeeCXC0$stats , event$time , ofile = base::file.path( event$dir_name , base::paste0("Diags_",event$time,"_CXC0.pdf") ) , verbose = verbose )
#
#
#stats_tmp = abind::abind( coffee$stats[,,,"multi"],
#						coffeeCX$stats[,,,"multi"],
#						coffeeC0$stats[,,,"multi"],
#						coffeeCXC0$stats[,,,"multi"],
#						along = 4,
#						use.dnns = TRUE,
#						new.names = base::c( dimnames(coffee$stats)[1:3] , list( models = base::c("No obs","CX","C0","CX+C0") ) )
#						)
#
#NSSEA::plot_stats_time( stats_tmp , event$time , ofile = base::file.path( event$dir_name , "MultiC.pdf" ) , verbose = verbose )

#}

## Save
##base::save( list = base::c( "event" , "mm_method" , "code" , "Enat" , "coffee" , "coffeeCX" , "coffeeC0" , "coffeeCXC0" , "X_splitted" ) , file = base::file.path( event$dir_name , "Results.Rdata" ) )
#
#coffee2netcdf( coffee , event , ofile = base::file.path( event$dir_name , "Coffee.nc" ) )


##########
## TODO ##
##########

#=> MM-plots with plumes (ie single model + aggregation in the same graph)
#=> source("illustrate_C0.R") ## Illutrstaion of C0


#########
## End ##
#########

cat("Done\n")

