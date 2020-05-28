
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

#library(latex2exp)
#library(RColorBrewer)

## Extend R toolbox

## fill_between {{{{

#' fill_between
#'
#' Plot function, fill between Xl and Xu along t
#'
#' @usage fill_between( Xl , Xu , t , col , alpha )
#'
#' @param Xl [vector] lower limit
#' @param Xu [vector] upper limit
#' @param t [vector] x axis
#' @param col [string or RGB] color
#' @param alpha [double] Transparency, double between 0 and 1.
#'
#' @return NULL
#'
#' @examples
#'
#' t = base::seq( 0 , 1 , length = 1000 )
#' X = t^2
#' Xl = X - 1.
#' Xu = X + 1.
#' plot( t , X , type = "l" , col = "red" )
#' fill_between( Xl , Xu , t , col = "red" , alpha = 0.5 )
#' 
#' @export
fill_between = function( Xl , Xu , t , col , alpha = 0.5 )
{
	col = if( is.character(col) ) grDevices::col2rgb( col ) / 255 else col
	graphics::polygon( base::c(t,base::rev(t)) , base::c(Xl,base::rev(Xu)) , col = grDevices::rgb( col[1] , col[2] , col[3] , alpha ) , border = NA )
}
##}}}


## 

## plot_X_dec {{{

#' plot_X_dec
#'
#' Plot decomposition of covariable
#'
#' @importFrom grDevices rgb
#'
#' @usage plot_X_dec( X_full , X_abs , ofile , verbose )
#'
#' @param X_full [list] list of covariates
#'
#' @param X_abs [array] Result of gam decomposition
#'
#' @param ofile [str] output file
#'
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_X_dec = function( X_full , X_abs , ofile , verbose = FALSE )
{
	if( verbose ) cat( "plot_X_dec        \r" )
	time     = as.numeric( dimnames(X_abs)$time )
	models   = dimnames(X_abs)$models
	n_models = length(models)
	X_q95    = base::apply( X_abs[,-1,,] , base::c(1,3,4) , stats::quantile , .95 )
	X_q05    = base::apply( X_abs[,-1,,] , base::c(1,3,4) , stats::quantile , .05 )
	
	grDevices::pdf(ofile)
	graphics::layout( matrix( 1:3 , nrow = 3 , ncol = 1 ) )
	
	for( imod in 1:n_models)
	{
		# Panel1: X_all and X_full (all data))
		time_full = as.numeric(names(X_full[[imod]]))
		idx_time = time_full %in% time
		
		
		graphics::par( fig = base::c(0,1,.65,1) , mgp = base::c(2.5,.7,0) , cex = 1 , font = 2 , mar = base::c(1,4,2,1) , font.axis = 2 , font.lab = 2 , tcl = -.4 , las = 1 )
		graphics::plot( time_full[idx_time] , X_full[[imod]][idx_time] , type = "p" , pch = 16 , xlab = "" , ylab = "T (K)" , cex = .4 , main = base::toupper( models[imod] ) , xaxt = "n" )
		graphics::axis( 1 , labels = FALSE )
		NSSEA::fill_between( X_q05[,"all",imod] , X_q95[,"all",imod] , time , col = rgb(1,0,0) , alpha = 0.5 )
		graphics::lines( time , X_abs[,"be","all",imod] , col = "brown" , lwd = 1 )
		graphics::mtext( "ALL" , adj = .02 , line = -1.2 )
		
		# Panel 2: X_ant
		graphics::par( fig = base::c(0,1,.38,.65) , new = TRUE , mar = base::c(1,4,0,1) )
		graphics::plot( time , X_abs[,"be","ant",imod] , type = "l" , col = "forestgreen" , xlab = "" , ylab = "T (K)" , font = 2 , font.lab = 2 , ylim = base::range( base::c( X_q05[,"ant",imod] , X_q95[,"ant",imod] ) ) , xaxt = "n" )  #xaxs="i",
		graphics::axis( 1 , labels = FALSE )
		NSSEA::fill_between( X_q05[,"ant",imod] , X_q95[,"ant",imod] , time , col = rgb(0,1,0) , alpha = 0.5 )
		graphics::abline( h = 0 )
		graphics::mtext( "ANT" , adj = .02 , line = -1.2 )
		
		# Panel 3: X_nat
		graphics::par( fig = base::c(0,1,0,.38) , new = TRUE , mar = base::c(4,4,0,1) )
		graphics::plot( time_full[idx_time] , X_full[[imod]][idx_time] , type = "p" , pch = 16 , xlab = "" , ylab = "T (K)" , font = 2 , cex = .4 , font.lab = 2 )
		graphics::title( xlab = "Years" , mgp = base::c(2,.7,0) )
		NSSEA::fill_between( X_q05[,"nat",imod] , X_q95[,"nat",imod] , time , col = rgb(0,0,1) , alpha = 0.5 )
		graphics::lines( time , X_abs[,"be","nat",imod] , col = "blue" )
		graphics::mtext( "NAT" , adj = .02 , line = -1.2 )
	}
	grDevices::dev.off()
	if( verbose ) cat( "plot_X_dec (Done)       \n" )
}
##}}}

## plot_ns_params_fitted {{{


#' plot_ns_params_fitted
#'
#' Plot ns params fitted
#'
#' @importFrom latex2exp TeX
#'
#' @usage  plot_ns_params_fitted( lY , coffee , ofile , verbose )
#'
#' @param lY [list] list of models
#'
#' @param coffee [coffee list] coffee variable
#'
#' @param ofile [str] output file
#'
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_ns_params_fitted = function( lY , coffee , ofile , verbose = FALSE )
{
	if( verbose ) cat( "plot_ns_params_fitted       \r" )
	models   = coffee$models
	n_models = coffee$n_models
	time     = coffee$time
	
	if( "multi" %in% models )
		n_models = n_models - 1
	
	
	grDevices::pdf(ofile)
	graphics::par( cex = 1.2 , mgp = base::c(2.5,.7,0) )
	for( imod in 1:n_models )
	{
		Y      = lY[[imod]]
		time_Y = as.numeric(names(Y))
		X      = coffee$X[as.character(time_Y),"be","all",imod]
		
		# Data for NS-fit
		xlab = latex2exp::TeX("$\\mathbf{x_t^{all}}$ \\textbf{\\ \\ \\ (K)}")
		ylab = latex2exp::TeX("$\\mathbf{y}$ \\textbf{\\ \\ \\ (K)}")
		graphics::plot( X , Y , type = "p" , pch = 16 , cex = .5 , xlab = xlab , ylab = ylab , main = base::toupper(models[imod]) , font.lab = 3 ,font = 2 , cex.lab = 1.2 , cex.axis = 1.2 )
		
		# Expectation effect 'mu'
		mu = coffee$ns_params[1,"be",imod] + coffee$ns_params[2,"be",imod] * base::range(X)
		graphics::lines( base::range(X) , mu , col = "red" , lwd = 3 )
		
		# Variance effect -- p05 and q95 WARNING: the following is correct only if sig_link=="identity"
		sig   = coffee$ns_params[3,"be",imod] + coffee$ns_params[4,"be",imod] * base::range(X)
		Y_q05 = mu - 1.64 * sig
		Y_q95 = mu + 1.64 * sig
		graphics::lines( base::range(X) , Y_q05 , col = "red" , lwd = 3 , lty = 2 )
		graphics::lines( base::range(X) , Y_q95 , col = "red" , lwd = 3 , lty = 2 )
	}
	grDevices::dev.off()
	if( verbose ) cat( "plot_ns_params_fitted (Done)       \n" )
}
##}}}

## link_params {{{

#' link_params
#'
#' Return link toolbox for FAR and RR
#'
#' @usage  link_params()
#'
#' @return link [list]
#'
#' @examples
#' ##
#' 
#' @export
link_params = function()
{
	## Link functions
	link    = list()
	link$rr = function(x) { base::atan( base::log(x) ) / (base::pi / 2) }	# Define the link function for RR
	link$p  = function(x) { ( 1. + base::atan( base::log(x) ) / (base::pi / 2) )^(1./1.2) }
	
	## Ticks
	ticks = list()
	ticks$rr         = list()
	ticks$rr$names   = base::c("0", "1/100","1/10", "1/3", "1/2", "2/3", "1", "1.5", "2", "3", "10", "100", "Inf")
	ticks$rr$values  = base::sapply( ticks$rr$names , NSSEA::eval_str )
	
	ticks$far        = list()
	ticks$far$values = 1. - 1. / ticks$rr$values
	ticks$far$names  = as.character( base::round( ticks$far$values , digit = 2 ) )
	
	ticks$p1         = list()
	ticks$p1$values  = base::c( 0 , 1e-5	  ,  1e-2   ,   1e-1  , 1/5 , 1/3 , 1/2 , 1 )
	ticks$p1$names   = base::c("0","10^{-5}","10^{-2}","10^{-1}","1/5","1/3","1/2","1")
	
	ticks$p2         = list()
	ticks$p2$values  = base::c( 0 , 1e-6   ,1e-5,1e-4,1e-3,  1e-2	  ,5e-2,1/30,1/20,  1e-1   , 1/5 , 1/3 , 1/2 , 1 )
	ticks$p2$names   = base::c("0","10^{-6}", "",  "",  "","10^{-2}",  "",  "",  "","10^{-1}","1/5","1/3","1/2","1")
	
	return( list( link = link , ticks = ticks ) )
}
##}}}


## Statistics plot

## plot_stats_probability {{{

#' plot_stats_probability
#'
#' Plot pall, pnat and RR/FAR
#'
#' @importFrom latex2exp TeX
#' @importFrom grDevices rgb
#'
#' @usage  plot_stats_probability( coffee , ofile , verbose )
#'
#' @param coffee [coffee list] coffee variable
#'
#' @param ofile [str] output file
#'
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_stats_probability = function( coffee, ofile , verbose = FALSE )
{
	if( verbose ) cat( "plot_stats_probability       \r" )
	time      = coffee$time
	models    = coffee$models
	n_models  = coffee$n_models
	stats_q05 = base::apply( coffee$stats[,-1,,,drop=FALSE] , base::c(1,3,4) , stats::quantile , .05 )
	stats_q95 = base::apply( coffee$stats[,-1,,,drop=FALSE] , base::c(1,3,4) , stats::quantile , .95 )
	
	dh = 0
	figh = 9 + dh
	grDevices::pdf( ofile , width = 7 , height = figh )
	graphics::layout( matrix( 1:3, nrow = 3 , ncol = 1 ) )
	graphics::par( cex = 1.2 , font = 2 , font.axis = 2 , font.lab = 2 , cex.axis = .75 , mgp = base::c(2.5,.7,0) )
	
	lt = link_params()
	link  = lt$link
	ticks = lt$ticks
	
	# Loop on models
	for( imod in 1:n_models )
	{
		# Panel 1: p1
		graphics::par( fig = base::c(0,1,1-3.15/figh,1) , mar = base::c(1,3.7,2,3.3) , las = 1 , tcl = -.4 )
		graphics::plot( time , link$p(coffee$stats[,"be","pall",imod]) , type = "l" , lwd = 2 , col = "brown" , ylim = base::c(0,1.05) , xlab = "" , ylab = TeX("\\textbf{$p_1(t)$}") , xaxt = "n" , yaxt = "n" , xaxs = "i" , yaxs = "i" , main = base::toupper(models[imod]) )
		graphics::axis( 1 , label = FALSE )
		graphics::axis( 2 , at = link$p(ticks$p1$values) , labels = TeX(base::paste0("\\textbf{",ticks$p1$names,"}")) )
		NSSEA::fill_between( link$p(stats_q05[,"pall",imod]) , link$p(stats_q95[,"pall",imod]) , time , col = rgb(1,0,0) , alpha = 0.5 )
		
		# Panel 2: p0
		graphics::par( fig = base::c(0,1,1-5.58/figh,1-3.15/figh) , new = TRUE , mar = base::c(1,3.7,0,3.3) )
		graphics::plot( time , link$p(coffee$stats[,"be","pnat",imod]) , type = "l" , lwd = 2 , col = "brown" , ylim = base::c(0,1.05) , xlab = "" , ylab = TeX("\\textbf{$p_0(t)$}") , xaxt = "n" , yaxt = "n" , xaxs = "i" , yaxs = "i" )
		graphics::axis( 1 , label = FALSE )
		graphics::axis( 2 , at = link$p(ticks$p1$values) , labels = TeX(paste0("\\textbf{",ticks$p1$names,"}")) )
		NSSEA::fill_between( link$p(stats_q05[,"pnat",imod]) , link$p(stats_q95[,"pnat",imod]) , time , col = rgb(1,0,0) , alpha = 0.5 )

		# Panel 3: RR/FAR
		graphics::par( fig = base::c(0,1,0,.38) , new = TRUE , mar = base::c(3,3.7,0,3.3) )
		graphics::plot( time , link$rr(coffee$stats[,"be","rr",imod]) , type = "l" , lwd = 2 , ylim = base::c(-1,1) , yaxt = "n" , col = "brown" , ylab = "" , xlab = "Year" , xaxs = "i" , yaxs = "i" , mgp = base::c(1.7,.4,0) )
		graphics::title( ylab = TeX("\\textbf{RR(t)}") )
		NSSEA::fill_between( link$rr(stats_q05[,"rr",imod]) , link$rr(stats_q95[,"rr",imod]) , time , col = rgb(1,0,0) , alpha = 0.5 )
		graphics::lines( time , link$rr(coffee$stats[,"be","rr",imod]) , col = "brown" , lwd = 2 )
		graphics::axis( 2 , at = link$rr(ticks$rr$values) , labels = TeX(paste0("\\textbf{",ticks$rr$names,"}")) )
		graphics::axis( 4 , at = link$rr(ticks$rr$values) , labels = ticks$far$names)
		graphics::abline( h = 0 )
		xlim = graphics::par("usr")
		x_label_right = xlim[2] + ( xlim[2] - xlim[1] ) * 2.6 / graphics::par("mar")[4] * graphics::par("mai")[4] / graphics::par("pin")[1]
		graphics::text( x_label_right , 0 , "FAR(t)" , xpd = TRUE , srt = 270 )
	}
	grDevices::dev.off()
	if( verbose ) cat( "plot_stats_probability (Done)      \n" )
}
##}}}

## plot_stats_intensity {{{

#' plot_stats_intensity
#'
#' Plot iall, inat and di
#'
#' @importFrom latex2exp TeX
#' @importFrom grDevices rgb
#'
#' @usage  plot_stats_intensity( coffee , ofile , verbose )
#'
#' @param coffee [coffee list] coffee variable
#'
#' @param ofile [str] output file
#'
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_stats_intensity = function( coffee , ofile , verbose = FALSE )
{
	if( verbose ) cat( "plot_stats_intensity      \r" )
	time      = coffee$time
	models    = coffee$models
	n_models  = coffee$n_models
	stats_q05 = base::apply( coffee$stats[,-1,,,drop=FALSE] , base::c(1,3,4) , stats::quantile , .05 )
	stats_q95 = base::apply( coffee$stats[,-1,,,drop=FALSE] , base::c(1,3,4) , stats::quantile , .95 )
	
	dh = 0
	figh = 9 + dh
	grDevices::pdf( ofile , width = 7 , height = figh )
	graphics::layout( matrix( 1:3 , nrow = 3 , ncol = 1 ) )
	graphics::par( cex = 1.2 , font = 2 , font.axis = 2 , font.lab = 2 , cex.axis = .75 , mgp = base::c(2.2,.6,0) )
	
	# Loop on models
	for( imod in 1:n_models )
	{
		# Panel 1: i1
		graphics::par( fig = base::c(0,1,1-3.15/figh,1) , mar = base::c(1,3.7,2,3.3) , las = 1 , tcl = -.4 )
		yll = base::range( coffee$stats[,"be",c("iall","inat"),imod] , stats_q95[,c("iall","inat"),imod] , stats_q05[,c("iall","inat"),imod] )
		graphics::plot( time , coffee$stats[,"be","iall",imod] , type = "l" , lwd = 2 , col = "brown" , ylim = yll , xlab = "" , ylab = TeX("$\\mathbf{i_1}(t)$ $$ $$ $$ (K)") , xaxt = "n" , xaxs = "i" , main = base::toupper(models[imod]) )
		graphics::axis( 1 , label = FALSE )
		NSSEA::fill_between( stats_q05[,"iall",imod] , stats_q95[,"iall",imod] , time , col = rgb(1,0,0) , alpha = 0.5 )
		
		# Panel 2: i0
		graphics::par( fig = base::c(0,1,1-5.58/figh,1-3.15/figh) , new = TRUE , mar = base::c(1,3.7,0,3.3) )
		graphics::plot( time , coffee$stats[,"be","inat",imod] , type = "l" , lwd = 2 , col = "brown" , ylim = yll , xlab = "" , ylab = TeX("$\\mathbf{i_0}(t)$ $$ $$ $$ (K)") , xaxt = "n" , xaxs = "i" )
		graphics::axis( 1 , label = FALSE )
		NSSEA::fill_between( stats_q05[,"inat",imod] , stats_q95[,"inat",imod] , time , col = rgb(1,0,0) , alpha = 0.5 )
		
		# Panel 3: di
		yll = base::range( coffee$stats[,"be","di",imod] , stats_q95[,"di",imod],stats_q05[,"di",imod] )
		graphics::par( fig = base::c(0,1,0,.38) , new = TRUE , mar = base::c(3,3.7,0,3.3) )
		graphics::plot( time , coffee$stats[,"be","di",imod] , type="l" , lwd = 2 , ylim = yll , yaxt = "n" , col = "brown" , ylab = "" , xlab = "Year" , xaxs = "i" , mgp = base::c(1.7,.4,0) )
		graphics::axis( 2 , labels = TRUE )
		graphics::title( ylab = TeX("$\\mathbf{\\delta i(t)}$ $$ $$ $$ (K)") )
		NSSEA::fill_between( stats_q05[,"di",imod] , stats_q95[,"di",imod] , time , col = rgb(1,0,0) , alpha = 0.5 )
		graphics::lines( time , coffee$stats[,"be","di",imod] , col = "brown" , lwd = 2 )
		graphics::abline( h = 0 )
	}
	grDevices::dev.off()
	if( verbose ) cat( "plot_stats_intensity (Done)      \n" )
}
##}}}

## plot_relative {{{

#' plot_relative
#'
#' Plot stats relative to a given time
#'
#' @importFrom latex2exp TeX
#'
#' @usage  plot_relative( coffee , time_rel , ofile , verbose )
#'
#' @param coffee [coffee list] coffee variable
#'
#' @param time_rel [vector] vector of time
#'
#' @param ofile [str] output file
#'
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_relative = function( coffee , time_rel , ofile , verbose = FALSE )
{
	if( verbose ) cat( "plot_relative      \r" )
	time       = coffee$time
	n_time     = coffee$n_time
	models     = coffee$models
	n_models   = coffee$n_models
	sample     = coffee$sample
	n_sample   = coffee$n_sample
	n_time_rel = length(time_rel)
	
	# Exhibits ofile names without extention (".pdf")
	ofile_pref = base::strsplit( ofile , split = ".pdf" )
	
	# stats_rel output
	stats_names = base::c( base::paste0( "rel_p_" , time_rel ) , base::paste0( "rel_i_" , time_rel ) )
	stats_rel  = array( NA , dim = base::c( n_time , n_sample + 1 , 2 * n_time_rel , n_models ) ,
 						dimnames = list( time = time , sample = dimnames(coffee$X)$sample , stats = stats_names , models = models )
						)
	
	# Loop on time_rel -- 1 output file for each time_rel
	for( i in 1:n_time_rel )
	{
		yrel = time_rel[i]
		# Calculate relative FAR / RR
		stats_rel[,,i,] = coffee$stats[,,"pall",] / ( NSSEA::ones(time) %o% coffee$stats[as.character(yrel),,"pall",] )
		# Calculate relative intensity
		stats_rel[,,n_time_rel+i,] = coffee$stats[,,"iall",] - ( NSSEA::ones(time) %o% coffee$stats[as.character(yrel),,"iall",] )
		
		# Ofile for each time_rel
		ofile_i = base::paste0( ofile_pref , "_" , as.character(yrel) , ".pdf" )
		grDevices::pdf(ofile_i)
		
		
		# Plot of frequency with loop on models
		lt = link_params()
		link  = lt$link
		ticks = lt$ticks
		
		for( imod in 1:n_models )
		{
			graphics::par( cex = 1.2 , font = 2 , mar = base::c(4,4,3,4) )
			graphics::plot( time, link$rr(stats_rel[,"be",i,imod]) , type = "l" , ylim = base::c(-1,1) , font = 2 , yaxt = "n" , col = "red" , ylab = "" , xlab = "" , xaxs = "i" , yaxs = "i" , mgp = base::c(2.6,.7,0) , main = base::toupper(models[imod]) )
			graphics::abline( h = 0 )
			graphics::lines( time , link$rr( stats_rel[,"be",i,imod]) , col = "red" , lwd = 3 )
			
			rr_q05 = base::apply( stats_rel[,-1,i,imod] , 1 , stats::quantile , p = .05 )
			rr_q95 = base::apply( stats_rel[,-1,i,imod] , 1 , stats::quantile , p = .95 )
			
			graphics::lines( time , link$rr(rr_q05) , col = "green" , lwd = 2 )
			graphics::lines( time , link$rr(rr_q95) , col = "green" , lwd = 2 )
			graphics::axis( 2 , at = link$rr(ticks$rr$values) , labels = ticks$rr$names  , las = 1 , font = 2 , mgp = base::c(2.6,.8,0) )
			graphics::axis( 4 , at = link$rr(ticks$rr$values) , labels = ticks$far$names , las = 1 , font = 2 , mgp = base::c(2.6,.8,0) )
			graphics::title( ylab = "RR"   , font.lab = 2 , cex = 1.2 , mgp = base::c(2.6,.8,0) )
			graphics::title( xlab = "Year" , font.lab = 2 , cex = 1.2 , mgp = base::c(2.6,.8,0) )
			xlim = graphics::par("usr")
			x_label_right = xlim[2] + (xlim[2]-xlim[1]) * 2.6 / graphics::par("mar")[4] * graphics::par("mai")[4] / graphics::par("pin")[1]
			graphics::text( x_label_right , 0 , "FAR" , xpd = TRUE , srt = 270 )
		}
		
		grDevices::dev.off()
	
	# To be added : plot of intensity
	}
	if( verbose ) cat( "plot_relative (Done)     \n" )
}
##}}}


## Multi model

## plot_multimodel {{{

#' plot_multimodel
#'
#' Plot multimodel
#'
#' @importFrom latex2exp TeX
#' @importFrom grDevices rgb
#' @importFrom RColorBrewer brewer.pal
#'
#' @usage  plot_multimodel( coffee , ofile , names , verbose )
#'
#' @param coffee [coffee list] coffee variable
#' @param ofile [str] output file
#' @param names [bool] ???
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_multimodel = function( coffee , ofile , names = FALSE , verbose = FALSE )
{
	if( verbose ) cat( "plot_multimodel      \r" )
	time     = coffee$time
	n_models = coffee$n_models - 1
	models   = coffee$models
	
	# Colors
	col1 = "gray40"
	col2 = rgb(1,0,0,.2)
	colX = if( names ) RColorBrewer::brewer.pal( n = n_models + 2 , name = "Set1" )[-base::c(1,6)] else col1
	
	## Link
	lt = link_params()
	link  = lt$link
	ticks = lt$ticks
	
	grDevices::pdf( ofile , width = 7 , height = 6 )
	graphics::par( cex = 1.1 , font = 2 , font.lab = 2 , font.axis = 2 ,lwd = 2 , mgp = base::c(3,.7,0) , mar = base::c(4,4,1,4) , las = 2 , tcl = -.4 , cex.lab = 1.2 )
	
	draw_y_axes = function( pref , lf = base::identity ) ##{{{
	{
		if( pref == "rr" )
		{ 
			graphics::axis( 2 , at = lf(ticks$rr$values) , labels = TeX(base::paste0("$\\mathbf{",ticks$rr$names ,"}$")) )
			graphics::axis( 4 , at = lf(ticks$rr$values) , labels = TeX(base::paste0("$\\mathbf{",ticks$far$names,"}$")) )
			graphics::title( ylab = "RR" )
			graphics::abline( h = 0 )
			xlim = graphics::par("usr")
			x_label_right = xlim[2] + (xlim[2]-xlim[1]) * ( graphics::par("mgp")[1]+.2) / graphics::par("mar")[4] * graphics::par("mai")[4] / graphics::par("pin")[1]
			graphics::text( x_label_right , 0 , "FAR" , xpd = TRUE , srt = 270 , cex = 1.2 )
		}
		else if( pref == "all" | pref == "nat" | pref == "ant")
		{
			graphics::axis(2)
			graphics::title( ylab= TeX( base::paste0("$\\mathbf{X_{",pref,"}}$ $$ $$ $\\mathbf{(K)}$") ) )
		}
		else if (pref=="p0" | pref=="p1")
		{
			print("Case to be created")
		}
		else if( pref == "di")
		{
			graphics::axis(2)
			graphics::title( ylab = TeX( base::paste0("$\\mathbf{\\delta i}$ $$ $$ $\\mathbf{(K)}$") ) )
		}
	}
	##}}}
	
	ylim_mm_ts = function( r , pref , names ) ##{{{
	{
		if( pref == "rr" )
		{
			ylim = base::c(-1,1)
		} else if( pref == "p0" | pref == "p1")
		{
			ylim = base::c(0,1)
		} else if( pref == "nat" & names )
		{
			ylim = base::c( r[1] , r[2] + .4 * (r[2]-r[1]) )
		} else
		{
			ylim = base::mean(r) + 1.05 * ( r - base::mean(r) )
		}
	}
	##}}}
	
	plot_MM_TS = function( Xsi , sample = 10 , names = FALSE , lf = identity , ... ) ##{{{
	{
		# Xsi		: input array to be plotted
		# sample	: number of realisations to plot (panels 3 and 4)
		# names	: whether names of individual models are to be written on the legend
		# lf		: link_function (in y)
		pref   = dimnames(Xsi)[3]
		Xs     = lf(Xsi)
		Xmat1  = Xs[,"be",,1:n_models]
		Xmat2  = Xs[,2:(sample+1),,"multi"]
		mm_q05 = base::apply(Xs[,-1,,"multi"] , 1 , stats::quantile , .05 )
		mm_q95 = base::apply(Xs[,-1,,"multi"] , 1 , stats::quantile , .95 )
		
		ylim = ylim_mm_ts( base::range(Xmat1,Xmat2,mm_q05,mm_q95) , pref , names )
		
		# Panel 1: Collection of models
		graphics::matplot( time , Xmat1 , ylim = ylim , type = "l" , lwd = 1.5 , lty = 1 , xlab = "Year" , ylab = "" , col = colX , yaxt = "n" , xaxs = "i" , yaxs = "i" , ... )
		draw_y_axes( pref , lf = lf )
		if( names ) 
			graphics::legend( "topleft" , models[1:n_models] , col = colX , lty = 1 , lwd = 1.5 , ncol = base::ceiling(n_models/5 ) )
		
		# Panel 2: Collection of models + multi-model range
		graphics::matplot( time , Xmat1 , ylim = ylim , type = "l" , lwd = 1.5 , lty = 1 , xlab = "Year" , ylab = "" , col = colX , yaxt = "n" , xaxs = "i" , yaxs = "i" , ... )
		draw_y_axes( pref , lf = lf )
		fill_between( mm_q05 , mm_q95 , time , col = col2 , alpha = 0.2 )
		graphics::lines( time , Xs[,"be",,n_models+1] , col = "brown" , lwd = 2 )
		if( names ) 
			graphics::legend( "topleft" , base::c(models[1:n_models],"mean") , col = colX , lty = 1 , lwd = 2 , ncol = base::ceiling(n_models/5) )
		
		# Panel 3: Resampled models
		graphics::matplot( time , Xmat2 , ylim = ylim , type = "l" , lwd = 1.5 , xlab = "Year" , ylab = "" , col = col1 , lty = 1 , yaxt = "n" , xaxs = "i" , yaxs = "i" )
		draw_y_axes( pref , lf = lf )
		
		# Panel 4: Resampled models + multi-model range
		graphics::matplot( time , Xmat2 , ylim = ylim , type = "l" , lwd = 1.5 , xlab = "Year" , ylab = "" , col = col1 , lty = 1 , yaxt = "n" , xaxs = "i" , yaxs = "i" )
		draw_y_axes( pref , lf = lf )
		fill_between( mm_q05 , mm_q95 , time , col = col2 , alpha = 0.2 )
		graphics::lines( time , Xs[,"be",,n_models+1] , col = "brown" , lwd = 2 )
		
		# Panel 5: multi-model range alone
		graphics::plot( time , Xs[,"be",,n_models+1] , type = "l" , ylim = ylim , col = "brown" , lwd = 2 , xlab = "Year" , ylab = "" , yaxt = "n" , xaxs = "i" , yaxs = "i" )
		draw_y_axes( pref , lf = lf )
		fill_between( mm_q05 , mm_q95 , time , col = rgb(1,0,0) , alpha = 0.5 )
		graphics::lines( time , Xs[,"be",,n_models+1] , col = "brown" , lwd = 2 )
	}
	##}}}
	
	Xant = coffee$X[,,"all",,drop=FALSE] - coffee$X[,,"nat",,drop=FALSE]
	dimnames(Xant)[3] = list( forcing = base::c("ant"))
	
	plot_MM_TS( coffee$X[,,"all",,drop=FALSE] , sample = 10 ) ## Xall
	plot_MM_TS( coffee$X[,,"nat",,drop=FALSE] , sample = 10 ) ## Xnat
	plot_MM_TS( Xant                   , sample = 10 ) ## Xant
	
	# RR / FAR
	plot_MM_TS( coffee$stats[,,"rr",,drop=FALSE] , sample = 10 , lf = link$rr )
	plot_MM_TS( coffee$stats[,,"di",,drop=FALSE] , sample = 10 )
	grDevices::dev.off()
	if( verbose ) cat( "plot_multimodel (Done)     \n" )
}
##}}}


## Constraints plot

## plot_constraints_CX {{{

#' plot_constraints_CX
#'
#' Plot results of constraints CX
#'
#' @importFrom latex2exp TeX
#' @importFrom grDevices rgb
#'
#' @usage  plot_constraints_CX( coffee , coffeeCX , Xo , ref_plot , ofile , sample , verbose )
#'
#' @param coffee [coffee list] coffee variable
#'
#' @param coffeeCX [coffee list] coffee variable with CX constraints
#'
#' @param Xo [array] array of observed covariable
#'
#' @param ref_plot [vector] time reference to plot
#'
#' @param ofile [str] output file
#'
#' @param sample [integer] not used
#'
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_constraints_CX = function( coffee , coffeeCX , Xo , ref_plot , ofile , sample = 0 , verbose = FALSE )
{
	if( verbose ) cat( "plot_constraints_CX     \r" )
	time    = coffee$time
	n_time  = coffee$n_time
	time_Xo = as.numeric(names(Xo))
	
	X    = coffee$X[,,"all","multi"]
	X_CX = coffeeCX$X[,,"all","multi"] 
	
	if( !is.null(ref_plot) )
	{
		# Check that ref_plot is included in year and year_obs_x
		if( base::all(ref_plot %in% time_Xo) & base::all(ref_plot %in% time) )
		{
			X    = X - NSSEA::ones(X[,1]) %o% base::apply( X[time %in% ref_plot,] , 2 , base::mean )
			X_CX = X_CX - NSSEA::ones(X_CX[,1]) %o% base::apply(X_CX[time %in% ref_plot,] , 2 , base::mean )
			Xo   = Xo - base::mean( Xo[time_Xo %in% ref_plot] )
#			Xo   = Xo - base::mean( Xo[ref_plot %in% time_Xo] )
		}
	}
	
	x_q95  = base::apply( X[,-1]    , 1 , stats::quantile , .95 )
	x_q05  = base::apply( X[,-1]    , 1 , stats::quantile , .05 )
	xc_q95 = base::apply( X_CX[,-1] , 1 , stats::quantile , .95 )
	xc_q05 = base::apply( X_CX[,-1] , 1 , stats::quantile , .05 )
	
	grDevices::pdf( ofile )
	graphics::par( font.lab = 2 , font.axis = 2 , cex.lab = 1.2 , mar = base::c(4,4,1,1) , mgp = base::c(2.5,.7,0) )
	graphics::plot( time_Xo , Xo , xlim = base::range(time) , ylim = base::range(Xo,x_q05,x_q95,xc_q05,xc_q95) , type = "p" , pch = 16 , cex = .8 , xlab = TeX("\\textbf{Year}") , ylab = TeX("\\textbf{Temperature $$ $$ ($^o C$)}") )
	fill_between( x_q05  , x_q95  , time , col = rgb(1,0,0) , alpha = 0.2 )
	fill_between( xc_q05 , xc_q95 , time , col = rgb(1,0,0) , alpha = 0.5 )
	
	colb = grDevices::col2rgb( "brown" ) / 255
	graphics::lines( time , X[,1]      , lwd = 1.5 , col = rgb(colb[1],colb[2],colb[3],alpha=.5) )
	graphics::lines( time , X_CX[,1] , lwd = 2   , col = rgb(colb[1],colb[2],colb[3],alpha=1)  )
	
#	if( sample > 0 )
#	{
#		X_sample      = matrix( 0 , n_time , sample )
#		X_Cons_sample = matrix( 0 , n_time , sample )
#		for (i in 1:sample)
#		{ 
#			graphics::lines( time , X_sample[,i] )
#			X_Cons_sample[,i] = A_full %*% ( mu_post + Sigma_post_sqrt %*% stats::rnorm(mu_post) )
#			graphics::lines( time , X_Cons_sample[,i] , col = "red" )
#		}
#	}
	grDevices::dev.off()
	if( verbose ) cat( "plot_constraints_CX (Done)     \n" )
}
##}}}

## plot_stats_time {{{

#' plot_stats_time
#'
#' Plot statistics for a given time
#'
#' @importFrom latex2exp TeX
#' @importFrom grDevices rgb
#'
#' @usage  plot_stats_time( ns_stats , time_res , ofile , verbose )
#'
#' @param ns_stats [stats] coffee$stats
#'
#' @param time_res [time]
#'
#' @param ofile [str] output file
#'
#' @param verbose [bool] print or not state of execution
#'
#' @return NULL
#'
#' @examples
#' ##
#' 
#' @export
plot_stats_time = function( ns_stats , time_res , ofile = NULL , verbose = FALSE )
{
	if( verbose ) cat( "plot_stats_time      \r" )
	# Reading arguments
	models   = dimnames(ns_stats)$models
	n_models = length(models)
	
	
	## Quantiles
	d_q05 = base::apply( ns_stats[as.character(time_res),-1,,,drop = FALSE] , base::c(3,4) , stats::quantile , .05 )
	d_q95 = base::apply( ns_stats[as.character(time_res),-1,,,drop = FALSE] , base::c(3,4) , stats::quantile , .95 )
	
	# pdf ofile
	if( is.null(ofile) )
		ofile = base::paste0( "diag_" , as.character(time_res) )
	
	## Plot params
	if( n_models < 8 )
	{
		height  = 2 + n_models / 3
		cex_mod = 1.2
		wd      = .25	# Half-width of confidance ranges (ie rectangles)
		ylim    = base::c( 0 , n_models )
	}
	else if( n_models < 16 )
	{
		height  = 2 + n_models / 5
		cex_mod = 1
		wd      = .3	
		ylim    = base::c( .5 , n_models - .5 )
	}
	else
	{
		height  = 2 + n_models / 7
		cex_mod = .8
		wd      = .4	
		ylim    = base::c( .5 , n_models - .5 )
	} 
	
	## Link function
	lt = link_params()
	link  = lt$link
	ticks = lt$ticks
	
	
	# Preliminary graphical parameters
	grDevices::pdf( ofile , width = 6 , height = height )
	graphics::par( cex = 1.2 , font = 2 , font.axis = 2 , font.lab = 2 , cex.axis = .75 , mgp = base::c(2.5,.5,0) , mar = base::c(3,6,3,1) , lwd = 2 , las = 1 , tcl = -.4 )
	
	
	# p0
	graphics::plot( 0 , n_models / 2 , xlim = base::c(0,1) , ylim = ylim , xlab = "" , ylab = "" , xaxt = "n" , yaxt = "n" , xaxs = "i" , lwd = 0 , pch = "." )
	graphics::axis( 1 , at = link$p(ticks$p1$values) , labels = TeX(base::paste0("\\textbf{",ticks$p1$names,"}")) )
	graphics::mtext( TeX("\\textbf{$p_0$}") , side = 1 , line = 1.7 , cex = 1.2 )
	for( i in 1:n_models )
	{
		graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
		v_be = ns_stats[as.character(time_res),"be","pnat",i]
		graphics::rect( link$p(d_q05["pnat",i]) , n_models - i + .5 - wd , link$p(d_q95["pnat",i]) , n_models - i + .5 + wd , col = "red" , border = NA )
		graphics::lines( link$p(v_be) * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
	}
	
	# p1
	graphics::plot( 0 , n_models / 2 , xlim = base::c(0,1) , ylim = ylim , xlab = "" , ylab = "" , xaxt = "n" , yaxt = "n" , xaxs = "i" , lwd = 0 , pch = "." )
	graphics::axis( 1 , at = link$p(ticks$p1$values) , labels = TeX(base::paste0("\\textbf{",ticks$p1$names,"}")) )
	graphics::mtext( TeX("\\textbf{$p_1$}") , side = 1 , line = 1.7 , cex = 1.2 )
	for( i in 1:n_models )
	{
		graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
		v_be = ns_stats[as.character(time_res),"be","pall",i]
		graphics::rect( link$p(d_q05["pall",i]) , n_models - i + .5 - wd , link$p(d_q95["pall",i]) , n_models - i + .5 + wd , col = "red" , border = NA )
		graphics::lines( link$p(v_be) * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
	}
	
	# FAR/RR
	graphics::plot( -1 , n_models / 2 , xlim = base::c(-1,1) , ylim = ylim , xlab = "" , ylab = "" , xaxt = "n" , yaxt = "n" , xaxs = "i" , lwd = 0 , pch = "." )
	graphics::axis( 1 , at = link$rr(ticks$rr$values) , labels = TeX(base::paste0("$\\mathbf{",ticks$rr$names,"}$")) , mgp = base::c(2.5,.4,0) )
	graphics::axis( 3 , at = link$rr(ticks$rr$values) , labels = TeX(base::paste0("$\\mathbf{",ticks$far$names,"}$")) )
	graphics::mtext( "RR"  , side = 1 , line = 1.7 , cex = 1.2 )
	graphics::mtext( "FAR" , side = 3 , line = 1.7 , cex = 1.2 )
	for( i in 1:n_models )
	{
		graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
		v_be = ns_stats[as.character(time_res),"be","rr",i]
		graphics::rect( link$rr(d_q05["rr",i]) , n_models - i + .5 - wd , link$rr(d_q95["rr",i]) , n_models - i + .5 + wd , col = "red" , border = NA )
		graphics::lines( link$rr(v_be) * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
	}
	
	# Relative FAR/RR
	if( dim(ns_stats)[3] > 6 )
	{
		graphics::plot( -1 , n_models / 2 , xlim = base::c(-1,1) , ylim = ylim , xlab = "" , ylab = "" , xaxt = "n" , yaxt = "n" , xaxs = "i" , lwd = 0 , pch = "." )
		graphics::axis( 1 , at = link$rr(ticks$rr$values) , labels = TeX(base::paste0("$\\mathbf{",ticks$rr$names,"}$")) , mgp = base::c(2.5,.4,0) )
		graphics::axis( 3 , at = link$rr(ticks$rr$values) , labels = TeX(base::paste0("$\\mathbf{",ticks$far$names,"}$")) )
		graphics::mtext( TeX("\\textbf{RR$_{rel}$}")  , side = 1 , line = 1.7 , cex = 1.2 )
		graphics::mtext( TeX("\\textbf{FAR$_{rel}$}") , side = 3 , line = 1.7 , cex = 1.2 )
		for( i in 1:n_models )
		{
			graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
			v_be = ns_stats[as.character(time_res),"be",7,i]
			graphics::rect( link$rr(d_q05[7,i]) , n_models - i + .5 - wd , link$rr(d_q95[7,i]) , n_models - i + .5 + wd , col = "red" , border = NA )
			graphics::lines( link$rr(v_be) * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
		}
	}
	
	
	# I0
	xll = base::range( d_q05[c("inat","iall"),] , d_q95[c("inat","iall"),] )
	graphics::plot( 0 , n_models / 2 , xlim = xll , ylim = ylim , xlab = "" , ylab = "" , yaxt = "n" , lwd = 0 , pch = "." )
	graphics::mtext( TeX("\\textbf{$i_0$}") , side = 1 , line = 1.7 , cex = 1.2 )
	for( i in 1:n_models )
	{
		graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
		v_be = ns_stats[as.character(time_res),"be","inat",i]
		graphics::rect( d_q05["inat",i] , n_models - i + .5 - wd , d_q95["inat",i] , n_models - i + .5 + wd , col = "red" , border = NA )
		graphics::lines( v_be * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
	}
	
	# I1
	graphics::plot( 0 , n_models / 2 , xlim = xll , ylim = ylim , xlab = "" , ylab = "" , yaxt = "n" , lwd = 0 , pch = "." )
	graphics::mtext( TeX("\\textbf{$i_1$}") , side = 1 , line = 1.7 , cex = 1.2 )
	for( i in 1:n_models )
	{
		graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
		v_be = ns_stats[as.character(time_res),"be","iall",i]
		graphics::rect( d_q05["iall",i] , n_models - i + .5 - wd , d_q95["iall",i] , n_models - i + .5 + wd , col = "red" , border = NA )
		graphics::lines( v_be * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
	}
	
	# delta I
	xll = base::range( 0 , d_q05["di",] , d_q95["di",] )
	graphics::plot( -1 , n_models / 2 , xlim = xll , ylim = ylim , xlab = "" , ylab = "" , xaxt = "n" , yaxt = "n" , lwd = 0 , pch = "." )
	graphics::axis( 1 , mgp = base::c(2.5,.4,0) )
	graphics::mtext( TeX("\\textbf{$\\delta i$}") , side = 1 , line = 1.7 , cex = 1.2 )
	for( i in 1:n_models ) 
	{
		graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
		v_be = ns_stats[as.character(time_res),"be","di",i]
		graphics::rect( d_q05["di",i] , n_models - i + .5 - wd , d_q95["di",i] , n_models - i + .5 + wd , col = "red" , border = NA )
		graphics::lines( v_be * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
	}
	
	# Relative delta i
	if( dim(ns_stats)[3] > 6 )
	{
		xll = base::range( 0 , d_q05[8,] , d_q95[8,] )
		graphics::plot( -1 , n_models / 2 , xlim = xll , ylim = ylim , xlab = "" , ylab = "" , xaxt = "n" , yaxt = "n" , lwd = 0 , pch = "." )
		graphics::axis( 1 , mgp = base::c(2.5,.4,0) )
		graphics::mtext( TeX("\\textbf{$\\delta i_{rel}$}") , side = 1 , line = 1.7 , cex = 1.2 )
		for( i in 1:n_models )
		{
			graphics::mtext( base::toupper(models[i]) , side = 2 , line = .8 , at = n_models + .5 - i , cex = cex_mod )
			v_be = ns_stats[as.character(time_res),"be",8,i]
			graphics::rect( d_q05[8,i] , n_models - i + .5 - wd , d_q95[8,i] , n_models - i + .5 + wd , col = "red" , border = NA )
			graphics::lines( v_be * base::c(1,1) , n_models - i + .5 + base::c(-wd,wd) , lwd = 2 )
		}
	}
	
	# Other diags (intensity)
	# To Be Done
	
	grDevices::dev.off()
	if( verbose ) cat( "plot_stats_time (Done)     \n" )
}

##}}}

