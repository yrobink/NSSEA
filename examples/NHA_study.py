# -*- coding: utf-8 -*-

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2020 & Octave Tessiot, 2023                           ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
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
## Copyright Yoann Robin, 2020                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## Ce logiciel est un programme informatique faisant partie de la librairie     ##
## NSSEA (Non-Stationary Statistics for Extreme Attribution). Cette librairie   ##
## permet d'estimer la probabilite d'un evenement (extreme) dans le monde       ##
## factuel / contre factuel (sans forcage anthropogenique) et de l'attribuer au ##
## changement climatique.                                                       ##
##                                                                              ##
## Ce logiciel est regi par la licence CeCILL-C soumise au droit francais et    ##
## respectant les principes de diffusion des logiciels libres. Vous pouvez      ##
## utiliser, modifier et/ou redistribuer ce programme sous les conditions       ##
## de la licence CeCILL-C telle que diffusee par le CEA, le CNRS et l'INRIA     ##
## sur le site "http://www.cecill.info".                                        ##
##                                                                              ##
## En contrepartie de l'accessibilite au code source et des droits de copie,    ##
## de modification et de redistribution accordes par cette licence, il n'est    ##
## offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,    ##
## seule une responsabilite restreinte pese sur l'auteur du programme, le       ##
## titulaire des droits patrimoniaux et les concedants successifs.              ##
##                                                                              ##
## A cet egard  l'attention de l'utilisateur est attiree sur les risques        ##
## associes au chargement,  a  l'utilisation, a la modification et/ou au        ##
## developpement et a la reproduction du logiciel par l'utilisateur etant       ##
## donne sa specificite de logiciel libre, qui peut le rendre complexe a        ##
## manipuler et qui le reserve donc a des developpeurs et des professionnels    ##
## avertis possedant  des  connaissances  informatiques approfondies.  Les      ##
## utilisateurs sont donc invites a charger  et  tester  l'adequation  du       ##
## logiciel a leurs besoins dans des conditions permettant d'assurer la         ##
## securite de leurs systemes et ou de leurs donnees et, plus generalement,     ##
## a l'utiliser et l'exploiter dans les memes conditions de securite.           ##
##                                                                              ##
## Le fait que vous puissiez acceder a cet en-tete signifie que vous avez       ##
## pris connaissance de la licence CeCILL-C, et que vous en avez accepte les    ##
## termes.                                                                      ##
##                                                                              ##
##################################################################################
##################################################################################

###############
## Libraries ##
###############

import sys,os
import pickle as pk
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import SDFC.link as sdl
import NSSEA as ns
import NSSEA.plot as nsp
import NSSEA.models as nsm

models_to_exclude = ['CESM2']
###############
## Fonctions ##
###############

def correct_miss( X , lo =  100 , up = 350 ):
#    return X
    mod = str(X.columns[0])
    bad = np.logical_or( X < lo , X > up )
    bad = np.logical_or( bad , np.isnan(X) )
    bad = np.logical_or( bad , np.logical_not(np.isfinite(X)) )
    if np.any(bad):
        idx,_ = np.where(bad)
        idx_co = np.copy(idx)
        for i in range(idx.size):
            j = 0
            while idx[i] + j in idx:
                j += 1
                if idx[i]+j == X.size :
                    j = -1
                    k = -1
                if idx[i]+j == -1:
                    raise('all values seem to be missed in X',X)
            idx_co[i] += j
        X.iloc[idx] = X.iloc[idx_co].values
    return X


def adapt_file_structure_X(df):
    ## Change variable name
    df = df.rename({'Tmm'+project_name:'tas'})
    
    ## Remove 2100
    df = df.sel(time=(df.time.dt.year!=2100))
    
    return df


def adapt_file_structure_Y(df):
    ## Change variable name
    df = df.rename({'Tnn'+project_name:'tas'})
    
    ## Remove one year to the time serie because the definition of winter is not the same
    for it in range (0, df.time.size):
        old_date           = df.time[it].values
        date_split         = np.datetime_as_string(old_date).split('-')
        date_split[0]      = str(int(date_split[0])-1)  # modify the year
        new_date           = np.datetime64('-'.join(date_split))
        df.time.values[it] = new_date
    
    ## If there is a time_bnds axis, remove one year too
    if 'time_bnds' in df.variables.keys():
        for it in range (0, df.time.size):
            for ibnds in range (0, df.bnds.size):
                old_date                      = df.time_bnds[it,ibnds].values
                date_split                    = np.datetime_as_string(old_date).split('-')
                date_split[0]                 = str(int(date_split[0])-1)  # modify the year
                new_date                      = np.datetime64('-'.join(date_split))
                df.time_bnds.values[it,ibnds] = new_date
    
    ## select only from 1850 to 2099
    df = df.sel(time=slice('1850','2099'))
    
    ## Remove realisation dimension, and put all in the time dimension with repeted time values
    df             = df.stack(time_realisation=('time','realisation'))
    df['new_time'] = df.time
    df             = df.swap_dims({'time_realisation':'new_time'})
    df             = df.drop(['time_realisation'])
    #df             = df.rename({'time':'old_time'})
    #df             = df.rename({'new_time':'time'})
    df.swap_dims({'new_time':'time'}).time
    
    return df


def load_data(pathInp, add_1year_to_Y=False):
    print("pathInp = ", pathInp)
    ## List of models X
    modelsX = [ f.split("/")[-1][:-3].split("_")[2] for f in os.listdir(os.path.join(pathInp,"X")) ]
    modelsX.sort()
    
    ## List of models Y
    modelsY = [ f.split("/")[-1][:-3].split("_")[2] for f in os.listdir(os.path.join(pathInp,"Y")) ]
    modelsY.sort()
    
    ## Merge the two lists to keep only common models
    modelsX.sort()
    modelsY.sort()
    models = list( (set(modelsX) & set(modelsY)) - set(models_to_exclude) )
    models.sort()
    print('Model list : ', models)
    
    ## Load X and Y
    lX = []
    lY = []
    for m in models:
        
        ## Load X
        dfX  = xr.open_dataset( os.path.join( pathInp , "X/Tmm{pn}_{s}_{m}_mean.nc".format(pn=project_name, s=scenario, m=m) ) )
        dfX  = adapt_file_structure_X(dfX)
        time = dfX.time["time.year"].values
        X    = pd.DataFrame( dfX.tas.values.ravel() , columns = [m] , index = time )
        lX.append( correct_miss(X) )
        
        ## Load Y
        dfY  = xr.open_dataset( os.path.join( pathInp , "Y/Tnn{pn}_{s}_{m}_all.nc".format(pn=project_name, s=scenario, m=m) ) )
        dfY  = adapt_file_structure_Y(dfY)
        time = dfY.time["time.year"].values
        if add_1year_to_Y:
            time = time+1
        Y    = pd.DataFrame( dfY.tas.values.ravel() , columns = [m] , index = time )
        lY.append( correct_miss(Y) )
    
    ## Load Xo
    path_Xo = os.path.join( pathInp , "Xo.nc" )
    dXo    = xr.open_dataset( os.path.join( pathInp , "Xo.nc" ) )
    timeXo = dXo.time["time.year"].values
    Xo  = pd.DataFrame( dXo.tasmin.values.squeeze() , columns = ["Xo"] , index = timeXo )
    
    ## Load Yo
    path_Yo = os.path.join( pathInp , "Yo.nc" )
    dYo    = xr.open_dataset( path_Yo )
    timeYo = dYo.time["time.year"].values
    if add_1year_to_Y:
        timeYo = timeYo+1
    Yo = pd.DataFrame( dYo.tasmin.values.squeeze() , columns = ["Yo"] , index = timeYo )
    
    return models,lX,lY,Xo,Yo

class NumpyLog: 
    def __init__(self):
        self._msg = []
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return "".join(self._msg)
    
    def write( self , msg ):
        self._msg.append(msg)



##########
## main ##
##########

if __name__ == "__main__":
    
    ## Entries
    
    # scenario is the name of ssp (only ssp245 is available in this example)
    scenario = 'historical-ssp245'
    
    # project_name is the name of area of interest ( Brazil , China , Europe or Texas )
    project_name = 'Europe'
    
    # event_year is the year when the event appeared. The convention for winter North Hemisphere is winter end of y-1 / early y. For event taking place in North Hemisphere, Yo/models time has been shifted by -180days, then we recommand you to add one year to Yo/models to respect the convention of winter y-1/y of the event year : add_1year_to_Y = True.
    if project_name=='Brazil':
        envent_year  = 2022
        add_1year_to_Y = False
    elif project_name=='China':
        envent_year  = 2015+1
        add_1year_to_Y = True
    elif project_name=='Europe':
        envent_year  = 2011+1
        add_1year_to_Y = True
    elif project_name=='Texas':
        envent_year  = 2020+1
        add_1year_to_Y = True
    
    # If test, reduce the quality of the analysis to make it faster.
    is_test = True
    
    # Set seed
    np.random.seed(42)
    
    ## Set a log class, the GEV produces sometimes overflow
    ##=====================================================
    nplog = NumpyLog()
    np.seterrcall(nplog)
    np.seterr( all = "log" , invalid = "log" )
    warnings.simplefilter("ignore")
    
    ## Path
    ##=====
    scriptpath = os.path.dirname(os.path.abspath(__file__))   # give the path of the folder where the script is -> root of data
    basepath   = os.path.join( scriptpath , 'NHA_study_data' , project_name )
    pathInp    = os.path.join( basepath , "input"  )
    pathOut    = os.path.join( basepath , "output/local_experiment" )
    
    ## Download data from zenodo
    ##==========================
    data_url = 'https://zenodo.org/record/8435974/files/NHA_study_data.zip?download=1'
    os.system(f'wget -O {scriptpath}/NHA_study_data.zip {data_url}')
    os.system(f'unzip -d {scriptpath}/NHA_study_data {scriptpath}/NHA_study_data.zip')
    
    ## Verify input and output folder
    ##===============================
    assert os.path.exists(pathInp), 'The input directory is not fund : '+pathInp
    print('pathOut =',pathOut)
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    
    
    ## Some global parameters
    ##=======================
    time_period    = np.arange( 1850 , 2101 , 1 , dtype = np.int )
    time_reference = np.arange( 1961 , 1991 , 1 , dtype = np.int )
    min_rate_accept  = 0.05
    n_mcmc_drawn_min = 500  if is_test else  5000
    n_mcmc_drawn_max = 1000 if is_test else 10000
    n_sample         = 50   if is_test else  1000
    ci               = 0.1
    verbose  = "--not-verbose" not in sys.argv
    ns_law   = nsm.GEVMin()
    kwargs_bayesian_constrain = {'min_rate_accept'  : min_rate_accept }
    event       = ns.Event( name      = "CO"+str(envent_year) ,
                            time      = envent_year ,
                            reference = time_reference ,
                            side      = "lower",
                            type_     = "value" ,
                            variable  = "tasmin" ,
                            unit      = "K" )
    
    ## Load models and observations
    ##=============================
    models,lX,lY,Xo,Yo = load_data(pathInp, add_1year_to_Y=add_1year_to_Y)
    
    ## Anomaly from observations
    ##==========================
    Yo -= Yo.loc[event.reference].mean()
    event.value = float(Yo.loc[event.time])
    Xo -= Xo.loc[event.reference].mean()
    print('Event extrem anomaly /Y :',event.value,'K')
    
    
    ## Models in anomaly
    ##==================
    for X in lX:
        X -= X.loc[event.reference].mean()
    for Y in lY:
        Y -= Y.loc[event.reference].mean()
    
    
    ## Define clim variable from input
    ##================================
    print('## Define clim variable from input')
    clim = ns.Climatology( event , time_period , models , n_sample , ns_law )
    
    
    ## Decomposition of covariates
    ##============================
    print('## Decomposition of covariates')
    Xebm   = ns.EBM().draw_sample( clim.time , n_sample + 1 , fix_first = 0 )
    clim   = ns.covariates_FC_GAM( clim , lX , Xebm , verbose = verbose )
    
    
    ## Fit distribution
    ##=================
    print('## Fit distribution')
    clim = ns.nslaw_fit( lY , clim , verbose = verbose )
    
    
    ## Multi-model
    ##============
    print('## Multi-model')
    clim = ns.infer_multi_model( clim , verbose = verbose )
    climMM = clim.copy()
    climMM.keep_models( "Multi_Synthesis" )
    
    
    ## Apply constraints
    ##==================
    print('## Apply constraints')
    climCX     = ns.constrain_covariate( climMM , Xo , time_reference , verbose = verbose )

    ns_param_prior       = climCX.law_coef.sel(model='Multi_Synthesis')
    ns_param_prior_std   = ns_param_prior.std('sample')
    
    kwargs_bayesian_constrain['transition'] = lambda x : x + np.random.normal( size=ns_law.n_ns_params , scale=ns_param_prior_std )
    
    climCXCB   = ns.constrain_law( climCX , Yo ,
                                   n_mcmc_drawn_min=n_mcmc_drawn_min , n_mcmc_drawn_max=n_mcmc_drawn_max ,
                                   verbose=verbose , keep='all' , **kwargs_bayesian_constrain)
    climC0     = ns.constraint_C0( climMM , Yo , verbose = verbose )
    climCXC0   = ns.constraint_C0( climCX , Yo , verbose = verbose )
    
    
    ## Compute stats
    ##==============
    print('## Compute stats')
    clim       = ns.statistics_attribution( clim     , verbose = verbose )
    climCX     = ns.statistics_attribution( climCX   , verbose = verbose )
    climCXCB   = ns.statistics_attribution( climCXCB , verbose = verbose )
    climCXC0   = ns.statistics_attribution( climCXC0 , verbose = verbose )
    clim       = ns.add_FAR( clim     , verbose = verbose )
    climCX     = ns.add_FAR( climCX   , verbose = verbose )
    climCXCB   = ns.add_FAR( climCXCB , verbose = verbose )
    climCXC0   = ns.add_FAR( climCXC0 , verbose = verbose )
    clim       = ns.add_return_time( clim     , verbose = verbose )
    climCX     = ns.add_return_time( climCX   , verbose = verbose )
    climCXCB   = ns.add_return_time( climCXCB , verbose = verbose )
    climCXC0   = ns.add_return_time(climCXC0 , verbose = verbose )
    
    params     = ns.build_params_along_time( clim     , verbose = verbose )
    paramsCX   = ns.build_params_along_time( climCX   , verbose = verbose )
    paramsCXCB = ns.build_params_along_time( climCXCB , verbose = verbose )
    paramsCXC0 = ns.build_params_along_time( climCXC0 , verbose = verbose )
    
    ## Compute never happend again
    ##=================
    print('## Compute never happend again')
    def Compute_happen_again(happend_y):
        happend_y_2100 = happend_y.copy(deep=True)*0
        for y in happend_y_2100.time[::-1].values:
            if y==happend_y_2100.time.values[-1]:
                happend_y_2100.loc[y] = happend_y.loc[y]
            else:
                happend_y_2100.loc[y] = happend_y_2100.loc[y+1] + happend_y.loc[y] - happend_y_2100.loc[y+1]*happend_y.loc[y]
        never_happend_y_2100 = 1-happend_y_2100
        happend_y_2100['stats'] = [f'happend_again_yEnd_{world}' for world in happend_y.stats.values]
        never_happend_y_2100['stats'] = [f'never_happend_again_yEnd_{world}' for world in happend_y.stats.values]
        return happend_y_2100,never_happend_y_2100
    
    happend_y_2100     ,never_happend_y_2100      = Compute_happen_again(clim.statistics.sel(stats=['pC', 'pF']))
    happend_y_2100_CX  ,never_happend_y_2100_CX   = Compute_happen_again(climCX.statistics.sel(stats=['pC', 'pF']))
    happend_y_2100_CXCB,never_happend_y_2100_CXCB = Compute_happen_again(climCXCB.statistics.sel(stats=['pC', 'pF']))
    happend_y_2100_CXC0,never_happend_y_2100_CXC0 = Compute_happen_again(climCXC0.statistics.sel(stats=['pC', 'pF']))
    clim.data     = xr.merge([clim.data    , xr.Dataset({"statistics":happend_y_2100     }), xr.Dataset({"statistics":never_happend_y_2100     })])
    climCX.data   = xr.merge([climCX.data  , xr.Dataset({"statistics":happend_y_2100_CX  }), xr.Dataset({"statistics":never_happend_y_2100_CX  })])
    climCXCB.data = xr.merge([climCXCB.data, xr.Dataset({"statistics":happend_y_2100_CXCB}), xr.Dataset({"statistics":never_happend_y_2100_CXCB})])
    climCXC0.data = xr.merge([climCXC0.data, xr.Dataset({"statistics":happend_y_2100_CXC0}), xr.Dataset({"statistics":never_happend_y_2100_CXC0})])
    
    ## Save in netcdf
    ##===============
    print('## Save in netcdf')
    for c,s in zip([clim,climCX,climCXC0,climCXCB],["","CX","CXC0","CXCB"]):
        c.to_netcdf( os.path.join( pathOut , "{}_clim{}.nc".format(event.name,s) ) )
    for p,s in zip([params,paramsCX,paramsCXC0,paramsCXCB],["","CX","CXC0","CXCB"]):
        p.to_dataset( name = "params{}".format(s) ).to_netcdf( os.path.join( pathOut , "{}_params{}.nc".format(event.name,s) ) )
    
    ## Plot
    ##=====
    print('## Plot')
    pltkwargs = { "verbose" : verbose , "ci" : ci }
    nsp.GAM_decomposition( clim , lX , os.path.join( pathOut , "GAM_decomposition.pdf" ) , **pltkwargs )
    nsp.constraint_covariate( clim , climCXCB , Xo , os.path.join( pathOut , "constraint_covariate.pdf" )  , **pltkwargs )
    nsp.summary( clim     , pathOut , t1 = 2040 , params = params     , **pltkwargs )
    nsp.summary( climCX   , pathOut , t1 = 2040 , params = paramsCX   , suffix = "CX"   , **pltkwargs )
    nsp.summary( climCXC0 , pathOut , t1 = 2040 , params = paramsCXC0 , suffix = "CXC0" , **pltkwargs )
    nsp.summary( climCXCB , pathOut , t1 = 2040 , params = paramsCXCB , suffix = "CXCB" , **pltkwargs )
    nsp.constraint_law( climCX , climCXCB , ofile = os.path.join( pathOut , "constraint_law.pdf" ) , **pltkwargs )
    nsp.statistics_time( [clim,climCX,climCXCB] , os.path.join( pathOut , "Statistics_time.pdf" ) , labels = clim.model.tolist() + ["Multi_CX","Multi_CXCB"] , colors = ["red","blue","green"] , **pltkwargs )
    
    
    ## Other plots
    ##============
    print('Plot never_happend_again')
    import matplotlib.pyplot as plt
    color={'never_happend_again_yEnd_pF':'red', 'never_happend_again_yEnd_pC':'blue'}
    plt.figure(figsize=(20,12))
    for i,(c,s) in enumerate(zip([clim,climCX,climCXC0,climCXCB],["/","CX","CXC0","CXCB"])):
        plt.subplot(2,2,i+1)
        for world in ['never_happend_again_yEnd_pF', 'never_happend_again_yEnd_pC']:
            plt.fill_between( x=c.statistics.time.values, y1=c.statistics.sel(stats=world, model='Multi_Synthesis').quantile(ci/2., 'sample'), y2=c.statistics.sel(stats=world, model='Multi_Synthesis').quantile(1-ci/2., 'sample'), alpha=0.5, color=color[world], linewidth=0.0)
        for world in ['never_happend_again_yEnd_pF', 'never_happend_again_yEnd_pC']:
            plt.plot(c.statistics.time.values, c.statistics.sel(stats=world, model='Multi_Synthesis', sample='BE'), color=color[world], label=world)
        plt.legend()
        plt.title(s)
        plt.xlabel('time')
        plt.ylabel('probability')
    plt.suptitle(f'Never happend again until 2100 - BE and {ci/2}-{1-ci/2} interval')
    plt.savefig(f'{pathOut}/never_happend_again.pdf')
    plt.close('all')
    


    print("Well done")


