# -*- coding: utf-8 -*-

#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################


__version__ = "0.2.4"

#############
## Imports ##
#############

## Tools
from NSSEA.__tools import ProgressBar


## Variables
from NSSEA.__variables import Event
from NSSEA.__variables import Coffee
from NSSEA.__variables import CXParams
from NSSEA.__variables import coffee2netcdf
from NSSEA.__variables import netcdf2coffee


## Covariate split
from NSSEA.__covariates import EBMModel
from NSSEA.__covariates import XSplitted
from NSSEA.__covariates import gam_decomposition
#from NSSEA.__covariates import gam_decomposition_old
#from NSSEA.__covariates import gam_decomposition_old_old
#from NSSEA.__covariates import gam_decomposition_old_old_old

## Stats
from NSSEA.__nsfit    import nslaw_fit
from NSSEA.__mm_infer import infer_multi_model
from NSSEA.__nsstats  import extremes_stats
from NSSEA.__nsstats  import RR_correction
from NSSEA.__nsstats  import stats_relative_event

## Constraints
from NSSEA.__constraints import constraints_CX
from NSSEA.__constraints import constraints_C0

