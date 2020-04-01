# -*- coding: utf-8 -*-

#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################


__version__ = "0.3.0a10"

#############
## Imports ##
#############

## Tools
from .__tools import ProgressBar


## Variables
from .__variables import Event
from .__variables import Climatology
from .__variables import to_netcdf
from .__variables import from_netcdf


## Covariate split
from .__covariates import EBM
from .__covariates import gam_decomposition
#from NSSEA.__covariates import XSplitted
#from NSSEA.__covariates import gam_decomposition_old
#from NSSEA.__covariates import gam_decomposition_old_old
#from NSSEA.__covariates import gam_decomposition_old_old_old

## Stats
from .__nsfit       import nslaw_fit
from .__multi_model import infer_multi_model
from .__multi_model import MultiModelParams
from .__nsstats     import extremes_stats
from .__nsstats     import stats_relative_event
from .__nsstats     import build_params_along_time
#from NSSEA.__nsstats  import RR_correction

## Constraints
from .__constraints import constraints_CX
from .__constraints import constraints_C0
from .__constraints import constraints_bayesian
