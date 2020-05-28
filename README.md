# NSSEA (Non-Stationary Statistics for Extreme Attribution)

Warning : the R version does not work

## Python instruction

Requires:
- python3
- [SDFC](https://github.com/yrobink/SDFC)
- numpy(>=1.15.0)
- scipy(>=0.19)
- xarray
- pandas
- matplotlib
- pygam(>=0.8.0)
- netCDF4
- texttable

Just use the command:
```
python3 setup.py install --user
```

## Data

In the example folder, dataset are given to test the library, corresponding to
the French 2003 and 2019 French heatwave. Data come from:

- [CMIP5](https://esgf-node.llnl.gov/projects/cmip5/)
- [EOBS](https://www.ecad.eu/download/ensembles/download.php)
- [HadCRUT4](https://crudata.uea.ac.uk/cru/data/temperature/)

## Example

![Alt](/figures/Intensities.png)
![Alt](/figures/Probabilities.png)


## License

Copyright Yoann Robin, Aurelien Ribes, 2020

This software is a computer program that is part of the NSSEA
(Non-Stationary Statistics for Extreme Attribution) This library makes it
possible to infer the probability of an (extreme) event in the factual /
counter-factual world (without anthropic forcing) to attribute it to climate
change.

This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the 
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also 
therefore means  that it is reserved for developers  and  experienced 
professionals having in-depth computer knowledge. Users are therefore 
encouraged to load and test the software's suitability as regards their 
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the   
same conditions as regards security.                                    

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.    

