CDF  3   
      lon       lat       time       bnds            CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Mon Aug 05 12:37:04 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_MPI-ESM-MR_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HWJuly19/tas/Y//tas_day_MPI-M_MPI-ESM-MR.nc
Mon Aug 05 12:37:04 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_MPI-ESM-MR_seldate.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_MPI-ESM-MR_setref.nc
Mon Aug 05 12:37:04 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_MPI-ESM-MR_merge.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_MPI-ESM-MR_seldate.nc
Mon Aug 05 12:37:04 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_18500101-18591231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_18600101-18691231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_18700101-18791231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_18800101-18891231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_18900101-18991231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_19000101-19091231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_19100101-19191231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_historical_r1i1p1_19200101-19291231_tmp_yearmean.nc
Mon Aug 05 12:37:02 2019: cdo yearmean /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_yearmean.nc
Mon Aug 05 12:37:02 2019: cdo -mergetime -selmon,7 -selday,23/25 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_seltime.nc
Mon Aug 05 12:37:02 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_fldmean.nc
Mon Aug 05 12:37:02 2019: cdo ifthen /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/land_sea_mask.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_var_mask.nc
Mon Aug 05 12:37:00 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_box.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_var.nc
Mon Aug 05 12:36:57 2019: cdo sellonlatbox,-5,10,42,51 /bdd/CMIP5/main/MPI-M/MPI-ESM-MR/rcp85/day/atmos/day/r1i1p1/latest/tas/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_MPI-ESM-MR_rcp85_r1i1p1_20900101-21001231_tmp_box.nc
Model raw output postprocessing with modelling environment (IMDI) at DKRZ: URL: http://svn-mad.zmaw.de/svn/mad/Model/IMDI/trunk, REV: 3998 2011-11-15T20:56:37Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.    source        �MPI-ESM-MR 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 4936), T63L47; land: JSBACH (REV: 4936); ocean: MPIOM (REV: 4936), GR15L40; sea ice: 4936; marine bgc: HAMOCC (REV: 4936);   institution       $Max Planck Institute for Meteorology   institute_id      MPI-M      experiment_id         
historical     model_id      
MPI-ESM-MR     forcing       GHG,Oz,SD,Sl,Vl,LU     parent_experiment_id      N/A    parent_experiment_rip         N/A    branch_time                  contact       cmip5-mpi-esm@dkrz.de      
references       �ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8;  MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127;  HAMOCC: Technical Documentation, http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;    initialization_method               physics_version             tracking_id       $16da87d6-eec3-4404-986b-5bb6e1d61165   product       output     
experiment        
historical     	frequency         day    creation_date         2011-10-08T16:30:06Z   
project_id        CMIP5      table_id      :Table day (27 April 2011) 86d1558d99b6ed1e7a886ab3fd717b58     title         5MPI-ESM-MR model output prepared for CMIP5 historical      parent_experiment         N/A    modeling_realm        atmos      realization             cmor_version      2.6.0      CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      proleptic_gregorian         �   	time_bnds                            �   tas                       standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   cell_methods      time: mean     history       J2011-10-08T16:30:06Z altered by CMOR: Treated scalar dimension: 'height'.      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_MPI-ESM-MR_historical_r0i0p0.nc areacella: areacella_fx_MPI-ESM-MR_historical_r0i0p0.nc                          @��@�,@��<�#��@��D�4MC��Y@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�uJ@��@�,@��<�#��@��D�4MC�Ȟ@��@�,@��<�#��@��D�4MC�w�@��@�,@��<�#��@��D�4MC�-	@��@�,@��<�#��@��D�4MC��Y@��@�,@��<�#��@��D�4MC�n^@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��%@��@�,@��<�#��@��D�4MC��y@��@�,@��<�#��@��D�4MC��M@��@�,@��<�#��@��D�4MC�)<@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�dv@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��n@��@�,@��<�#��@��D�4MC���@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�ya@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��T@�@�,@�<�#��@�D�4MC�S�@�@�,@�<�#��@�D�4MC��q@�
@�,@�
<�#��@�
D�4MC��@�
@�,@�
<�#��@�
D�4MC���@�
@�,@�
<�#��@�
D�4MC�B7@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��h@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC��O@�@�,@�<�#��@�D�4MC�{�@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC��]@�@�,@�<�#��@�D�4MC�=@�@�,@�<�#��@�D�4MC��>@�@�,@�<�#��@�D�4MC�_a@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��;@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�(x@�"@�,@�"<�#��@�"D�4MC�^@�"@�,@�"<�#��@�"D�4MC���@�"@�,@�"<�#��@�"D�4MC��@�&@�,@�&<�#��@�&D�4MC���@�&@�,@�&<�#��@�&D�4MC���@�&@�,@�&<�#��@�&D�4MC�j@�*@�,@�*<�#��@�*D�4MC��M@�*@�,@�*<�#��@�*D�4MC���@�*@�,@�*<�#��@�*D�4MC��{@�.@�,@�.<�#��@�.D�4MC��p@�.@�,@�.<�#��@�.D�4MC���@�.@�,@�.<�#��@�.D�4MC�`�@�2@�,@�2<�#��@�2D�4MC��@�2@�,@�2<�#��@�2D�4MC���@�2@�,@�2<�#��@�2D�4MC�/@�6@�,@�6<�#��@�6D�4MC�/@�6@�,@�6<�#��@�6D�4MC���@�6@�,@�6<�#��@�6D�4MC��=@�:@�,@�:<�#��@�:D�4MC���@�:@�,@�:<�#��@�:D�4MC��@�:@�,@�:<�#��@�:D�4MC���@�>@�,@�><�#��@�>D�4MC�H�@�>@�,@�><�#��@�>D�4MC��@�>@�,@�><�#��@�>D�4MC�y�@�B@�,@�B<�#��@�BD�4MC�F@�B@�,@�B<�#��@�BD�4MC�f�@�B@�,@�B<�#��@�BD�4MC�kC@�F@�,@�F<�#��@�FD�4MC�@�F@�,@�F<�#��@�FD�4MC��@�F@�,@�F<�#��@�FD�4MC�5(@�J@�,@�J<�#��@�JD�4MC��~@�J@�,@�J<�#��@�JD�4MC���@�J@�,@�J<�#��@�JD�4MC��@�N@�,@�N<�#��@�ND�4MC�7$@�N@�,@�N<�#��@�ND�4MC�M"@�N@�,@�N<�#��@�ND�4MC�*�@�R@�,@�R<�#��@�RD�4MC�n�@�R@�,@�R<�#��@�RD�4MC���@�R@�,@�R<�#��@�RD�4MC�@�@�V@�,@�V<�#��@�VD�4MC��N@�V@�,@�V<�#��@�VD�4MC�z�@�V@�,@�V<�#��@�VD�4MC��W@�Z@�,@�Z<�#��@�ZD�4MC��@�Z@�,@�Z<�#��@�ZD�4MC�K@�Z@�,@�Z<�#��@�ZD�4MC��@�^@�,@�^<�#��@�^D�4MC�@�^@�,@�^<�#��@�^D�4MC�:@�^@�,@�^<�#��@�^D�4MC���@�b@�,@�b<�#��@�bD�4MC��@�b@�,@�b<�#��@�bD�4MC�~:@�b@�,@�b<�#��@�bD�4MC��@�f@�,@�f<�#��@�fD�4MC���@�f@�,@�f<�#��@�fD�4MC�6�@�f@�,@�f<�#��@�fD�4MC��k@�j@�,@�j<�#��@�jD�4MC���@�j@�,@�j<�#��@�jD�4MC�*�@�j@�,@�j<�#��@�jD�4MC���@�n@�,@�n<�#��@�nD�4MC���@�n@�,@�n<�#��@�nD�4MC�S@�n@�,@�n<�#��@�nD�4MC��l@�r@�,@�r<�#��@�rD�4MC�`n@�r@�,@�r<�#��@�rD�4MC��q@�r@�,@�r<�#��@�rD�4MC�
@�v@�,@�v<�#��@�vD�4MC�'@�v@�,@�v<�#��@�vD�4MC��@�v@�,@�v<�#��@�vD�4MC��>@�z@�,@�z<�#��@�zD�4MC�@�z@�,@�z<�#��@�zD�4MC���@�z@�,@�z<�#��@�zD�4MC�� @�~@�,@�~<�#��@�~D�4MC�*@�~@�,@�~<�#��@�~D�4MC�!0@�~@�,@�~<�#��@�~D�4MC��l@��@�,@��<�#��@��D�4MC��>@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�^�@��@�,@��<�#��@��D�4MC�c @��@�,@��<�#��@��D�4MC�;�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�?X@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�e�@��@�,@��<�#��@��D�4MC�h�@��@�,@��<�#��@��D�4MC��7@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�B@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC�W�@��@�,@��<�#��@��D�4MC�?�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�9q@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�/�@��@�,@��<�#��@��D�4MC�/F@��@�,@��<�#��@��D�4MC��[@��@�,@��<�#��@��D�4MC��d@��@�,@��<�#��@��D�4MC�Z�@��@�,@��<�#��@��D�4MC�m�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�Ѹ@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�]P@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�S�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�7�@��@�,@��<�#��@��D�4MC�K�@��@�,@��<�#��@��D�4MC�e�@��@�,@��<�#��@��D�4MC�n�@��@�,@��<�#��@��D�4MC�N�@��@�,@��<�#��@��D�4MC��*@��@�,@��<�#��@��D�4MC�_@��@�,@��<�#��@��D�4MC�'�@��@�,@��<�#��@��D�4MC�,4@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�~�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�Q�@��@�,@��<�#��@��D�4MC�n@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��k@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC� <@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�G�@��@�,@��<�#��@��D�4MC�?�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��o@��@�,@��<�#��@��D�4MC�-@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��e@��@�,@��<�#��@��D�4MC�J@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�v;@��@�,@��<�#��@��D�4MC��G@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�/�@��@�,@��<�#��@��D�4MC��a@��@�,@��<�#��@��D�4MC�x+@��@�,@��<�#��@��D�4MC��1@��@�,@��<�#��@��D�4MC�0�@��@�,@��<�#��@��D�4MC�B�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�vT@��@�,@��<�#��@��D�4MC�G�@��@�,@��<�#��@��D�4MC�q?@��@�,@��<�#��@��D�4MC��<@��@�,@��<�#��@��D�4MC�kG@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@�@�,@�<�#��@�D�4MC�Fj@�@�,@�<�#��@�D�4MC��3@�@�,@�<�#��@�D�4MC�:�@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�	�@�@�,@�<�#��@�D�4MC��k@�
@�,@�
<�#��@�
D�4MC�HC@�
@�,@�
<�#��@�
D�4MC��a@�
@�,@�
<�#��@�
D�4MC��@�@�,@�<�#��@�D�4MC��9@�@�,@�<�#��@�D�4MC��x@�@�,@�<�#��@�D�4MC�d�@�@�,@�<�#��@�D�4MC�D�@�@�,@�<�#��@�D�4MC�k&@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC�Q#@�@�,@�<�#��@�D�4MC��4@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC� +@�@�,@�<�#��@�D�4MC��k@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��@�"@�,@�"<�#��@�"D�4MC��@�"@�,@�"<�#��@�"D�4MC���@�"@�,@�"<�#��@�"D�4MC���@�&@�,@�&<�#��@�&D�4MC��L@�&@�,@�&<�#��@�&D�4MC��S@�&@�,@�&<�#��@�&D�4MC��:@�*@�,@�*<�#��@�*D�4MC��w@�*@�,@�*<�#��@�*D�4MC�c#@�*@�,@�*<�#��@�*D�4MC��.@�.@�,@�.<�#��@�.D�4MC��P@�.@�,@�.<�#��@�.D�4MC�"�@�.@�,@�.<�#��@�.D�4MC�6*@�2@�,@�2<�#��@�2D�4MC�:�@�2@�,@�2<�#��@�2D�4MC���@�2@�,@�2<�#��@�2D�4MC�7]@�6@�,@�6<�#��@�6D�4MC�>@�6@�,@�6<�#��@�6D�4MC�?W@�6@�,@�6<�#��@�6D�4MC��w@�:@�,@�:<�#��@�:D�4MC�b�@�:@�,@�:<�#��@�:D�4MC�K1@�:@�,@�:<�#��@�:D�4MC�9�@�>@�,@�><�#��@�>D�4MC�>q@�>@�,@�><�#��@�>D�4MC�!�@�>@�,@�><�#��@�>D�4MC�,�@�B@�,@�B<�#��@�BD�4MC��@@�B@�,@�B<�#��@�BD�4MC���@�B@�,@�B<�#��@�BD�4MC�jf@�F@�,@�F<�#��@�FD�4MC�k�@�F@�,@�F<�#��@�FD�4MC�I�@�F@�,@�F<�#��@�FD�4MC��>@�J@�,@�J<�#��@�JD�4MC�W@�J@�,@�J<�#��@�JD�4MC�Wn@�J@�,@�J<�#��@�JD�4MC��l@�N@�,@�N<�#��@�ND�4MC�Q�@�N@�,@�N<�#��@�ND�4MC�h@�N@�,@�N<�#��@�ND�4MC�,�@�R@�,@�R<�#��@�RD�4MC���@�R@�,@�R<�#��@�RD�4MC�s�@�R@�,@�R<�#��@�RD�4MC���@�V@�,@�V<�#��@�VD�4MC�vl@�V@�,@�V<�#��@�VD�4MC�M�@�V@�,@�V<�#��@�VD�4MC�+�@�Z@�,@�Z<�#��@�ZD�4MC�>-@�Z@�,@�Z<�#��@�ZD�4MC�	A@�Z@�,@�Z<�#��@�ZD�4MC��H@�^@�,@�^<�#��@�^D�4MC�D�@�^@�,@�^<�#��@�^D�4MC��@�^@�,@�^<�#��@�^D�4MC��|@�b@�,@�b<�#��@�bD�4MC���@�b@�,@�b<�#��@�bD�4MC��@�b@�,@�b<�#��@�bD�4MC�;t@�f@�,@�f<�#��@�fD�4MC��D@�f@�,@�f<�#��@�fD�4MC�}U@�f@�,@�f<�#��@�fD�4MC�V�@�j@�,@�j<�#��@�jD�4MC�'V@�j@�,@�j<�#��@�jD�4MC��k@�j@�,@�j<�#��@�jD�4MC�r�@�n@�,@�n<�#��@�nD�4MC�/3@�n@�,@�n<�#��@�nD�4MC�&�@�n@�,@�n<�#��@�nD�4MC��`@�r@�,@�r<�#��@�rD�4MC��f@�r@�,@�r<�#��@�rD�4MC�:�@�r@�,@�r<�#��@�rD�4MC��@�v@�,@�v<�#��@�vD�4MC�y�@�v@�,@�v<�#��@�vD�4MC�r�@�v@�,@�v<�#��@�vD�4MC��J@�z@�,@�z<�#��@�zD�4MC���@�z@�,@�z<�#��@�zD�4MC���@�z@�,@�z<�#��@�zD�4MC���@�~@�,@�~<�#��@�~D�4MC��@�~@�,@�~<�#��@�~D�4MC�i�@�~@�,@�~<�#��@�~D�4MC�\5@��@�,@��<�#��@��D�4MC��0@��@�,@��<�#��@��D�4MC�Y@��@�,@��<�#��@��D�4MC�d�@��@�,@��<�#��@��D�4MC�
�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��G@��@�,@��<�#��@��D�4MC�b�@��@�,@��<�#��@��D�4MC�1�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��:@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�2x@��@�,@��<�#��@��D�4MC�p�@��@�,@��<�#��@��D�4MC�C�@��@�,@��<�#��@��D�4MC��0@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�kO@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��Z@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��$@��@�,@��<�#��@��D�4MC��V@��@�,@��<�#��@��D�4MC�yu@��@�,@��<�#��@��D�4MC��{@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�i<@��@�,@��<�#��@��D�4MC�'@��@�,@��<�#��@��D�4MC�.;@��@�,@��<�#��@��D�4MC��5@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�З@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�C�@��@�,@��<�#��@��D�4MC��5@��@�,@��<�#��@��D�4MC�T7@��@�,@��<�#��@��D�4MC�*�@��@�,@��<�#��@��D�4MC�F�@��@�,@��<�#��@��D�4MC�+{@��@�,@��<�#��@��D�4MC�̐@��@�,@��<�#��@��D�4MC�]�@��@�,@��<�#��@��D�4MC�θ@��@�,@��<�#��@��D�4MC��C@��@�,@��<�#��@��D�4MC�\/@��@�,@��<�#��@��D�4MC�	5@��@�,@��<�#��@��D�4MC�<�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��>@��@�,@��<�#��@��D�4MC�x@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�`u@��@�,@��<�#��@��D�4MC��t@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�A1@��@�,@��<�#��@��D�4MC��=@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�0\@��@�,@��<�#��@��D�4MC�Y�@��@�,@��<�#��@��D�4MC�-@��@�,@��<�#��@��D�4MC��w@��@�,@��<�#��@��D�4MC�w8@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�5�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�4�@��@�,@��<�#��@��D�4MC�5@��@�,@��<�#��@��D�4MC�I@��@�,@��<�#��@��D�4MC�@�@��@�,@��<�#��@��D�4MC�94@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�[T@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�	@��@�,@��<�#��@��D�4MC�{�@��@�,@��<�#��@��D�4MC�7@��@�,@��<�#��@��D�4MC�>C@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�#@��@�,@��<�#��@��D�4MC��x@��@�,@��<�#��@��D�4MC�e�@��@�,@��<�#��@��D�4MC�!�@��@�,@��<�#��@��D�4MC�:@��@�,@��<�#��@��D�4MC�,@��@�,@��<�#��@��D�4MC�eH@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC�^�@��@�,@��<�#��@��D�4MC��T@��@�,@��<�#��@��D�4MC�l@��@�,@��<�#��@��D�4MC��@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�վ@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC�n}@�@�,@�<�#��@�D�4MC�n*@�
@�,@�
<�#��@�
D�4MC�cx@�
@�,@�
<�#��@�
D�4MC���@�
@�,@�
<�#��@�
D�4MC�T@�@�,@�<�#��@�D�4MC�9�@�@�,@�<�#��@�D�4MC�#�@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�T�@�@�,@�<�#��@�D�4MC�<F@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�h@�@�,@�<�#��@�D�4MC�ח@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��J@�@�,@�<�#��@�D�4MC��e@�@�,@�<�#��@�D�4MC�A0@�@�,@�<�#��@�D�4MC��c@�@�,@�<�#��@�D�4MC�Z�@�@�,@�<�#��@�D�4MC�V�@�"@�,@�"<�#��@�"D�4MC�T<@�"@�,@�"<�#��@�"D�4MC�ԉ@�"@�,@�"<�#��@�"D�4MC�
�@�&@�,@�&<�#��@�&D�4MC���@�&@�,@�&<�#��@�&D�4MC��@�&@�,@�&<�#��@�&D�4MC�	@�*@�,@�*<�#��@�*D�4MC��1@�*@�,@�*<�#��@�*D�4MC���@�*@�,@�*<�#��@�*D�4MC���@�.@�,@�.<�#��@�.D�4MC�%�@�.@�,@�.<�#��@�.D�4MC���@�.@�,@�.<�#��@�.D�4MC���@�2@�,@�2<�#��@�2D�4MC�j�@�2@�,@�2<�#��@�2D�4MC�%f@�2@�,@�2<�#��@�2D�4MC�W�@�6@�,@�6<�#��@�6D�4MC�ϧ@�6@�,@�6<�#��@�6D�4MC�^f@�6@�,@�6<�#��@�6D�4MC�$o@�:@�,@�:<�#��@�:D�4MC�ڐ@�:@�,@�:<�#��@�:D�4MC�vH@�:@�,@�:<�#��@�:D�4MC�+�@�>@�,@�><�#��@�>D�4MC�{�@�>@�,@�><�#��@�>D�4MC�hT@�>@�,@�><�#��@�>D�4MC�Tg@�B@�,@�B<�#��@�BD�4MC�Ej@�B@�,@�B<�#��@�BD�4MC��0@�B@�,@�B<�#��@�BD�4MC��@�F@�,@�F<�#��@�FD�4MC�_>@�F@�,@�F<�#��@�FD�4MC��@�F@�,@�F<�#��@�FD�4MC���@�J@�,@�J<�#��@�JD�4MC��@�J@�,@�J<�#��@�JD�4MC���@�J@�,@�J<�#��@�JD�4MC���@�N@�,@�N<�#��@�ND�4MC��)@�N@�,@�N<�#��@�ND�4MC��Q@�N@�,@�N<�#��@�ND�4MC���@�R@�,@�R<�#��@�RD�4MC���@�R@�,@�R<�#��@�RD�4MC��o@�R@�,@�R<�#��@�RD�4MC�;�@�V@�,@�V<�#��@�VD�4MC��@�V@�,@�V<�#��@�VD�4MC�lN@�V@�,@�V<�#��@�VD�4MC���@�Z@�,@�Z<�#��@�ZD�4MC��@�^@�,@�^<�#��@�^D�4MC��0@�b@�,@�b<�#��@�bD�4MC��u@�f@�,@�f<�#��@�fD�4MC��N@�j@�,@�j<�#��@�jD�4MC�+�@�n@�,@�n<�#��@�nD�4MC�0G@�r@�,@�r<�#��@�rD�4MC�Hp@�v@�,@�v<�#��@�vD�4MC��$@�z@�,@�z<�#��@�zD�4MC���@�~@�,@�~<�#��@�~D�4MC�N�@��@�,@��<�#��@��D�4MC�ʽ@��@�,@��<�#��@��D�4MC�w@��@�,@��<�#��@��D�4MC�,=@��@�,@��<�#��@��D�4MC� p@��@�,@��<�#��@��D�4MC��{@��@�,@��<�#��@��D�4MC�P�@��@�,@��<�#��@��D�4MC�K�@��@�,@��<�#��@��D�4MC�\@��@�,@��<�#��@��D�4MC��4@��@�,@��<�#��@��D�4MC��>@��@�,@��<�#��@��D�4MC�u�@��@�,@��<�#��@��D�4MC�:�@��@�,@��<�#��@��D�4MC�;�@��@�,@��<�#��@��D�4MC�$@��@�,@��<�#��@��D�4MC��$@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��&@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�g�@��@�,@��<�#��@��D�4MC��d@��@�,@��<�#��@��D�4MC��,@��@�,@��<�#��@��D�4MC��t@��@�,@��<�#��@��D�4MC�{Z@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��/@��@�,@��<�#��@��D�4MC�W�@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��>@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�p�@� X�@�G��y@�"h�&�C�{@� X�@�G��y@�"h�&�C�x�@� X�@�G��y@�"h�&�C�6�@� X�@�G��y@�"h�&�C�{�@�	 X�@�	G��y@�	"h�&�C���@� X�@�G��y@�"h�&�C�\�@� X�@�G��y@�"h�&�C���@� X�@�G��y@�"h�&�C�J�@� X�@�G��y@�"h�&�C�W�@� X�@�G��y@�"h�&�C�E%@� X�@�G��y@�"h�&�C��@� X�@�G��y@�"h�&�C��>@� X�@�G��y@�"h�&�C��[@� X�@�G��y@�"h�&�C��#@� X�@�G��y@�"h�&�C��V@� X�@�G��y@�"h�&�C��S@�! X�@�!G��y@�!"h�&�C�jl@�# X�@�#G��y@�#"h�&�C���@�% X�@�%G��y@�%"h�&�C��@�' X�@�'G��y@�'"h�&�C�H�@�) X�@�)G��y@�)"h�&�C�X�@�+ X�@�+G��y@�+"h�&�C�#W@�- X�@�-G��y@�-"h�&�C���@�/ X�@�/G��y@�/"h�&�C�
@�1 X�@�1G��y@�1"h�&�C���@�3 X�@�3G��y@�3"h�&�C���@�5 X�@�5G��y@�5"h�&�C�sY@�7 X�@�7G��y@�7"h�&�C��O@�9 X�@�9G��y@�9"h�&�C���@�; X�@�;G��y@�;"h�&�C�)@�= X�@�=G��y@�="h�&�C�9	@�? X�@�?G��y@�?"h�&�C��}@�A X�@�AG��y@�A"h�&�C�@�@�C X�@�CG��y@�C"h�&�C�g/@�E X�@�EG��y@�E"h�&�C�%�@�G X�@�GG��y@�G"h�&�C�v@�I X�@�IG��y@�I"h�&�C�@�K X�@�KG��y@�K"h�&�C��@�M X�@�MG��y@�M"h�&�C��U@�O X�@�OG��y@�O"h�&�C��@�Q X�@�QG��y@�Q"h�&�C�� @�S X�@�SG��y@�S"h�&�C�kB@�U X�@�UG��y@�U"h�&�C�ƚ@�W X�@�WG��y@�W"h�&�C�l�@�Y X�@�YG��y@�Y"h�&�C���@�[ X�@�[G��y@�["h�&�C�^%@�] X�@�]G��y@�]"h�&�C�Ӧ@�_ X�@�_G��y@�_"h�&�C�K-@�a X�@�aG��y@�a"h�&�C��<@�c X�@�cG��y@�c"h�&�C�y�@�e X�@�eG��y@�e"h�&�C���@�g X�@�gG��y@�g"h�&�C�Sk@�i X�@�iG��y@�i"h�&�C�=�