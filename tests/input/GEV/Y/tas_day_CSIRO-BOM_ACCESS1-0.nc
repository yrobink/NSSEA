CDF  B   
      lon       lat       time       bnds            CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      MTue Sep 24 15:43:50 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_ACCESS1-0_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HWJuly19D3/tasmax/Y//tas_day_CSIRO-BOM_ACCESS1-0.nc
Tue Sep 24 15:43:50 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_ACCESS1-0_seldate.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_ACCESS1-0_setref.nc
Tue Sep 24 15:43:50 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_ACCESS1-0_merge.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_ACCESS1-0_seldate.nc
Tue Sep 24 15:43:50 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_historical_r1i1p1_18500101-18741231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_historical_r1i1p1_18750101-18991231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_historical_r1i1p1_19000101-19241231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_historical_r1i1p1_19250101-19491231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_historical_r1i1p1_19500101-19741231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_historical_r1i1p1_19750101-19991231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_historical_r1i1p1_20000101-20051231_tmp_yearmean.nc
Tue Sep 24 15:43:49 2019: cdo yearmax /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_yearmean.nc
Tue Sep 24 15:43:49 2019: cdo -runmean,3 /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_runmean.nc
Tue Sep 24 15:43:49 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_fldmean.nc
Tue Sep 24 15:43:49 2019: cdo ifthen /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/land_sea_mask.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_var_mask.nc
Tue Sep 24 15:43:47 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_box.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_var.nc
Tue Sep 24 15:43:42 2019: cdo sellonlatbox,-5,10,42,51 /bdd/CMIP5/main/CSIRO-BOM/ACCESS1-0/rcp85/day/atmos/day/r1i1p1/latest/tas/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_ACCESS1-0_rcp85_r1i1p1_20810101-21001231_tmp_box.nc
CMIP5 compliant file produced from raw ACCESS model output using the ACCESS Post-Processor and CMOR2. 2013-02-26T03:20:00Z CMOR rewrote data to comply with CF standards and CMIP5 requirements. Wed Feb 27 11:21:32 2013: updated version number to v20130227.      source       ACCESS1-0 2011. Atmosphere: AGCM v1.0 (N96 grid-point, 1.875 degrees EW x approx 1.25 degree NS, 38 levels); ocean: NOAA/GFDL MOM4p1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S, 50 levels); sea ice: CICE4.1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S); land: MOSES2 (1.875 degree EW x 1.25 degree NS, 4 levels    institution       {CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), and BOM (Bureau of Meteorology, Australia)    institute_id      	CSIRO-BOM      experiment_id         
historical     model_id      	ACCESS1-0      forcing       aGHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)      parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @��p       contact       �The ACCESS wiki: http://wiki.csiro.au/confluence/display/ACCESS/Home. Contact Tony.Hirst@csiro.au regarding the ACCESS coupled climate model. Contact Peter.Uhe@csiro.au regarding ACCESS coupled climate model CMIP5 datasets.    
references        FSee http://wiki.csiro.au/confluence/display/ACCESS/ACCESS+Publications     initialization_method               physics_version             tracking_id       $fc853c2a-f038-4614-95be-713fa0fcab74   version_number        	v20130227      product       output     
experiment        
historical     	frequency         day    creation_date         2013-02-26T02:21:38Z   
project_id        CMIP5      table_id      =Table day (01 February 2012) b6353e9919862612c81d65cae757c88a      title         4ACCESS1-0 model output prepared for CMIP5 historical   parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.8.0      CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           T   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           \   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      proleptic_gregorian         d   	time_bnds                            l   tas                       standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   cell_methods      time: mean     history       �2013-02-26T02:21:38Z altered by CMOR: Treated scalar dimension: 'height'. 2013-02-26T02:21:38Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_ACCESS1-0_historical_r0i0p0.nc areacella: areacella_fx_ACCESS1-0_historical_r0i0p0.nc            |                @���`X@��     @����,C��@���`X@��     @����,C��`@���`X@��     @����,C��@��!B@���?O��@����,C�!y@��!B@���?O��@����,C�p�@��!B@���?O��@����,C�N}@����,@���?O��@����,C�#�@����,@���?O��@����,C��@����,@���?O��@����,C�&�@��!B@���?O��@����,C���@��!B@���?O��@����,C��o@��!B@���?O��@����,C�t�@��!B@���?O��@����,C�� @��!B@���?O��@����,C���@��!B@���?O��@����,C�q�@��!B@���?O��@� ��,C�46@��!B@���?O��@� ��,C��@��!B@���?O��@� ��,C�`@���,@���?O��@���,C��@���,@���?O��@���,C��,@���,@���?O��@���,C�q�@�!B@��?O��@���,C�u�@�!B@��?O��@���,C�d/@�!B@��?O��@���,C�@�
!B@��?O��@���,C��^@�
!B@��?O��@���,C�4�@�
!B@��?O��@���,C���@�!B@��?O��@���,C�!�@�!B@��?O��@���,C�a�@�!B@��?O��@���,C���@���,@��?O��@���,C�ֻ@���,@��?O��@���,C��'@���,@��?O��@���,C�o�@�!B@��?O��@���,C�i�@�!B@��?O��@���,C�^@�!B@��?O��@���,C��,@�!B@��?O��@���,C��
@�!B@��?O��@���,C�yj@�!B@��?O��@���,C�_@�!B@��?O��@� ��,C��@�!B@��?O��@� ��,C�WF@�!B@��?O��@� ��,C���@�"��,@��?O��@�$��,C�#�@�"��,@��?O��@�$��,C�?@�"��,@��?O��@�$��,C��U@�&!B@�#�?O��@�(��,C�z�@�&!B@�#�?O��@�(��,C��@�&!B@�#�?O��@�(��,C���@�*!B@�'�?O��@�,��,C�mC@�*!B@�'�?O��@�,��,C��~@�*!B@�'�?O��@�,��,C���@�.!B@�+�?O��@�0��,C��@�.!B@�+�?O��@�0��,C��f@�.!B@�+�?O��@�0��,C���@�2��,@�/�?O��@�4��,C�*C@�2��,@�/�?O��@�4��,C�D�@�2��,@�/�?O��@�4��,C�H�@�6!B@�3�?O��@�8��,C�>s@�6!B@�3�?O��@�8��,C��6@�6!B@�3�?O��@�8��,C�Gu@�:!B@�7�?O��@�<��,C��l@�:!B@�7�?O��@�<��,C�n�@�:!B@�7�?O��@�<��,C��1@�>!B@�;�?O��@�@��,C�Xj@�>!B@�;�?O��@�@��,C�n@�>!B@�;�?O��@�@��,C���@�B��,@�?�?O��@�D��,C��6@�B��,@�?�?O��@�D��,C�ׅ@�B��,@�?�?O��@�D��,C��n@�F!B@�C�?O��@�H��,C��@�F!B@�C�?O��@�H��,C���@�F!B@�C�?O��@�H��,C�\�@�J��,@�G�?O��@�L     C�-@�J��,@�G�?O��@�L     C��F@�J��,@�G�?O��@�L     C�Nb@�N�`X@�L     @�P��,C�:�@�N�`X@�L     @�P��,C��@�N�`X@�L     @�P��,C���@�R��,@�O�?O��@�T��,C�9v@�R��,@�O�?O��@�T��,C��+@�R��,@�O�?O��@�T��,C��~@�V!B@�S�?O��@�X��,C���@�V!B@�S�?O��@�X��,C���@�V!B@�S�?O��@�X��,C��@�Z!B@�W�?O��@�\��,C��.@�Z!B@�W�?O��@�\��,C�}3@�Z!B@�W�?O��@�\��,C�]�@�^!B@�[�?O��@�`��,C�#�@�^!B@�[�?O��@�`��,C���@�^!B@�[�?O��@�`��,C��w@�b��,@�_�?O��@�d��,C�T�@�b��,@�_�?O��@�d��,C���@�b��,@�_�?O��@�d��,C� @�f!B@�c�?O��@�h��,C�]@�f!B@�c�?O��@�h��,C�hV@�f!B@�c�?O��@�h��,C���@�j!B@�g�?O��@�l��,C��@�j!B@�g�?O��@�l��,C��q@�j!B@�g�?O��@�l��,C��=@�n!B@�k�?O��@�p��,C�7�@�n!B@�k�?O��@�p��,C�I@�n!B@�k�?O��@�p��,C�m@�r��,@�o�?O��@�t��,C���@�r��,@�o�?O��@�t��,C�D�@�r��,@�o�?O��@�t��,C���@�v!B@�s�?O��@�x��,C��#@�v!B@�s�?O��@�x��,C���@�v!B@�s�?O��@�x��,C��l@�z!B@�w�?O��@�|��,C��_@�z!B@�w�?O��@�|��,C�k'@�z!B@�w�?O��@�|��,C� @�~!B@�{�?O��@����,C��2@�~!B@�{�?O��@����,C��@�~!B@�{�?O��@����,C���@����,@��?O��@����,C�y�@����,@��?O��@����,C�N@����,@��?O��@����,C�q@��!B@���?O��@����,C��D@��!B@���?O��@����,C��@��!B@���?O��@����,C��@��!B@���?O��@����,C�ɕ@��!B@���?O��@����,C�� @��!B@���?O��@����,C�k�@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C��I@����,@���?O��@����,C���@����,@���?O��@����,C�E@����,@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C�P�@��!B@���?O��@����,C��Y@��!B@���?O��@����,C�c�@��!B@���?O��@����,C��g@��!B@���?O��@����,C�17@��!B@���?O��@����,C�=�@��!B@���?O��@����,C�y@��!B@���?O��@����,C�"�@����,@���?O��@����,C���@����,@���?O��@����,C�1f@����,@���?O��@����,C�}�@��!B@���?O��@����,C��9@��!B@���?O��@����,C�x0@��!B@���?O��@����,C��=@��!B@���?O��@����,C��@��!B@���?O��@����,C��r@��!B@���?O��@����,C�r�@����,@���?O��@��     C�k�@����,@���?O��@��     C��@����,@���?O��@��     C�Ԫ@���`X@��     @����,C��0@���`X@��     @����,C��@���`X@��     @����,C��@��!B@���?O��@����,C��@��!B@���?O��@����,C�38@��!B@���?O��@����,C�S�@��!B@���?O��@����,C�2Q@��!B@���?O��@����,C�=>@��!B@���?O��@����,C�yW@��!B@���?O��@����,C���@��!B@���?O��@����,C�k�@��!B@���?O��@����,C��R@����,@���?O��@����,C�N@����,@���?O��@����,C��F@����,@���?O��@����,C��N@��!B@���?O��@����,C�p@��!B@���?O��@����,C�Ƃ@��!B@���?O��@����,C��0@��!B@���?O��@����,C���@��!B@���?O��@����,C�_�@��!B@���?O��@����,C�w�@��!B@���?O��@����,C�@��!B@���?O��@����,C�5�@��!B@���?O��@����,C�^@����,@���?O��@����,C�?�@����,@���?O��@����,C���@����,@���?O��@����,C�S@��!B@���?O��@����,C�h�@��!B@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C��9@��!B@���?O��@����,C�?�@��!B@���?O��@����,C�5m@��!B@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C�͒@����,@���?O��@����,C��@����,@���?O��@����,C��c@����,@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C��"@��!B@���?O��@����,C�|�@��!B@���?O��@����,C�]I@��!B@���?O��@����,C��@��!B@���?O��@����,C��@��!B@���?O��@����,C���@��!B@���?O��@����,C��a@��!B@���?O��@����,C��0@����,@���?O��@����,C��8@����,@���?O��@����,C��@����,@���?O��@����,C��@��!B@���?O��@����,C��A@��!B@���?O��@����,C��!@��!B@���?O��@����,C���@��!B@���?O��@����,C�W�@��!B@���?O��@����,C�ʌ@��!B@���?O��@����,C���@��!B@���?O��@� ��,C�b�@��!B@���?O��@� ��,C�P�@��!B@���?O��@� ��,C��@���,@���?O��@���,C�k�@���,@���?O��@���,C�e@���,@���?O��@���,C��@�!B@��?O��@���,C��3@�!B@��?O��@���,C���@�!B@��?O��@���,C���@�
!B@��?O��@���,C�z�@�
!B@��?O��@���,C�M�@�
!B@��?O��@���,C�9�@�!B@��?O��@���,C�0�@�!B@��?O��@���,C���@�!B@��?O��@���,C�r�@�`X@��?O��@�     C���@�`X@��?O��@�     C��@�`X@��?O��@�     C�͇@��`X@�     @���,C�,@��`X@�     @���,C�0�@��`X@�     @���,C��D@�!B@��?O��@���,C���@�!B@��?O��@���,C�E�@�!B@��?O��@���,C���@�!B@��?O��@� ��,C�8q@�!B@��?O��@� ��,C�͓@�!B@��?O��@� ��,C�.@�"��,@��?O��@�$��,C�t@�"��,@��?O��@�$��,C��u@�"��,@��?O��@�$��,C��@�&!B@�#�?O��@�(��,C���@�&!B@�#�?O��@�(��,C�T@�&!B@�#�?O��@�(��,C�f@�*!B@�'�?O��@�,��,C�w@�*!B@�'�?O��@�,��,C��Q@�*!B@�'�?O��@�,��,C��w@�.!B@�+�?O��@�0��,C��@�.!B@�+�?O��@�0��,C�Q8@�.!B@�+�?O��@�0��,C��@�2��,@�/�?O��@�4��,C��=@�2��,@�/�?O��@�4��,C���@�2��,@�/�?O��@�4��,C��(@�6!B@�3�?O��@�8��,C��B@�6!B@�3�?O��@�8��,C��)@�6!B@�3�?O��@�8��,C��>@�:!B@�7�?O��@�<��,C�T@�:!B@�7�?O��@�<��,C�y@�:!B@�7�?O��@�<��,C�N�@�>!B@�;�?O��@�@��,C�y�@�>!B@�;�?O��@�@��,C��@�>!B@�;�?O��@�@��,C���@�B��,@�?�?O��@�D��,C���@�B��,@�?�?O��@�D��,C�ך@�B��,@�?�?O��@�D��,C�~@�F!B@�C�?O��@�H��,C�v:@�F!B@�C�?O��@�H��,C� @�F!B@�C�?O��@�H��,C��1@�J!B@�G�?O��@�L��,C��@�J!B@�G�?O��@�L��,C�8l@�J!B@�G�?O��@�L��,C��;@�N!B@�K�?O��@�P��,C��
@�N!B@�K�?O��@�P��,C��t@�N!B@�K�?O��@�P��,C�D@�R��,@�O�?O��@�T��,C�;�@�R��,@�O�?O��@�T��,C�K}@�R��,@�O�?O��@�T��,C��@�V!B@�S�?O��@�X��,C���@�V!B@�S�?O��@�X��,C�(G@�V!B@�S�?O��@�X��,C�w�@�Z!B@�W�?O��@�\��,C��@�Z!B@�W�?O��@�\��,C�G@�Z!B@�W�?O��@�\��,C��@�^!B@�[�?O��@�`��,C�t�@�^!B@�[�?O��@�`��,C�~)@�^!B@�[�?O��@�`��,C��W@�b��,@�_�?O��@�d��,C��K@�b��,@�_�?O��@�d��,C��J@�b��,@�_�?O��@�d��,C���@�f!B@�c�?O��@�h��,C�-�@�f!B@�c�?O��@�h��,C��.@�f!B@�c�?O��@�h��,C���@�j!B@�g�?O��@�l��,C�}�@�j!B@�g�?O��@�l��,C���@�j!B@�g�?O��@�l��,C�]�@�n!B@�k�?O��@�p��,C��w@�n!B@�k�?O��@�p��,C��@�n!B@�k�?O��@�p��,C���@�r��,@�o�?O��@�t��,C�!"@�r��,@�o�?O��@�t��,C�!V@�r��,@�o�?O��@�t��,C�a/@�v��,@�s�?O��@�x     C�u�@�v��,@�s�?O��@�x     C�d{@�v��,@�s�?O��@�x     C��@�z�`X@�x     @�|��,C�0@�z�`X@�x     @�|��,C��P@�z�`X@�x     @�|��,C�k@�~!B@�{�?O��@����,C���@�~!B@�{�?O��@����,C��f@�~!B@�{�?O��@����,C�:�@����,@��?O��@����,C��=@����,@��?O��@����,C�C@����,@��?O��@����,C���@��!B@���?O��@����,C�?�@��!B@���?O��@����,C�F@��!B@���?O��@����,C�@��!B@���?O��@����,C��@��!B@���?O��@����,C��@��!B@���?O��@����,C�.k@��!B@���?O��@����,C��@��!B@���?O��@����,C��A@��!B@���?O��@����,C��h@����,@���?O��@����,C�N@����,@���?O��@����,C�e�@����,@���?O��@����,C���@��!B@���?O��@����,C�Z@��!B@���?O��@����,C�u�@��!B@���?O��@����,C���@��!B@���?O��@����,C��M@��!B@���?O��@����,C��@��!B@���?O��@����,C�D�@��!B@���?O��@����,C��@��!B@���?O��@����,C�G!@��!B@���?O��@����,C��/@����,@���?O��@����,C�'�@����,@���?O��@����,C�Э@����,@���?O��@����,C��!@��!B@���?O��@����,C���@��!B@���?O��@����,C��n@��!B@���?O��@����,C��E@��!B@���?O��@����,C���@��!B@���?O��@����,C��.@��!B@���?O��@����,C��	@��!B@���?O��@����,C���@��!B@���?O��@����,C�`�@��!B@���?O��@����,C��@����,@���?O��@����,C��7@����,@���?O��@����,C�a�@����,@���?O��@����,C�"�@��!B@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C�{\@��!B@���?O��@����,C���@��!B@���?O��@����,C�=�@��!B@���?O��@����,C�b(@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C���@����,@���?O��@����,C�s�@����,@���?O��@����,C�Ή@����,@���?O��@����,C�9�@��!B@���?O��@����,C�~�@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C�F�@��!B@���?O��@����,C���@��!B@���?O��@����,C�H`@��!B@���?O��@����,C�+�@��!B@���?O��@����,C��m@����,@���?O��@����,C�I�@����,@���?O��@����,C���@����,@���?O��@����,C�ś@��!B@���?O��@����,C��+@��!B@���?O��@����,C��@��!B@���?O��@����,C��v@����,@���?O��@��     C�z@����,@���?O��@��     C���@����,@���?O��@��     C��Z@���`X@��     @����,C�[Y@���`X@��     @����,C��Q@���`X@��     @����,C��@����,@���?O��@����,C���@����,@���?O��@����,C��2@����,@���?O��@����,C��,@��!B@���?O��@����,C�3�@��!B@���?O��@����,C��,@��!B@���?O��@����,C���@��!B@���?O��@����,C��i@��!B@���?O��@����,C�71@��!B@���?O��@����,C�p@��!B@���?O��@����,C��o@��!B@���?O��@����,C���@��!B@���?O��@����,C���@����,@���?O��@����,C�r�@����,@���?O��@����,C�G@����,@���?O��@����,C���@��!B@���?O��@����,C��S@��!B@���?O��@����,C�s@��!B@���?O��@����,C�T@��!B@���?O��@����,C��@��!B@���?O��@����,C��@@��!B@���?O��@����,C���@��!B@���?O��@� ��,C��n@��!B@���?O��@� ��,C���@��!B@���?O��@� ��,C���@���,@���?O��@���,C��%@���,@���?O��@���,C��:@���,@���?O��@���,C�~�@�!B@��?O��@���,C��M@�!B@��?O��@���,C��@�!B@��?O��@���,C�p�@�
!B@��?O��@���,C��m@�
!B@��?O��@���,C��E@�
!B@��?O��@���,C�!@�!B@��?O��@���,C��@�!B@��?O��@���,C�@d@�!B@��?O��@���,C�5�@���,@��?O��@���,C�E5@���,@��?O��@���,C��@���,@��?O��@���,C���@�!B@��?O��@���,C���@�!B@��?O��@���,C���@�!B@��?O��@���,C��@�!B@��?O��@���,C�T�@�!B@��?O��@���,C�
�@�!B@��?O��@���,C���@�!B@��?O��@� ��,C��@�!B@��?O��@� ��,C�yW@�!B@��?O��@� ��,C�6�@�"��,@��?O��@�$��,C���@�"��,@��?O��@�$��,C��@�"��,@��?O��@�$��,C���@�&!B@�#�?O��@�(��,C�c�@�&!B@�#�?O��@�(��,C��J@�&!B@�#�?O��@�(��,C�l@�*!B@�'�?O��@�,��,C�I�@�*!B@�'�?O��@�,��,C�@�*!B@�'�?O��@�,��,C���@�.!B@�+�?O��@�0��,C�Ĝ@�.!B@�+�?O��@�0��,C��:@�.!B@�+�?O��@�0��,C��V@�2��,@�/�?O��@�4��,C�}�@�2��,@�/�?O��@�4��,C���@�2��,@�/�?O��@�4��,C��.@�6!B@�3�?O��@�8��,C��d@�6!B@�3�?O��@�8��,C�T�@�6!B@�3�?O��@�8��,C�I0@�:!B@�7�?O��@�<��,C�u�@�:!B@�7�?O��@�<��,C��9@�:!B@�7�?O��@�<��,C�w@�>��,@�;�?O��@�@     C���@�>��,@�;�?O��@�@     C��D@�>��,@�;�?O��@�@     C�
�@�B!B@�@     @�D��,C�v=@�B!B@�@     @�D��,C���@�B!B@�@     @�D��,C��@�F!B@�C�?O��@�H��,C�kY@�F!B@�C�?O��@�H��,C���@�F!B@�C�?O��@�H��,C�؅@�J!B@�G�?O��@�L��,C�2 @�J!B@�G�?O��@�L��,C��h@�J!B@�G�?O��@�L��,C��h@�N!B@�K�?O��@�P��,C��.@�N!B@�K�?O��@�P��,C�p@�N!B@�K�?O��@�P��,C�^�@�R��,@�O�?O��@�T��,C���@�R��,@�O�?O��@�T��,C�V�@�R��,@�O�?O��@�T��,C�1�@�V��,@�S�?O��@�X     C��c@�V��,@�S�?O��@�X     C�}�@�V��,@�S�?O��@�X     C��*@�Z�`X@�X     @�\��,C�~�@�Z�`X@�X     @�\��,C���@�^!B@�[�?O��@�`��,C�Y@�^!B@�[�?O��@�`��,C��@�b��,@�_�?O��@�d��,C���@�b��,@�_�?O��@�d��,C��@�f!B@�c�?O��@�h��,C�?@�f!B@�c�?O��@�h��,C�l@�j!B@�g�?O��@�l��,C�\a@�j!B@�g�?O��@�l��,C��M@�n!B@�k�?O��@�p��,C��@�n!B@�k�?O��@�p��,C�O�@�r��,@�o�?O��@�t��,C���@�r��,@�o�?O��@�t��,C�H"@�v!B@�s�?O��@�x��,C��[@�v!B@�s�?O��@�x��,C��J@�z!B@�w�?O��@�|��,C���@�z!B@�w�?O��@�|��,C�1@�~!B@�{�?O��@����,C�n�@�~!B@�{�?O��@����,C��@����,@��?O��@����,C��@����,@��?O��@����,C�Y�@��!B@���?O��@����,C��V@��!B@���?O��@����,C��6@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C�\�@��!B@���?O��@����,C�22@��`X@���?O��@��     C�̊@����,@���?O��@����,C���@��!B@���?O��@����,C�tH@��!B@���?O��@����,C��@��!B@���?O��@����,C�j�@����,@���?O��@����,C��w@��!B@���?O��@����,C��V@��!B@���?O��@����,C��l@��!B@���?O��@����,C���@����,@���?O��@����,C�J@��!B@���?O��@����,C��@����,@���?O��@��     C�)\@���`X@��     @����,C�/�@����,@���?O��@����,C���@��!B@���?O��@����,C�߲@��!B@���?O��@����,C���@��!B@���?O��@����,C�6@����,@���?O��@����,C�w@��!B@���?O��@����,C�;@��!B@���?O��@����,C���@��!B@���?O��@����,C��y@����,@���?O��@����,C�L�@��!B@���?O��@����,C��@��!B@���?O��@����,C�|�@��!B@���?O��@����,C���@����,@���?O��@����,C��<@��!B@���?O��@����,C��@��!B@���?O��@����,C�yq@��!B@���?O��@� `XC��@�`X@���?O��@�`XC��)@��!@������@�`XC�D<@��!@������@�`XC���@��!@������@�`XC���@�	`X@������@�
`XC�/�@��!@�	�����@�`XC���@��!@������@�`XC��7@�`X@������@�     C��@��!@�     @�`XC�ږ@��!@������@�`XC��$@��!@������@�`XC���@��!@������@�`XC��&@�`X@������@�`XC��@��!@������@�`XC�s�@��!@������@�`XC��E@��!@������@� `XC�ah@�!`X@������@�"`XC�-w@�#�!@�!�����@�$`XC���@�%�!@�#�����@�&`XC���@�'�!@�%�����@�(`XC���@�)`X@�'�����@�*`XC��@�+�!@�)�����@�,`XC�"@�-�!@�+�����@�.`XC�=6@�/�!@�-�����@�0`XC��?@�1`X@�/�����@�2`XC�5w@�3�!@�1�����@�4`XC��C@�5�!@�3�����@�6`XC��D@�7�!@�5�����@�8`XC��@�9`X@�7�����@�:`XC�N�@�;�!@�9�����@�<`XC�Rp@�=�!@�;�����@�>`XC��@�?�!@�=�����@�@`XC�]�@�A �,@�?�����@�B     C���@�C��,@�B     @�D`XC�W]@�E�!@�C�����@�F`XC���@�G�!@�E�����@�H`XC���@�I`X@�G�����@�J`XC��!@�K�!@�I�����@�L`XC���@�M�!@�K�����@�N`XC��:@�O�!@�M�����@�P`XC�ׇ@�Q`X@�O�����@�R`XC� �@�S�!@�Q�����@�T`XC��u@�U�!@�S�����@�V`XC��v@�W�!@�U�����@�X`XC�}=@�Y`X@�W�����@�Z`XC���@�[�!@�Y�����@�\`XC�N]@�]�!@�[�����@�^`XC�{�@�_�!@�]�����@�``XC�ϟ@�a`X@�_�����@�b`XC��@�c�!@�a�����@�d`XC���@�e�!@�c�����@�f`XC�mi@�g�!@�e�����@�h`XC��@�i`X@�g�����@�j     C�.$