CDF  �   
      lon       lat       time       bnds         &   CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Tue Sep 24 18:13:15 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_CESM1-CAM5_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HWJuly19D3/tasmax/Y//tas_day_NSF-DOE-NCAR_CESM1-CAM5.nc
Tue Sep 24 18:13:15 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_CESM1-CAM5_seldate.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_CESM1-CAM5_setref.nc
Tue Sep 24 18:13:15 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_CESM1-CAM5_merge.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467//tas_day_CESM1-CAM5_seldate.nc
Tue Sep 24 18:13:15 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_historical_r1i1p1_18500101-18841231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_historical_r1i1p1_18850101-19191231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_historical_r1i1p1_19200101-19541231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_historical_r1i1p1_19550101-19891231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_historical_r1i1p1_19900101-20051231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r1i1p1_20060101-20401231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r1i1p1_20410101-20751231_tmp_yearmean.nc
Tue Sep 24 18:13:14 2019: cdo yearmax /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_yearmean.nc
Tue Sep 24 18:13:14 2019: cdo -runmean,3 /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_runmean.nc
Tue Sep 24 18:13:13 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_fldmean.nc
Tue Sep 24 18:13:13 2019: cdo ifthen /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/land_sea_mask.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_var_mask.nc
Tue Sep 24 18:13:11 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_box.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_var.nc
Tue Sep 24 18:12:53 2019: cdo sellonlatbox,-5,10,42,51 /bdd/CMIP5/main/NSF-DOE-NCAR/CESM1-CAM5/rcp85/day/atmos/day/r3i1p1/latest/tas/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231.nc /data/yrobin//tmp/EUPHEME_HWJuly19D3_data_tasmax.sh560467/tas_day_CESM1-CAM5_rcp85_r3i1p1_20760101-21001231_tmp_box.nc
2013-05-29T00:15:15Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        
CESM1-CAM5     institution       HNSF/DOE NCAR (National Center for Atmospheric Research) Boulder, CO, USA   institute_id      NSF-DOE-NCAR   experiment_id         
historical     model_id      
CESM1-CAM5     forcing       $Sl GHG Vl SS Ds SD BC MD OC Oz AA LU   parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @t         contact       cesm_data@ucar.edu     comment       (CESM home page: http://www.cesm.ucar.edu   
references        �Neale, R., et.al. 2012: Coupled simulations from CESM1 using the Community Atmosphere Model version 5: (CAM5). See also http://www.cesm.ucar.edu/publications      initialization_method               physics_version             tracking_id       $f6e337da-e719-4369-bae3-f3a6842aa5fe   acknowledgements     �The CESM project is supported by the National Science Foundation and the Office of Science (BER) of the U.S. Department of Energy. NCAR is sponsored by the National Science Foundation. This research used resources of the Oak Ridge Leadership Computing Facility, located in the National Center for Computational Sciences at Oak Ridge National Laboratory, which is supported by the Office of Science (BER) of the Department of Energy under Contract DE-AC05-00OR22725.      cesm_casename         b.e10.B20TRC5CN.f09_g16.001    cesm_repotag      cesm1_0_beta20     cesm_compset      	B20TRC5CN      
resolution        f09_g16 (0.9x1.25_gx1v6)   forcing_note      �Additional information on the external forcings used in this experiment can be found at http://www.cesm.ucar.edu/CMIP5/forcing_information     processed_by      8strandwg on silver.cgd.ucar.edu at 20130606  -094725.387   processing_code_information       �Last Changed Rev: 1108 Last Changed Date: 2013-04-29 10:59:00 -0600 (Mon, 29 Apr 2013) Repository UUID: d2181dbe-5796-6825-dc7f-cbd98591f93d   product       output     
experiment        
historical     	frequency         day    creation_date         2013-06-06T15:48:13Z   
project_id        CMIP5      table_id      <Table day (12 January 2012) 7757d80c56ae0b9009f150afa4850c4e   title         5CESM1-CAM5 model output prepared for CMIP5 historical      parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.8.1      CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      365_day         �   	time_bnds                            �   tas                    
   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   original_name         TREFHT     comment       TREFHT no change       cell_methods      time: mean (interval: 1 day)       history      2013-06-06T15:47:25Z altered by CMOR: Treated scalar dimension: 'height'. 2013-06-06T15:47:25Z altered by CMOR: Reordered dimensions, original order: lat lon time. 2013-06-06T15:47:25Z altered by CMOR: replaced missing value flag (-1e+32) with standard missing value (1e+20).    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CESM1-CAM5_historical_r0i0p0.nc areacella: areacella_fx_CESM1-CAM5_historical_r0i0p0.nc                          @���`X@��     @����,C���@��!B@���?O��@����,C�%@��!B@���?O��@����,C��@��!B@���?O��@����,C���@��!B@���?O��@����,C�,�@��!B@���?O��@� ��,C�c�@�!B@���?O��@���,C�H@�!B@��?O��@���,C��Y@�
!B@��?O��@���,C���@�!B@��?O��@���,C�FX@�!B@��?O��@���,C�g	@�!B@��?O��@���,C���@�!B@��?O��@���,C�(�@�!B@��?O��@� ��,C���@�"!B@��?O��@�$��,C��`@�&!B@�#�?O��@�(��,C��G@�*!B@�'�?O��@�,��,C�}�@�.!B@�+�?O��@�0��,C��@�2!B@�/�?O��@�4��,C�
�@�6!B@�3�?O��@�8��,C���@�:!B@�7�?O��@�<��,C�y^@�>!B@�;�?O��@�@��,C���@�B!B@�?�?O��@�D��,C�\@�F!B@�C�?O��@�H��,C���@�J!B@�G�?O��@�L��,C��
@�N!B@�K�?O��@�P��,C��Z@�R!B@�O�?O��@�T��,C�q@�V!B@�S�?O��@�X��,C��P@�Z!B@�W�?O��@�\��,C�.@�^!B@�[�?O��@�`��,C���@�b!B@�_�?O��@�d��,C��C@�f!B@�c�?O��@�h��,C��Y@�j!B@�g�?O��@�l��,C���@�n!B@�k�?O��@�p��,C��K@�r��,@�o�?O��@�t     C���@�v�`X@�t     @�x��,C�x@�z!B@�w�?O��@�|��,C��5@�~!B@�{�?O��@����,C���@��!B@��?O��@����,C�Pt@��!B@���?O��@����,C�u@��!B@���?O��@����,C�}�@��!B@���?O��@����,C�wV@��!B@���?O��@����,C��@��!B@���?O��@����,C���@��!B@���?O��@����,C�&�@��!B@���?O��@����,C��0@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C�/u@��!B@���?O��@����,C��
@��!B@���?O��@����,C���@��!B@���?O��@����,C��k@��!B@���?O��@����,C��M@��!B@���?O��@����,C���@��!B@���?O��@����,C�(�@��!B@���?O��@����,C��@��!B@���?O��@����,C�@��!B@���?O��@����,C�<�@��!B@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C���@��!B@���?O��@����,C��&@��!B@���?O��@����,C�:@��!B@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C��P@��!B@���?O��@����,C�-K@��!B@���?O��@����,C�L�@��!B@���?O��@����,C��b@����,@���?O��@�      C�H�@��`X@�      @���,C�_�@�!B@��?O��@���,C�v�@�
!B@��?O��@���,C�3�@�!B@��?O��@���,C�e�@�!B@��?O��@���,C��2@�!B@��?O��@���,C�g�@�!B@��?O��@���,C��f@�!B@��?O��@� ��,C� @�"!B@��?O��@�$��,C��@�&!B@�#�?O��@�(��,C���@�*!B@�'�?O��@�,��,C�kp@�.!B@�+�?O��@�0��,C��@�2!B@�/�?O��@�4��,C��y@�6!B@�3�?O��@�8��,C��@�:!B@�7�?O��@�<��,C��-@�>!B@�;�?O��@�@��,C��q@�B!B@�?�?O��@�D��,C�Ր@�F!B@�C�?O��@�H��,C��@�J!B@�G�?O��@�L��,C��`@�N!B@�K�?O��@�P��,C��_@�R!B@�O�?O��@�T��,C�y�@�V!B@�S�?O��@�X��,��X�@�Z!B@�W�?O��@�\��,C��@�^!B@�[�?O��@�`��,C�G�@�b!B@�_�?O��@�d��,C�"N@�f!B@�c�?O��@�h��,C��2@�j!B@�g�?O��@�l��,C�F@�n!B@�k�?O��@�p��,C�O�@�r!B@�o�?O��@�t��,C��]@�v!B@�s�?O��@�x��,C�@�z!B@�w�?O��@�|��,C�N�@�~!B@�{�?O��@����,C�n�@��!B@��?O��@����,C��6@��!B@���?O��@����,C���@����,@���?O��@��     C���@���`X@��     @����,C��@��!B@���?O��@����,C��@��!B@���?O��@����,C��.@��!B@���?O��@����,C�<@��!B@���?O��@����,C�<\@��!B@���?O��@����,C��r@��!B@���?O��@����,C�A�@��!B@���?O��@����,C�i@��!B@���?O��@����,C��@��!B@���?O��@����,C�y�@��!B@���?O��@����,C��+@��!B@���?O��@����,C�H @��!B@���?O��@����,C��?@��!B@���?O��@����,C�˸@��!B@���?O��@����,C��@��!B@���?O��@����,C��o@��!B@���?O��@����,C�D@��!B@���?O��@����,C�U@��!B@���?O��@����,C�x@��!B@���?O��@����,C���@��!B@���?O��@����,C�!@��!B@���?O��@����,C��@��!B@���?O��@����,C�&@��!B@���?O��@����,C��@��!B@���?O��@����,C���@��!B@���?O��@����,C�Ǝ@��!B@���?O��@����,C���@��!B@���?O��@����,C�
�@��!B@���?O��@� ��,C��@�!B@���?O��@���,C���@�!B@��?O��@���,C�T @�
!B@��?O��@���,C�_�@�!B@��?O��@���,C�}g@�!B@��?O��@���,C�*e@���,@��?O��@�     C�p@��`X@�     @���,C���@�!B@��?O��@� ��,C��@�"!B@��?O��@�$��,C��C@�&!B@�#�?O��@�(��,C�:@�*!B@�'�?O��@�,��,C�T`@�.!B@�+�?O��@�0��,C�U>@�2!B@�/�?O��@�4��,C�tC@�6!B@�3�?O��@�8��,C�b�@�:!B@�7�?O��@�<��,C�:�@�>!B@�;�?O��@�@��,C��W@�B!B@�?�?O��@�D��,C�	�@�F!B@�C�?O��@�H��,C�@�J!B@�G�?O��@�L��,C��K@�N!B@�K�?O��@�P��,C��m@�R!B@�O�?O��@�T��,C�t@�V��,@�S�?O��@�X     C�h@�Z�`X@�X     @�\��,C��s@�Z�`X@�X     @�\��,C�"@�Z�`X@�X     @�\��,C�M�@�^!B@�[�?O��@�`��,C��@�^!B@�[�?O��@�`��,C��s@�^!B@�[�?O��@�`��,C�e�@�b!B@�_�?O��@�d��,C�do@�b!B@�_�?O��@�d��,C�h@�b!B@�_�?O��@�d��,C��@�f!B@�c�?O��@�h��,C�}6@�f!B@�c�?O��@�h��,C��@�f!B@�c�?O��@�h��,C���@�j!B@�g�?O��@�l��,C��x@�j!B@�g�?O��@�l��,C��g@�j!B@�g�?O��@�l��,C��@�n!B@�k�?O��@�p��,C��@�n!B@�k�?O��@�p��,C�MT@�n!B@�k�?O��@�p��,C��@�r!B@�o�?O��@�t��,C�r�@�r!B@�o�?O��@�t��,C��$@�r!B@�o�?O��@�t��,C���@�v!B@�s�?O��@�x��,C�Gw@�v!B@�s�?O��@�x��,C��?@�v!B@�s�?O��@�x��,C��@�z!B@�w�?O��@�|��,C�5�@�z!B@�w�?O��@�|��,C�@�z!B@�w�?O��@�|��,C� �@�~!B@�{�?O��@����,C��H@�~!B@�{�?O��@����,C��@�~!B@�{�?O��@����,C��T@��!B@��?O��@����,C��@��!B@��?O��@����,C���@��!B@��?O��@����,C���@��!B@���?O��@����,C�Q�@��!B@���?O��@����,C��=@��!B@���?O��@����,C��]@��!B@���?O��@����,C��~@��!B@���?O��@����,C���@��!B@���?O��@����,C��{@��!B@���?O��@����,C���@��!B@���?O��@����,C��H@��!B@���?O��@����,C�S@��!B@���?O��@����,C���@��!B@���?O��@����,C�!�@��!B@���?O��@����,C�(P@��!B@���?O��@����,C��@��!B@���?O��@����,C�c@��!B@���?O��@����,C��]@��!B@���?O��@����,C���@��!B@���?O��@����,C�n@��!B@���?O��@����,C���@��!B@���?O��@����,C��]@��!B@���?O��@����,C�ڜ@��!B@���?O��@����,C��@��!B@���?O��@����,C��/@��!B@���?O��@����,C��@��!B@���?O��@����,C�E�@��!B@���?O��@����,C���@��!B@���?O��@����,C�Ύ@��!B@���?O��@����,C�!@��!B@���?O��@����,C��z@��!B@���?O��@����,C�:+@��!B@���?O��@����,C��t@��!B@���?O��@����,C��@��!B@���?O��@����,C�j�@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C���@��!B@���?O��@����,C�et@��!B@���?O��@����,C��n@��!B@���?O��@����,C���@��!B@���?O��@����,C�ǫ@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C�_@��!B@���?O��@����,C���@��!B@���?O��@����,C�"j@��!B@���?O��@����,C�K@��!B@���?O��@����,C��=@��!B@���?O��@����,C�#@��!B@���?O��@����,C��t@��!B@���?O��@����,C���@��!B@���?O��@����,C��R@��!B@���?O��@����,C���@��!B@���?O��@����,C�@��!B@���?O��@����,C�y�@��!B@���?O��@����,C���@��!B@���?O��@����,C�Mo@��!B@���?O��@����,C�ʳ@��!B@���?O��@����,C��*@��!B@���?O��@����,C�|@��!B@���?O��@����,C���@��!B@���?O��@����,C��@��!B@���?O��@����,C�{�@��!B@���?O��@����,C���@��!B@���?O��@����,C��!@��!B@���?O��@����,C��@��!B@���?O��@����,C��p@��!B@���?O��@����,C�D@��!B@���?O��@����,C��@��!B@���?O��@����,C���@����,@���?O��@��     C��@����,@���?O��@��     C�-k@����,@���?O��@��     C��W@���`X@��     @����,C��@���`X@��     @����,C��G@���`X@��     @����,C�L @��!B@���?O��@����,C�?@��!B@���?O��@����,C�*@��!B@���?O��@����,C�@��!B@���?O��@����,C���@��!B@���?O��@����,C�ބ@��!B@���?O��@����,C�B�@��!B@���?O��@����,C���@��!B@���?O��@����,C�)>@��!B@���?O��@����,C��:@��!B@���?O��@����,C���@��!B@���?O��@����,C���@��!B@���?O��@����,C�K�@��!B@���?O��@����,C��'@��!B@���?O��@����,C�}@��!B@���?O��@����,C�=�@��!B@���?O��@� `XC��N@��!B@���?O��@� `XC�}�@��!B@���?O��@� `XC�_�@��!@���?O��@�`XC�z�@��!@���?O��@�`XC�^T@��!@���?O��@�`XC���@��!@������@�`XC�R@��!@������@�`XC�|d@��!@������@�`XC�M�@��!@������@�`XC��-@��!@������@�`XC���@��!@������@�`XC�\�@��!@������@�`XC�c+@��!@������@�`XC���@��!@������@�`XC���@�	�!@������@�
`XC�&J@�	�!@������@�
`XC���@�	�!@������@�
`XC�E�@��!@�	�����@�`XC�c@��!@�	�����@�`XC��4@��!@�	�����@�`XC���@��!@������@�`XC��;@��!@������@�`XC��@��!@������@�`XC�~�@��!@������@�`XC�?@��!@������@�`XC�1�@��!@������@�`XC��@��!@������@�`XC���@��!@������@�`XC���@��!@������@�`XC���@��!@������@�`XC�f@��!@������@�`XC���@��!@������@�`XC���@��!@������@�`XC��@��!@������@�`XC��`@��!@������@�`XC��a@��!@������@�`XC��@��!@������@�`XC��5@��!@������@�`XC���@��!@������@�`XC�ǖ@��!@������@�`XC�_@��!@������@�`XC��@��!@������@�`XC���@��!@������@�`XC�2#@��!@������@�`XC�
�@��!@������@�`XC�wg@��!@������@�`XC�v�@��!@������@�`XC�a�@��!@������@� `XC��;@��!@������@� `XC��v@��!@������@� `XC���@�!�!@������@�"`XC� �@�!�!@������@�"`XC�2n@�!�!@������@�"`XC�K<@�#�!@�!�����@�$`XC��N@�#�!@�!�����@�$`XC���@�#�!@�!�����@�$`XC�u@@�%�!@�#�����@�&`XC�Ӊ@�%�!@�#�����@�&`XC�̌@�%�!@�#�����@�&`XC��r@�'�!@�%�����@�(`XC�L�@�'�!@�%�����@�(`XC��j@�'�!@�%�����@�(`XC�/�@�)�!@�'�����@�*`XC�o@�)�!@�'�����@�*`XC��^@�)�!@�'�����@�*`XC��@�+�!@�)�����@�,`XC�:@�+�!@�)�����@�,`XC��Z@�+�!@�)�����@�,`XC�gq@�-�!@�+�����@�.`XC��@�-�!@�+�����@�.`XC���@�-�!@�+�����@�.`XC�g�@�/�!@�-�����@�0`XC�P@�/�!@�-�����@�0`XC��:@�/�!@�-�����@�0`XC�5@�1�!@�/�����@�2`XC��t@�1�!@�/�����@�2`XC���@�1�!@�/�����@�2`XC���@�3�!@�1�����@�4`XC�E@�3�!@�1�����@�4`XC��@�3�!@�1�����@�4`XC���@�5�!@�3�����@�6`XC�B�@�5�!@�3�����@�6`XC��L@�5�!@�3�����@�6`XC��@�7`X@�5�����@�8     C��\@�7`X@�5�����@�8     C���@�7`X@�5�����@�8     C�w�@�9��,@�8     @�:`XC���@�9��,@�8     @�:`XC�h~@�9��,@�8     @�:`XC�xi@�;�!@�9�����@�<`XC�?�@�;�!@�9�����@�<`XC��&@�;�!@�9�����@�<`XC���@�=�!@�;�����@�>`XC��&@�=�!@�;�����@�>`XC�S�@�=�!@�;�����@�>`XC���@�?�!@�=�����@�@`XC���@�?�!@�=�����@�@`XC��@�?�!@�=�����@�@`XC�0�@�A�!@�?�����@�B`XC�_u@�A�!@�?�����@�B`XC���@�A�!@�?�����@�B`XC��#@�C�!@�A�����@�D`XC�H�@�C�!@�A�����@�D`XC�4�@�C�!@�A�����@�D`XC���@�E�!@�C�����@�F`XC��@�E�!@�C�����@�F`XC��@�E�!@�C�����@�F`XC�^�@�G�!@�E�����@�H`XC�Н@�G�!@�E�����@�H`XC�m�@�G�!@�E�����@�H`XC���@�I�!@�G�����@�J`XC���@�I�!@�G�����@�J`XC���@�I�!@�G�����@�J`XC��@�K�!@�I�����@�L`XC��!@�K�!@�I�����@�L`XC��@�K�!@�I�����@�L`XC��@�M�!@�K�����@�N`XC�ǒ@�M�!@�K�����@�N`XC���@�M�!@�K�����@�N`XC�o�@�O�!@�M�����@�P`XC�N2@�O�!@�M�����@�P`XC���@�O�!@�M�����@�P`XC��@�Q�!@�O�����@�R`XC�sL@�Q�!@�O�����@�R`XC���@�Q�!@�O�����@�R`XC�Kq@�S�!@�Q�����@�T`XC��Y@�S�!@�Q�����@�T`XC�w@�S�!@�Q�����@�T`XC�Ԕ@�U�!@�S�����@�V`XC���@�U�!@�S�����@�V`XC��@�U�!@�S�����@�V`XC��@�W�!@�U�����@�X`XC��&@�W�!@�U�����@�X`XC�~C@�W�!@�U�����@�X`XC�%@�Y�!@�W�����@�Z`XC���@�Y�!@�W�����@�Z`XC�͊@�Y�!@�W�����@�Z`XC�ŋ@�[�!@�Y�����@�\`XC��#@�[�!@�Y�����@�\`XC�*S@�[�!@�Y�����@�\`XC���@�]�!@�[�����@�^`XC���@�]�!@�[�����@�^`XC��"@�]�!@�[�����@�^`XC�8_@�_�!@�]�����@�``XC�?�@�_�!@�]�����@�``XC��r@�_�!@�]�����@�``XC�s�@�a�!@�_�����@�b`XC��@@�a�!@�_�����@�b`XC�<�@�a�!@�_�����@�b`XC�D@�c�!@�a�����@�d`XC��`@�c�!@�a�����@�d`XC�$�@�c�!@�a�����@�d`XC���@�e�!@�c�����@�f`XC�B�@�e�!@�c�����@�f`XC�|�@�e�!@�c�����@�f`XC���@�g�!@�e�����@�h`XC�%@�g�!@�e�����@�h`XC��@�g�!@�e�����@�h`XC�_�@�i`X@�g�����@�j     C��@�i`X@�g�����@�j     C�W@�i`X@�g�����@�j     C�QW