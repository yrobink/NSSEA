CDF  3   
      lon       lat       time       bnds            CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      Tue Aug 06 07:59:23 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016//tas_day_IPSL-CM5A-MR_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HWJuly19/tas/Y//tas_day_IPSL_IPSL-CM5A-MR.nc
Tue Aug 06 07:59:23 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016//tas_day_IPSL-CM5A-MR_seldate.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016//tas_day_IPSL-CM5A-MR_setref.nc
Tue Aug 06 07:59:23 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016//tas_day_IPSL-CM5A-MR_merge.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016//tas_day_IPSL-CM5A-MR_seldate.nc
Tue Aug 06 07:59:22 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r1i1p1_18500101-18991231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r1i1p1_19000101-19491231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r1i1p1_19500101-19991231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r1i1p1_20000101-20051231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r2i1p1_18500101-18991231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r2i1p1_19000101-19491231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r2i1p1_19500101-19991231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_historical_r2i1p1_20000101-20051231_tmp_yearmean.nc
Tue Aug 06 07:59:20 2019: cdo yearmean /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_yearmean.nc
Tue Aug 06 07:59:19 2019: cdo -mergetime -selmon,7 -selday,23/25 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_seltime.nc
Tue Aug 06 07:59:18 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_fldmean.nc
Tue Aug 06 07:59:18 2019: cdo ifthen /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/land_sea_mask.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_var_mask.nc
Tue Aug 06 07:59:14 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_box.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_var.nc
Tue Aug 06 07:59:06 2019: cdo sellonlatbox,-5,10,42,51 /bdd/CMIP5/main/IPSL/IPSL-CM5A-MR/rcp85/day/atmos/day/r1i1p1/latest/tas/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas.sh486016/tas_day_IPSL-CM5A-MR_rcp85_r1i1p1_20560101-21001231_tmp_box.nc
2011-11-05T13:02:02Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �IPSL-CM5A-MR (2010) : atmos : LMDZ4 (LMDZ4_v5, 144x143x39); ocean : ORCA2 (NEMOV2_3, 2x2L31); seaIce : LIM2 (NEMOV2_3); ocnBgchem : PISCES (NEMOV2_3); land : ORCHIDEE (orchidee_1_9_4_AR5)    institution       3IPSL (Institut Pierre Simon Laplace, Paris, France)    institute_id      IPSL   experiment_id         
historical     model_id      IPSL-CM5A-MR   forcing       &Nat,Ant,GHG,SA,Oz,LU,SS,Ds,BC,MD,OC,AA     parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @��        contact       ?ipsl-cmip5 _at_ ipsl.jussieu.fr Data manager : Sebastien Denvil    comment       HThis 20th century simulation include natural and anthropogenic forcings.   
references        NModel documentation and further reference available here : http://icmc.ipsl.fr     initialization_method               physics_version             tracking_id       $3a94d490-43bb-4ecf-824a-ef8d51411fe7   product       output     
experiment        
historical     	frequency         day    creation_date         2011-09-23T00:57:21Z   
project_id        CMIP5      table_id      =Table day (10 February 2011) 80e409bd73611e9d25d049ad2059c310      title         7IPSL-CM5A-MR model output prepared for CMIP5 historical    parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           <   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           D   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      365_day         L   	time_bnds                            T   tas                    	   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   original_name         t2m    cell_methods      "time: mean (interval: 30 minutes)      history       �2011-09-23T00:57:10Z altered by CMOR: Treated scalar dimension: 'height'. 2011-09-23T00:57:10Z altered by CMOR: replaced missing value flag (9.96921e+36) with standard missing value (1e+20). 2011-09-23T00:57:21Z altered by CMOR: Inverted axis: lat.       associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_IPSL-CM5A-MR_historical_r0i0p0.nc areacella: areacella_fx_IPSL-CM5A-MR_historical_r0i0p0.nc          d                @��@�,@��<�#��@��D�4MC�6�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�#�@��@�,@��<�#��@��D�4MC�(@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��+@��@�,@��<�#��@��D�4MC�Q@��@�,@��<�#��@��D�4MC��6@��@�,@��<�#��@��D�4MC�'�@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC�?@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�'@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�D�@�@�,@�<�#��@�D�4MC�۲@�@�,@�<�#��@�D�4MC�}1@�@�,@�<�#��@�D�4MC�f�@�@�,@�<�#��@�D�4MC�>�@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC���@�
@�,@�
<�#��@�
D�4MC�:@�
@�,@�
<�#��@�
D�4MC�9�@�
@�,@�
<�#��@�
D�4MC�r@�@�,@�<�#��@�D�4MC��[@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�{@�@�,@�<�#��@�D�4MC�v@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��s@�@�,@�<�#��@�D�4MC�^(@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC�>@�@�,@�<�#��@�D�4MC�v�@�@�,@�<�#��@�D�4MC��>@�@�,@�<�#��@�D�4MC�H�@�@�,@�<�#��@�D�4MC�7�@�@�,@�<�#��@�D�4MC��@�"@�,@�"<�#��@�"D�4MC�"P@�"@�,@�"<�#��@�"D�4MC��X@�"@�,@�"<�#��@�"D�4MC�Z�@�&@�,@�&<�#��@�&D�4MC�S@@�&@�,@�&<�#��@�&D�4MC���@�&@�,@�&<�#��@�&D�4MC���@�*@�,@�*<�#��@�*D�4MC�6�@�*@�,@�*<�#��@�*D�4MC���@�*@�,@�*<�#��@�*D�4MC���@�.@�,@�.<�#��@�.D�4MC�4�@�.@�,@�.<�#��@�.D�4MC���@�.@�,@�.<�#��@�.D�4MC�C`@�2@�,@�2<�#��@�2D�4MC�lV@�2@�,@�2<�#��@�2D�4MC���@�2@�,@�2<�#��@�2D�4MC��@�6@�,@�6<�#��@�6D�4MC�:f@�6@�,@�6<�#��@�6D�4MC��@�6@�,@�6<�#��@�6D�4MC�e�@�:@�,@�:<�#��@�:D�4MC��5@�:@�,@�:<�#��@�:D�4MC�4@�:@�,@�:<�#��@�:D�4MC�@�>@�,@�><�#��@�>D�4MC��*@�>@�,@�><�#��@�>D�4MC��@�>@�,@�><�#��@�>D�4MC�\@�B@�,@�B<�#��@�BD�4MC�(�@�B@�,@�B<�#��@�BD�4MC�!!@�B@�,@�B<�#��@�BD�4MC�zy@�F@�,@�F<�#��@�FD�4MC�{z@�F@�,@�F<�#��@�FD�4MC��@�F@�,@�F<�#��@�FD�4MC� V@�J@�,@�J<�#��@�JD�4MC��	@�J@�,@�J<�#��@�JD�4MC�Ҧ@�J@�,@�J<�#��@�JD�4MC��|@�N@�,@�N<�#��@�ND�4MC��	@�N@�,@�N<�#��@�ND�4MC�<@�N@�,@�N<�#��@�ND�4MC��i@�R@�,@�R<�#��@�RD�4MC���@�R@�,@�R<�#��@�RD�4MC�J@�R@�,@�R<�#��@�RD�4MC�v4@�V@�,@�V<�#��@�VD�4MC�x@�V@�,@�V<�#��@�VD�4MC��@�V@�,@�V<�#��@�VD�4MC��@�Z@�,@�Z<�#��@�ZD�4MC�
@�Z@�,@�Z<�#��@�ZD�4MC��=@�Z@�,@�Z<�#��@�ZD�4MC���@�^@�,@�^<�#��@�^D�4MC��@�^@�,@�^<�#��@�^D�4MC�9@�^@�,@�^<�#��@�^D�4MC�$)@�b@�,@�b<�#��@�bD�4MC��n@�b@�,@�b<�#��@�bD�4MC��,@�b@�,@�b<�#��@�bD�4MC�y�@�f@�,@�f<�#��@�fD�4MC�ѷ@�f@�,@�f<�#��@�fD�4MC�kZ@�f@�,@�f<�#��@�fD�4MC��>@�j@�,@�j<�#��@�jD�4MC��@�j@�,@�j<�#��@�jD�4MC�'@�j@�,@�j<�#��@�jD�4MC���@�n@�,@�n<�#��@�nD�4MC�q�@�n@�,@�n<�#��@�nD�4MC��@�n@�,@�n<�#��@�nD�4MC���@�r@�,@�r<�#��@�rD�4MC�G@�r@�,@�r<�#��@�rD�4MC��@�r@�,@�r<�#��@�rD�4MC�?@�v@�,@�v<�#��@�vD�4MC�KM@�v@�,@�v<�#��@�vD�4MC��L@�v@�,@�v<�#��@�vD�4MC��@�z@�,@�z<�#��@�zD�4MC�v+@�z@�,@�z<�#��@�zD�4MC���@�z@�,@�z<�#��@�zD�4MC��@�~@�,@�~<�#��@�~D�4MC�d@�~@�,@�~<�#��@�~D�4MC��V@�~@�,@�~<�#��@�~D�4MC�f�@��@�,@��<�#��@��D�4MC�>�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�W�@��@�,@��<�#��@��D�4MC�
�@��@�,@��<�#��@��D�4MC�
�@��@�,@��<�#��@��D�4MC��q@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�%�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�^�@��@�,@��<�#��@��D�4MC��-@��@�,@��<�#��@��D�4MC�r @��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��l@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�p�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�g&@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�	@��@�,@��<�#��@��D�4MC�k�@��@�,@��<�#��@��D�4MC�{}@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC�gR@��@�,@��<�#��@��D�4MC��g@��@�,@��<�#��@��D�4MC�"@��@�,@��<�#��@��D�4MC��$@��@�,@��<�#��@��D�4MC�Lj@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�f�@��@�,@��<�#��@��D�4MC�>�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�"}@��@�,@��<�#��@��D�4MC��&@��@�,@��<�#��@��D�4MC�9�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�q@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�ӊ@��@�,@��<�#��@��D�4MC��O@��@�,@��<�#��@��D�4MC��&@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��~@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�դ@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�Pr@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�	�@��@�,@��<�#��@��D�4MC�/@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��_@��@�,@��<�#��@��D�4MC��K@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��1@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�ِ@��@�,@��<�#��@��D�4MC�J@��@�,@��<�#��@��D�4MC��4@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�o@��@�,@��<�#��@��D�4MC��2@��@�,@��<�#��@��D�4MC�=�@��@�,@��<�#��@��D�4MC�G@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�6
@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��~@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�f@��@�,@��<�#��@��D�4MC�j�@��@�,@��<�#��@��D�4MC�_�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�P@��@�,@��<�#��@��D�4MC�P@��@�,@��<�#��@��D�4MC�$;@��@�,@��<�#��@��D�4MC� �@��@�,@��<�#��@��D�4MC��X@��@�,@��<�#��@��D�4MC�6�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�M�@��@�,@��<�#��@��D�4MC� �@�@�,@�<�#��@�D�4MC�%%@�@�,@�<�#��@�D�4MC��{@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC��h@�@�,@�<�#��@�D�4MC�%K@�@�,@�<�#��@�D�4MC���@�
@�,@�
<�#��@�
D�4MC�Ia@�
@�,@�
<�#��@�
D�4MC��@�
@�,@�
<�#��@�
D�4MC��!@�@�,@�<�#��@�D�4MC�@�@�,@�<�#��@�D�4MC��u@�@�,@�<�#��@�D�4MC�2�@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��j@�@�,@�<�#��@�D�4MC�{�@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�=@�@�,@�<�#��@�D�4MC��z@�@�,@�<�#��@�D�4MC�u"@�@�,@�<�#��@�D�4MC��e@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�=@�@�,@�<�#��@�D�4MC�:�@�"@�,@�"<�#��@�"D�4MC�Q�@�"@�,@�"<�#��@�"D�4MC��@�"@�,@�"<�#��@�"D�4MC�%�@�&@�,@�&<�#��@�&D�4MC���@�&@�,@�&<�#��@�&D�4MC�r�@�&@�,@�&<�#��@�&D�4MC��>@�*@�,@�*<�#��@�*D�4MC�1�@�*@�,@�*<�#��@�*D�4MC���@�*@�,@�*<�#��@�*D�4MC�]�@�.@�,@�.<�#��@�.D�4MC�O,@�.@�,@�.<�#��@�.D�4MC�we@�.@�,@�.<�#��@�.D�4MC��S@�2@�,@�2<�#��@�2D�4MC��+@�2@�,@�2<�#��@�2D�4MC��@�2@�,@�2<�#��@�2D�4MC��N@�6@�,@�6<�#��@�6D�4MC�3>@�6@�,@�6<�#��@�6D�4MC��1@�6@�,@�6<�#��@�6D�4MC���@�:@�,@�:<�#��@�:D�4MC�#J@�:@�,@�:<�#��@�:D�4MC��N@�:@�,@�:<�#��@�:D�4MC��@�>@�,@�><�#��@�>D�4MC��@�>@�,@�><�#��@�>D�4MC���@�>@�,@�><�#��@�>D�4MC��3@�B@�,@�B<�#��@�BD�4MC�:u@�B@�,@�B<�#��@�BD�4MC�(�@�B@�,@�B<�#��@�BD�4MC�+[@�F@�,@�F<�#��@�FD�4MC��@�F@�,@�F<�#��@�FD�4MC�:p@�F@�,@�F<�#��@�FD�4MC�� @�J@�,@�J<�#��@�JD�4MC��r@�J@�,@�J<�#��@�JD�4MC�W�@�J@�,@�J<�#��@�JD�4MC��@�N@�,@�N<�#��@�ND�4MC��@�N@�,@�N<�#��@�ND�4MC���@�N@�,@�N<�#��@�ND�4MC��=@�R@�,@�R<�#��@�RD�4MC�z@�R@�,@�R<�#��@�RD�4MC�A@�R@�,@�R<�#��@�RD�4MC�Z@@�V@�,@�V<�#��@�VD�4MC��@�V@�,@�V<�#��@�VD�4MC���@�V@�,@�V<�#��@�VD�4MC�*k@�Z@�,@�Z<�#��@�ZD�4MC��]@�Z@�,@�Z<�#��@�ZD�4MC��@�Z@�,@�Z<�#��@�ZD�4MC�g�@�^@�,@�^<�#��@�^D�4MC�6�@�^@�,@�^<�#��@�^D�4MC���@�^@�,@�^<�#��@�^D�4MC��"@�b@�,@�b<�#��@�bD�4MC��@�b@�,@�b<�#��@�bD�4MC���@�b@�,@�b<�#��@�bD�4MC�2~@�f@�,@�f<�#��@�fD�4MC���@�f@�,@�f<�#��@�fD�4MC���@�f@�,@�f<�#��@�fD�4MC�z�@�j@�,@�j<�#��@�jD�4MC���@�j@�,@�j<�#��@�jD�4MC��@�j@�,@�j<�#��@�jD�4MC�ӵ@�n@�,@�n<�#��@�nD�4MC�O@�n@�,@�n<�#��@�nD�4MC���@�n@�,@�n<�#��@�nD�4MC�1@�r@�,@�r<�#��@�rD�4MC�d
@�r@�,@�r<�#��@�rD�4MC�FK@�r@�,@�r<�#��@�rD�4MC��.@�v@�,@�v<�#��@�vD�4MC��e@�v@�,@�v<�#��@�vD�4MC�$@�v@�,@�v<�#��@�vD�4MC�B�@�z@�,@�z<�#��@�zD�4MC�m�@�z@�,@�z<�#��@�zD�4MC���@�z@�,@�z<�#��@�zD�4MC��@�~@�,@�~<�#��@�~D�4MC���@�~@�,@�~<�#��@�~D�4MC�M;@�~@�,@�~<�#��@�~D�4MC��@��@�,@��<�#��@��D�4MC�:�@��@�,@��<�#��@��D�4MC�^@��@�,@��<�#��@��D�4MC�{A@��@�,@��<�#��@��D�4MC�_�@��@�,@��<�#��@��D�4MC�
@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�:)@��@�,@��<�#��@��D�4MC�_�@��@�,@��<�#��@��D�4MC�O�@��@�,@��<�#��@��D�4MC��P@��@�,@��<�#��@��D�4MC�ƿ@��@�,@��<�#��@��D�4MC�D`@��@�,@��<�#��@��D�4MC�D�@��@�,@��<�#��@��D�4MC�v�@��@�,@��<�#��@��D�4MC�I]@��@�,@��<�#��@��D�4MC�d�@��@�,@��<�#��@��D�4MC�/�@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��c@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�9Z@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�f`@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�
H@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��m@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�2�@��@�,@��<�#��@��D�4MC�m�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�6�@��@�,@��<�#��@��D�4MC�7�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��u@��@�,@��<�#��@��D�4MC�|�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC�N@��@�,@��<�#��@��D�4MC�E\@��@�,@��<�#��@��D�4MC�[!@��@�,@��<�#��@��D�4MC�{�@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC�L@��@�,@��<�#��@��D�4MC�C3@��@�,@��<�#��@��D�4MC��l@��@�,@��<�#��@��D�4MC�&�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�B�@��@�,@��<�#��@��D�4MC��3@��@�,@��<�#��@��D�4MC�Yk@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��J@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��K@��@�,@��<�#��@��D�4MC�0�@��@�,@��<�#��@��D�4MC�E@��@�,@��<�#��@��D�4MC��Q@��@�,@��<�#��@��D�4MC�O;@��@�,@��<�#��@��D�4MC�(@��@�,@��<�#��@��D�4MC��,@��@�,@��<�#��@��D�4MC�ݮ@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�}�@��@�,@��<�#��@��D�4MC�3�@��@�,@��<�#��@��D�4MC�=�@��@�,@��<�#��@��D�4MC�
<@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��>@��@�,@��<�#��@��D�4MC��P@��@�,@��<�#��@��D�4MC�n@��@�,@��<�#��@��D�4MC��T@��@�,@��<�#��@��D�4MC�Ne@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�JM@��@�,@��<�#��@��D�4MC�v	@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��}@��@�,@��<�#��@��D�4MC�Z�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�>�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�>v@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�=}@��@�,@��<�#��@��D�4MC��;@��@�,@��<�#��@��D�4MC�5�@��@�,@��<�#��@��D�4MC��
@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@�@�,@�<�#��@�D�4MC�i]@�@�,@�<�#��@�D�4MC�Z!@�@�,@�<�#��@�D�4MC�Z�@�@�,@�<�#��@�D�4MC�3�@�@�,@�<�#��@�D�4MC�+@�@�,@�<�#��@�D�4MC�X�@�
@�,@�
<�#��@�
D�4MC���@�
@�,@�
<�#��@�
D�4MC��@�
@�,@�
<�#��@�
D�4MC��@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC�o�@�@�,@�<�#��@�D�4MC�,�@�@�,@�<�#��@�D�4MC�]�@�@�,@�<�#��@�D�4MC��}@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC��1@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�-;@�@�,@�<�#��@�D�4MC�ҙ@�@�,@�<�#��@�D�4MC�,{@�@�,@�<�#��@�D�4MC�_�@�@�,@�<�#��@�D�4MC��-@�@�,@�<�#��@�D�4MC�o#@�"@�,@�"<�#��@�"D�4MC��%@�"@�,@�"<�#��@�"D�4MC���@�"@�,@�"<�#��@�"D�4MC���@�&@�,@�&<�#��@�&D�4MC���@�&@�,@�&<�#��@�&D�4MC�^�@�&@�,@�&<�#��@�&D�4MC��@�*@�,@�*<�#��@�*D�4MC���@�*@�,@�*<�#��@�*D�4MC��T@�*@�,@�*<�#��@�*D�4MC��@�.@�,@�.<�#��@�.D�4MC��@�.@�,@�.<�#��@�.D�4MC��e@�.@�,@�.<�#��@�.D�4MC��N@�2@�,@�2<�#��@�2D�4MC��3@�2@�,@�2<�#��@�2D�4MC��@�2@�,@�2<�#��@�2D�4MC���@�6@�,@�6<�#��@�6D�4MC�t�@�6@�,@�6<�#��@�6D�4MC�K@�6@�,@�6<�#��@�6D�4MC���@�:@�,@�:<�#��@�:D�4MC�w�@�:@�,@�:<�#��@�:D�4MC��!@�:@�,@�:<�#��@�:D�4MC���@�>@�,@�><�#��@�>D�4MC��@�>@�,@�><�#��@�>D�4MC��@�>@�,@�><�#��@�>D�4MC�^@�B@�,@�B<�#��@�BD�4MC�Ro@�B@�,@�B<�#��@�BD�4MC���@�B@�,@�B<�#��@�BD�4MC�*@�F@�,@�F<�#��@�FD�4MC�E$@�F@�,@�F<�#��@�FD�4MC�2^@�F@�,@�F<�#��@�FD�4MC�V�@�J@�,@�J<�#��@�JD�4MC�k@�J@�,@�J<�#��@�JD�4MC�j@�J@�,@�J<�#��@�JD�4MC��@�N@�,@�N<�#��@�ND�4MC�Y@�N@�,@�N<�#��@�ND�4MC��@�N@�,@�N<�#��@�ND�4MC��L@�R@�,@�R<�#��@�RD�4MC��w@�R@�,@�R<�#��@�RD�4MC��@�R@�,@�R<�#��@�RD�4MC�Dw@�V@�,@�V<�#��@�VD�4MC�
%@�V@�,@�V<�#��@�VD�4MC�W�@�V@�,@�V<�#��@�VD�4MC�l@�Z@�,@�Z<�#��@�ZD�4MC���@�^@�,@�^<�#��@�^D�4MC�ǵ@�b@�,@�b<�#��@�bD�4MC��@�f@�,@�f<�#��@�fD�4MC���@�j@�,@�j<�#��@�jD�4MC�p@�n@�,@�n<�#��@�nD�4MC���@�r@�,@�r<�#��@�rD�4MC���@�v@�,@�v<�#��@�vD�4MC�8�@�z@�,@�z<�#��@�zD�4MC�W1@�~@�,@�~<�#��@�~D�4MC�)�@��@�,@��<�#��@��D�4MC�s�@��@�,@��<�#��@��D�4MC�k�@��@�,@��<�#��@��D�4MC�$@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�]Y@��@�,@��<�#��@��D�4MC�"�@��@�,@��<�#��@��D�4MC�2@@��@�,@��<�#��@��D�4MC�Qi@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��G@��@�,@��<�#��@��D�4MC�G�@��@�,@��<�#��@��D�4MC�AQ@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��o@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�8Q@��@�,@��<�#��@��D�4MC�&*@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�VV@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�2�@��@�,@��<�#��@��D�4MC��/@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�wD@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�a�@��@�,@��<�#��@��D�4MC�H0@��@�,@��<�#��@��D�4MC��L@��@�,@��<�#��@��D�4MC��@� X�@�G��y@�"h�&�C�Zb@� X�@�G��y@�"h�&�C�Q?@� X�@�G��y@�"h�&�C�2�@� X�@�G��y@�"h�&�C���@�	 X�@�	G��y@�	"h�&�C��t@� X�@�G��y@�"h�&�C��@� X�@�G��y@�"h�&�C��]@� X�@�G��y@�"h�&�C��U@� X�@�G��y@�"h�&�C��@� X�@�G��y@�"h�&�C��F@� X�@�G��y@�"h�&�C�]�@� X�@�G��y@�"h�&�C��[@� X�@�G��y@�"h�&�C���@� X�@�G��y@�"h�&�C���@� X�@�G��y@�"h�&�C��s@� X�@�G��y@�"h�&�C��4@�! X�@�!G��y@�!"h�&�C��6@�# X�@�#G��y@�#"h�&�C�;�@�% X�@�%G��y@�%"h�&�C���@�' X�@�'G��y@�'"h�&�C�/�@�) X�@�)G��y@�)"h�&�C��@�+ X�@�+G��y@�+"h�&�C���@�- X�@�-G��y@�-"h�&�C�@�/ X�@�/G��y@�/"h�&�C���@�1 X�@�1G��y@�1"h�&�C���@�3 X�@�3G��y@�3"h�&�C���@�5 X�@�5G��y@�5"h�&�C��@�7 X�@�7G��y@�7"h�&�C��@�9 X�@�9G��y@�9"h�&�C��w@�; X�@�;G��y@�;"h�&�C�֛@�= X�@�=G��y@�="h�&�C���@�? X�@�?G��y@�?"h�&�C��@�A X�@�AG��y@�A"h�&�C�0@�C X�@�CG��y@�C"h�&�C�e�@�E X�@�EG��y@�E"h�&�C���@�G X�@�GG��y@�G"h�&�C���@�I X�@�IG��y@�I"h�&�C���@�K X�@�KG��y@�K"h�&�C�!@�M X�@�MG��y@�M"h�&�C���@�O X�@�OG��y@�O"h�&�C��N@�Q X�@�QG��y@�Q"h�&�C�<�@�S X�@�SG��y@�S"h�&�C��f@�U X�@�UG��y@�U"h�&�C�j-@�W X�@�WG��y@�W"h�&�C�p=@�Y X�@�YG��y@�Y"h�&�C�#�@�[ X�@�[G��y@�["h�&�C�!�@�] X�@�]G��y@�]"h�&�C��f@�_ X�@�_G��y@�_"h�&�C�� @�a X�@�aG��y@�a"h�&�C��;@�c X�@�cG��y@�c"h�&�C��u@�e X�@�eG��y@�e"h�&�C�A�@�g X�@�gG��y@�g"h�&�C�/�@�i X�@�iG��y@�i"h�&�C�k�