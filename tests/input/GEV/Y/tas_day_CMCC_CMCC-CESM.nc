CDF   �   
      lon       lat       time       bnds            CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      ;Mon Aug 05 09:49:37 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_CMCC-CESM_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HWJuly19/tas/Y//tas_day_CMCC_CMCC-CESM.nc
Mon Aug 05 09:49:37 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_CMCC-CESM_seldate.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_CMCC-CESM_setref.nc
Mon Aug 05 09:49:37 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_CMCC-CESM_merge.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706//tas_day_CMCC-CESM_seldate.nc
Mon Aug 05 09:49:37 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18500101-18541231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18550101-18591231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18600101-18641231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18650101-18691231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18700101-18741231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18750101-18791231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18800101-18841231_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_historical_r1i1p1_18850101-18891231_tmp_yearmean.nc
Mon Aug 05 09:49:36 2019: cdo yearmean /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_yearmean.nc
Mon Aug 05 09:49:36 2019: cdo -mergetime -selmon,7 -selday,23/25 /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_seltime.nc
Mon Aug 05 09:49:36 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_fldmean.nc
Mon Aug 05 09:49:35 2019: cdo ifthen /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/land_sea_mask.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_var.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_var_mask.nc
Mon Aug 05 09:49:33 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_box.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_var.nc
Mon Aug 05 09:49:32 2019: cdo sellonlatbox,-5,10,42,51 /bdd/CMIP5/main/CMCC/CMCC-CESM/rcp85/day/atmos/day/r1i1p1/latest/tas/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231.nc /data/yrobin//tmp/EUPHEME_HWJuly19_data_tas_Y.sh485706/tas_day_CMCC-CESM_rcp85_r1i1p1_20960101-21001231_tmp_box.nc
Model output postprocessed with Afterburner and CDO (https://code.zmaw.de/projects) 2012-09-19T15:53:03Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.    source        	CMCC-CESM      institution       KCMCC - Centro Euro-Mediterraneo per i Cambiamenti Climatici, Bologna, Italy    institute_id      CMCC   experiment_id         
historical     model_id      	CMCC-CESM      forcing       Nat,Ant,GHG,SA,Oz,Sl   parent_experiment_id      N/A    parent_experiment_rip         N/A    branch_time                  contact       'Marcello Vichi (marcello.vichi@cmcc.it)    comment       �Equilibrium reached after more than 1500-year spin-up of the physics, 200-year spin-up of carbon content and 276 year at pre-industrial GHG concentrations after which data were output with nominal date of January 1850.     
references        Mmodel described in the documentation at http://www.cmcc.it/data-models/models      initialization_method               physics_version             tracking_id       $4a240880-8508-452b-aa57-960674fc7820   product       output     
experiment        
historical     	frequency         day    creation_date         2012-08-01T14:32:02Z   
project_id        CMIP5      table_id      :Table day (27 April 2011) 86d1558d99b6ed1e7a886ab3fd717b58     title         4CMCC-CESM model output prepared for CMIP5 historical   parent_experiment         N/A    modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      standard        �   	time_bnds                            �   tas                    	   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   original_name         temp2      cell_methods      time: mean (interval: 1 day)       history       �2012-08-01T14:32:00Z altered by CMOR: Treated scalar dimension: 'height'. 2012-08-01T14:32:02Z altered by CMOR: Inverted axis: lat.    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CMCC-CESM_historical_r0i0p0.nc areacella: areacella_fx_CMCC-CESM_historical_r0i0p0.nc            �                @��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�s<@��@�,@��<�#��@��D�4MC��#@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�=�@�@�,@�<�#��@�D�4MC�n@�@�,@�<�#��@�D�4MC��@�
@�,@�
<�#��@�
D�4MC���@�@�,@�<�#��@�D�4MC�R�@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��>@�@�,@�<�#��@�D�4MC�:8@�@�,@�<�#��@�D�4MC���@�"@�,@�"<�#��@�"D�4MC���@�&@�,@�&<�#��@�&D�4MC��V@�*@�,@�*<�#��@�*D�4MC�Ų@�.@�,@�.<�#��@�.D�4MC�m@�2@�,@�2<�#��@�2D�4MC��<@�6@�,@�6<�#��@�6D�4MC��e@�:@�,@�:<�#��@�:D�4MC��@�>@�,@�><�#��@�>D�4MC��?@�B@�,@�B<�#��@�BD�4MC��@�F@�,@�F<�#��@�FD�4MC��(@�J@�,@�J<�#��@�JD�4MC��@�N@�,@�N<�#��@�ND�4MC�'i@�R@�,@�R<�#��@�RD�4MC���@�V@�,@�V<�#��@�VD�4MC��}@�Z@�,@�Z<�#��@�ZD�4MC�I�@�^@�,@�^<�#��@�^D�4MC��p@�b@�,@�b<�#��@�bD�4MC��@�f@�,@�f<�#��@�fD�4MC�f~@�j@�,@�j<�#��@�jD�4MC�e@�n@�,@�n<�#��@�nD�4MC��@�r@�,@�r<�#��@�rD�4MC��V@�v@�,@�v<�#��@�vD�4MC�W�@�z@�,@�z<�#��@�zD�4MC�B�@�~@�,@�~<�#��@�~D�4MC��(@��@�,@��<�#��@��D�4MC�n�@��@�,@��<�#��@��D�4MC��\@��@�,@��<�#��@��D�4MC��A@��@�,@��<�#��@��D�4MC�l@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�dT@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��K@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�:@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�"�@��@�,@��<�#��@��D�4MC�(�@��@�,@��<�#��@��D�4MC�Ѳ@��@�,@��<�#��@��D�4MC��;@��@�,@��<�#��@��D�4MC�d�@��@�,@��<�#��@��D�4MC�vF@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�#�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�<^@��@�,@��<�#��@��D�4MC�7>@��@�,@��<�#��@��D�4MC�� @��@�,@��<�#��@��D�4MC�+M@�@�,@�<�#��@�D�4MC��_@�@�,@�<�#��@�D�4MC�Օ@�
@�,@�
<�#��@�
D�4MC�~'@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC��4@�"@�,@�"<�#��@�"D�4MC�G�@�&@�,@�&<�#��@�&D�4MC�/m@�*@�,@�*<�#��@�*D�4MC�^@�.@�,@�.<�#��@�.D�4MC��?@�2@�,@�2<�#��@�2D�4MC��@�6@�,@�6<�#��@�6D�4MC��x@�:@�,@�:<�#��@�:D�4MC��)@�>@�,@�><�#��@�>D�4MC��P@�B@�,@�B<�#��@�BD�4MC�a@�F@�,@�F<�#��@�FD�4MC���@�J@�,@�J<�#��@�JD�4MC�.@�N@�,@�N<�#��@�ND�4MC���@�R@�,@�R<�#��@�RD�4MC�d�@�V@�,@�V<�#��@�VD�4MC�5�@�Z@�,@�Z<�#��@�ZD�4MC�r�@�^@�,@�^<�#��@�^D�4MC���@�b@�,@�b<�#��@�bD�4MC��@�f@�,@�f<�#��@�fD�4MC��q@�j@�,@�j<�#��@�jD�4MC���@�n@�,@�n<�#��@�nD�4MC��@�r@�,@�r<�#��@�rD�4MC���@�v@�,@�v<�#��@�vD�4MC��D@�z@�,@�z<�#��@�zD�4MC���@�~@�,@�~<�#��@�~D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�r�@��@�,@��<�#��@��D�4MC��Z@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�*�@��@�,@��<�#��@��D�4MC�Y�@��@�,@��<�#��@��D�4MC�C�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��9@��@�,@��<�#��@��D�4MC��b@��@�,@��<�#��@��D�4MC�0R@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�v@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�s�@��@�,@��<�#��@��D�4MC�c@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�u@��@�,@��<�#��@��D�4MC�!|@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC�3�@��@�,@��<�#��@��D�4MC��\@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC� �@��@�,@��<�#��@��D�4MC��K@��@�,@��<�#��@��D�4MC��A@��@�,@��<�#��@��D�4MC�"@��@�,@��<�#��@��D�4MC��@�@�,@�<�#��@�D�4MC�KE@�@�,@�<�#��@�D�4MC�h@�
@�,@�
<�#��@�
D�4MC��P@�@�,@�<�#��@�D�4MC���@�@�,@�<�#��@�D�4MC�\�@�@�,@�<�#��@�D�4MC�j�@�@�,@�<�#��@�D�4MC��@�@�,@�<�#��@�D�4MC�զ@�"@�,@�"<�#��@�"D�4MC��X@�&@�,@�&<�#��@�&D�4MC���@�*@�,@�*<�#��@�*D�4MC�\�@�.@�,@�.<�#��@�.D�4MC��@�2@�,@�2<�#��@�2D�4MC�*/@�6@�,@�6<�#��@�6D�4MC�	�@�:@�,@�:<�#��@�:D�4MC��@�>@�,@�><�#��@�>D�4MC���@�B@�,@�B<�#��@�BD�4MC�q�@�F@�,@�F<�#��@�FD�4MC�U�@�J@�,@�J<�#��@�JD�4MC�ω@�N@�,@�N<�#��@�ND�4MC�q@�R@�,@�R<�#��@�RD�4MC�5@�V@�,@�V<�#��@�VD�4MC�`4@�Z@�,@�Z<�#��@�ZD�4MC���@�^@�,@�^<�#��@�^D�4MC��@�b@�,@�b<�#��@�bD�4MC�R@�f@�,@�f<�#��@�fD�4MC��w@�j@�,@�j<�#��@�jD�4MC���@�n@�,@�n<�#��@�nD�4MC���@�r@�,@�r<�#��@�rD�4MC��@�v@�,@�v<�#��@�vD�4MC���@�z@�,@�z<�#��@�zD�4MC�\@�~@�,@�~<�#��@�~D�4MC��o@��@�,@��<�#��@��D�4MC�g4@��@�,@��<�#��@��D�4MC��|@��@�,@��<�#��@��D�4MC�o�@��@�,@��<�#��@��D�4MC�F6@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�Y@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��<@��@�,@��<�#��@��D�4MC��b@��@�,@��<�#��@��D�4MC�@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��^@��@�,@��<�#��@��D�4MC�!�@��@�,@��<�#��@��D�4MC�(�@��@�,@��<�#��@��D�4MC�y(@��@�,@��<�#��@��D�4MC��Z@��@�,@��<�#��@��D�4MC�g�@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC�F@��@�,@��<�#��@��D�4MC��	@��@�,@��<�#��@��D�4MC��k@��@�,@��<�#��@��D�4MC��{@��@�,@��<�#��@��D�4MC��o@��@�,@��<�#��@��D�4MC�k�@��@�,@��<�#��@��D�4MC�B)@��@�,@��<�#��@��D�4MC�vv@��@�,@��<�#��@��D�4MC���@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC�3�@��@�,@��<�#��@��D�4MC��>@��@�,@��<�#��@��D�4MC��@��@�,@��<�#��@��D�4MC��@� X�@�G��y@�"h�&�C���@� X�@�G��y@�"h�&�C�@� X�@�G��y@�"h�&�C���@� X�@�G��y@�"h�&�C��0@�	 X�@�	G��y@�	"h�&�C��@� X�@�G��y@�"h�&�C���@� X�@�G��y@�"h�&�C�4�@� X�@�G��y@�"h�&�C�Q@� X�@�G��y@�"h�&�C��b@� X�@�G��y@�"h�&�C��@� X�@�G��y@�"h�&�C�q{@� X�@�G��y@�"h�&�C�'3@� X�@�G��y@�"h�&�C���@� X�@�G��y@�"h�&�C��I@� X�@�G��y@�"h�&�C�͊@� X�@�G��y@�"h�&�C��@�! X�@�!G��y@�!"h�&�C���@�# X�@�#G��y@�#"h�&�C�ܻ@�% X�@�%G��y@�%"h�&�C���@�' X�@�'G��y@�'"h�&�C���@�) X�@�)G��y@�)"h�&�C��)@�+ X�@�+G��y@�+"h�&�C�<�@�- X�@�-G��y@�-"h�&�C���@�/ X�@�/G��y@�/"h�&�C�\@�1 X�@�1G��y@�1"h�&�C�I�@�3 X�@�3G��y@�3"h�&�C�|�@�5 X�@�5G��y@�5"h�&�C�bX@�7 X�@�7G��y@�7"h�&�C�Ev@�9 X�@�9G��y@�9"h�&�C��@�; X�@�;G��y@�;"h�&�C�AB@�= X�@�=G��y@�="h�&�C�(@�? X�@�?G��y@�?"h�&�C��@�A X�@�AG��y@�A"h�&�C�]@�C X�@�CG��y@�C"h�&�C�@�@�E X�@�EG��y@�E"h�&�C�ֺ@�G X�@�GG��y@�G"h�&�C���@�I X�@�IG��y@�I"h�&�C���@�K X�@�KG��y@�K"h�&�C�F@�M X�@�MG��y@�M"h�&�C�$p@�O X�@�OG��y@�O"h�&�C�b�@�Q X�@�QG��y@�Q"h�&�C���@�S X�@�SG��y@�S"h�&�C��@�U X�@�UG��y@�U"h�&�C��;@�W X�@�WG��y@�W"h�&�C��@�Y X�@�YG��y@�Y"h�&�C��@�[ X�@�[G��y@�["h�&�C��'@�] X�@�]G��y@�]"h�&�C�k@�_ X�@�_G��y@�_"h�&�C��1@�a X�@�aG��y@�a"h�&�C��,@�c X�@�cG��y@�c"h�&�C�~�@�e X�@�eG��y@�e"h�&�C��^@�g X�@�gG��y@�g"h�&�C��@�i X�@�iG��y@�i"h�&�C�@8