CDF  �   
      lon       lat       time       bnds            CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sat Oct 19 17:50:40 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_EC-EARTH_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HW03/rcp85/Y//tas_mon_ICHEC_EC-EARTH.nc
Sat Oct 19 17:50:40 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_EC-EARTH_seldate.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_EC-EARTH_setref.nc
Sat Oct 19 17:50:40 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_EC-EARTH_merge.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_EC-EARTH_seldate.nc
Sat Oct 19 17:50:40 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_185001-185912_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_186001-186912_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_187001-187912_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_188001-188912_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_189001-189912_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_190001-190912_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_191001-191912_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_historical_r14i1p1_192001-192912_tmp_yearmean.nc
Sat Oct 19 17:50:38 2019: cdo yearmean /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_yearmean.nc
Sat Oct 19 17:50:38 2019: cdo selseas,JJA /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_seltime.nc
Sat Oct 19 17:50:38 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_var.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_fldmean.nc
Sat Oct 19 17:50:38 2019: cdo ifthen /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/land_sea_mask.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_var.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_var_mask.nc
Sat Oct 19 17:50:36 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_box.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_var.nc
Sat Oct 19 17:50:35 2019: cdo sellonlatbox,-5,10,42,51 /bdd/CMIP5/main/ICHEC/EC-EARTH/rcp85/mon/atmos/Amon/r8i1p1/latest/tas/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_EC-EARTH_rcp85_r8i1p1_205101-210012_tmp_box.nc
ISAC-CNR RCP8.5 run. 2012-01-02T12:49:31Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �EC-EARTH 2.3 (2011); atmosphere: IFS (cy31R1+modifications, T159L62); ocean: NEMO (version2+modifications, ORCA1-42lev); sea ice: LIM2; land: HTessel      institution       &EC-Earth (European Earth System Model)     institute_id      ICHEC      experiment_id         
historical     model_id      EC-EARTH   forcing       Nat,Ant    parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @�X        contact       0Alastair McKinstry <alastair.mckinstry@ichec.ie>   comment       �Equilibrium reached after preindustrial spin-up after which data were output starting with nominal date of January 1850 \n  Met Eireann ensemble run, Local contact Emily Gleeson <emily.gleeson@met.ie>   
references        uModel described by Hazeleger et al. (Bull. Amer. Meteor. Soc., 2010, 91, 1357-1363). Also see http://ecearth.knmi.nl.      initialization_method               physics_version             tracking_id       $7dd2d38a-2163-4183-9829-2b045243c0db   product       output     
experiment        
historical     	frequency         mon    creation_date         2013-01-04T12:58:07Z   
project_id        CMIP5      table_id      :Table Amon (26 July 2011) b26379e76858ab98b927917878a63d01     title         3EC-EARTH model output prepared for CMIP5 historical    parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.8.0      nco_openmp_thread_number            CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      standard        �   	time_bnds                            �   tas                    	   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   original_name         2T     cell_methods      time: mean (interval: 3 hours)     history      92013-01-04T12:58:07Z altered by CMOR: Treated scalar dimension: 'height'. 2013-01-04T12:58:07Z altered by CMOR: replaced missing value flag (1e+28) with standard missing value (1e+20). 2013-01-04T12:58:07Z altered by CMOR: Converted type from 'd' to 'f'. 2013-01-04T12:58:07Z altered by CMOR: Inverted axis: lat.       associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_EC-EARTH_historical_r0i0p0.nc areacella: areacella_fx_EC-EARTH_historical_r0i0p0.nc          �                @��*����@�骪���@�ꪪ���C�.%@��*����@�骪���@�ꪪ���C���@��*����@�骪���@�ꪪ���C�j�@��*����@�骪���@�ꪪ���C���@��*����@�骪���@�ꪪ���C��@��*����@�������@����C�/:@��*����@�������@����C��b@��*����@�������@����C��1@��*����@�������@����C��w@��*����@�������@����C��W@��*����@�񪪪��@�򪪪��C��@��*����@�񪪪��@�򪪪��C�P�@��*����@�񪪪��@�򪪪��C��@��*����@�񪪪��@�򪪪��C��@��*����@�񪪪��@�򪪪��C�|�@��*����@�������@�������C�а@��*����@�������@�������C���@��*����@�������@�������C��i@��*����@�������@�������C�v�@��*����@�������@�������C��@��*����@�������@�������C��k@��*����@�������@�������C��@��*����@�������@�������C�&�@��*����@�������@�������C�f>@��*����@�������@�������C�� @��*����@�������@�������C��U@��*����@�������@�������C�+@��*����@�������@�������C�F�@��*����@�������@�������C�G@��*����@�������@�������C�m|@�*����@������@������C��@�*����@������@������C� =@�*����@������@������C��L@�*����@������@������C��@�*����@������@������C�,�@�*����@������@������C�׈@�*����@������@������C��@�*����@������@������C��@�*����@������@������C�U@�*����@������@������C�&�@�
*����@�	�����@�
�����C�o�@�
*����@�	�����@�
�����C���@�
*����@�	�����@�
�����C��g@�
*����@�	�����@�
�����C�ّ@�
*����@�	�����@�
�����C���@�*����@������@������C�©@�*����@������@������C���@�*����@������@������C�sx@�*����@������@������C�/�@�*����@������@������C��@�*����@������@������C�1�@�*����@������@������C�c�@�*����@������@������C��@�*����@������@������C�U�@�*����@������@������C���@�*����@������@������C��*@�*����@������@������C��@�*����@������@������C���@�*����@������@������C���@�*����@������@������C�<@�*����@������@������C��@�*����@������@������C�W@�*����@������@������C��@�*����@������@������C�I@�*����@������@������C�F^@�*����@������@������C�T@�*����@������@������C��m@�*����@������@������C���@�*����@������@������C��b@�*����@������@������C���@�"*����@�!�����@�"�����C��@�"*����@�!�����@�"�����C�K�@�"*����@�!�����@�"�����C��[@�"*����@�!�����@�"�����C�ڑ@�"*����@�!�����@�"�����C���@�&*����@�%�����@�&�����C���@�&*����@�%�����@�&�����C��#@�&*����@�%�����@�&�����C�V@�&*����@�%�����@�&�����C��@�&*����@�%�����@�&�����C���@�**����@�)�����@�*�����C�*�@�**����@�)�����@�*�����C� �@�**����@�)�����@�*�����C��@�**����@�)�����@�*�����C���@�**����@�)�����@�*�����C���@�.*����@�-�����@�.�����C�#&@�.*����@�-�����@�.�����C�w�@�.*����@�-�����@�.�����C���@�.*����@�-�����@�.�����C�	�@�.*����@�-�����@�.�����C��@�2*����@�1�����@�2�����C��7@�2*����@�1�����@�2�����C�'�@�2*����@�1�����@�2�����C�͚@�2*����@�1�����@�2�����C��@�2*����@�1�����@�2�����C��}@�6*����@�5�����@�6�����C��@�6*����@�5�����@�6�����C�w�@�6*����@�5�����@�6�����C���@�6*����@�5�����@�6�����C���@�6*����@�5�����@�6�����C��@�:*����@�9�����@�:�����C��@�:*����@�9�����@�:�����C��;@�:*����@�9�����@�:�����C�E�@�:*����@�9�����@�:�����C���@�:*����@�9�����@�:�����C���@�>*����@�=�����@�>�����C���@�>*����@�=�����@�>�����C���@�>*����@�=�����@�>�����C�`@�>*����@�=�����@�>�����C��u@�>*����@�=�����@�>�����C���@�B*����@�A�����@�B�����C�̅@�B*����@�A�����@�B�����C�F�@�B*����@�A�����@�B�����C��@�B*����@�A�����@�B�����C��@�B*����@�A�����@�B�����C���@�F*����@�E�����@�F�����C��i@�F*����@�E�����@�F�����C��@�F*����@�E�����@�F�����C�U�@�F*����@�E�����@�F�����C�Ø@�F*����@�E�����@�F�����C���@�J*����@�I�����@�J�����C�@�J*����@�I�����@�J�����C��@�J*����@�I�����@�J�����C���@�J*����@�I�����@�J�����C�]@�J*����@�I�����@�J�����C�U�@�N*����@�M�����@�N�����C�`�@�N*����@�M�����@�N�����C��@�N*����@�M�����@�N�����C���@�N*����@�M�����@�N�����C�U@�N*����@�M�����@�N�����C�� @�R*����@�Q�����@�R�����C��f@�R*����@�Q�����@�R�����C�uj@�R*����@�Q�����@�R�����C��g@�R*����@�Q�����@�R�����C���@�R*����@�Q�����@�R�����C� 	@�V*����@�U�����@�V�����C��+@�V*����@�U�����@�V�����C��@�V*����@�U�����@�V�����C��{@�V*����@�U�����@�V�����C��@�V*����@�U�����@�V�����C�=`@�Z*����@�Y�����@�Z�����C��@�Z*����@�Y�����@�Z�����C�=@�Z*����@�Y�����@�Z�����C���@�Z*����@�Y�����@�Z�����C�D)@�Z*����@�Y�����@�Z�����C�̾@�^*����@�]�����@�^�����C�w@�^*����@�]�����@�^�����C���@�^*����@�]�����@�^�����C���@�^*����@�]�����@�^�����C�� @�^*����@�]�����@�^�����C�k�@�b*����@�a�����@�b�����C��k@�b*����@�a�����@�b�����C�$l@�b*����@�a�����@�b�����C�d(@�b*����@�a�����@�b�����C�C�@�b*����@�a�����@�b�����C��@�f*����@�e�����@�f�����C��-@�f*����@�e�����@�f�����C���@�f*����@�e�����@�f�����C�k@�f*����@�e�����@�f�����C�2�@�f*����@�e�����@�f�����C��.@�j*����@�i�����@�j�����C���@�j*����@�i�����@�j�����C�	P@�j*����@�i�����@�j�����C��@�j*����@�i�����@�j�����C��@�j*����@�i�����@�j�����C���@�n*����@�m�����@�n�����C� �@�n*����@�m�����@�n�����C��@�n*����@�m�����@�n�����C���@�n*����@�m�����@�n�����C�#E@�n*����@�m�����@�n�����C��K@�r*����@�q�����@�r�����C��@�r*����@�q�����@�r�����C���@�r*����@�q�����@�r�����C�xB@�r*����@�q�����@�r�����C�&9@�r*����@�q�����@�r�����C�W�@�v*����@�u�����@�v�����C���@�v*����@�u�����@�v�����C��>@�v*����@�u�����@�v�����C��R@�v*����@�u�����@�v�����C� P@�v*����@�u�����@�v�����C�f�@�z*����@�y�����@�z�����C�u�@�z*����@�y�����@�z�����C���@�z*����@�y�����@�z�����C���@�z*����@�y�����@�z�����C���@�z*����@�y�����@�z�����C��q@�~*����@�}�����@�~�����C�I@�~*����@�}�����@�~�����C���@�~*����@�}�����@�~�����C�z0@�~*����@�}�����@�~�����C�E�@�~*����@�}�����@�~�����C��@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C�9�@��*����@�������@�������C�ȳ@��*����@�������@�������C��C@��*����@�������@�������C�c�@��*����@�������@�������C��p@��*����@�������@�������C��6@��*����@�������@�������C�#h@��*����@�������@�������C�N:@��*����@�������@�������C�m/@��*����@�������@�������C���@��*����@�������@�������C��m@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C�y�@��*����@�������@�������C���@��*����@�������@�������C��&@��*����@�������@�������C��@��*����@�������@�������C��;@��*����@�������@�������C�F@��*����@�������@�������C�-@��*����@�������@�������C�l*@��*����@�������@�������C�j�@��*����@�������@�������C�j[@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C�q�@��*����@�������@�������C�6�@��*����@�������@�������C�_@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C��?@��*����@�������@�������C�0v@��*����@�������@�������C�:�@��*����@�������@�������C��F@��*����@�������@�������C���@��*����@�������@�������C�Ψ@��*����@�������@�������C��@��*����@�������@�������C��^@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C� G@��*����@�������@�������C�C@��*����@�������@�������C�3L@��*����@�������@�������C�7}@��*����@�������@�������C���@��*����@�������@�������C��p@��*����@�������@�������C�a�@��*����@�������@�������C��r@��*����@�������@�������C�i-@��*����@�������@�������C�G*@��*����@�������@�������C��@��*����@�������@�������C�S@��*����@�������@�������C�g~@��*����@�������@�������C��i@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C���@��*����@�������@�������C�C"@��*����@�������@�������C��@��*����@�������@�������C�p1@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C�^�@��*����@�������@�������C�s@��*����@�������@�������C�A�@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C�ָ@��*����@�������@�������C�6�@��*����@�������@�������C��<@��*����@�������@�������C�p�@��*����@�������@�������C���@��*����@�������@�������C��P@��*����@�������@�������C��i@��*����@�������@�������C��t@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C�{@��*����@�������@�������C��@��*����@�������@�ª����C�7�@��*����@�������@�ª����C�J�@��*����@�������@�ª����C���@��*����@�������@�ª����C���@��*����@�������@�ª����C���@��*����@�������@�ª����C���@��*����@�Ū����@�ƪ����C��B@��*����@�Ū����@�ƪ����C�0�@��*����@�Ū����@�ƪ����C��@��*����@�Ū����@�ƪ����C��@��*����@�Ū����@�ƪ����C�Ka@��*����@�Ū����@�ƪ����C���@��*����@�ɪ����@�ʪ����C�qL@��*����@�ɪ����@�ʪ����C��O@��*����@�ɪ����@�ʪ����C�[�@��*����@�ɪ����@�ʪ����C��@��*����@�ɪ����@�ʪ����C�	�@��*����@�ɪ����@�ʪ����C��@��*����@�ͪ����@�Ϊ����C��@��*����@�ͪ����@�Ϊ����C���@��*����@�ͪ����@�Ϊ����C�6w@��*����@�ͪ����@�Ϊ����C��@��*����@�ͪ����@�Ϊ����C��o@��*����@�ͪ����@�Ϊ����C�-@��*����@�Ѫ����@�Ҫ����C��I@��*����@�Ѫ����@�Ҫ����C��\@��*����@�Ѫ����@�Ҫ����C�,8@��*����@�Ѫ����@�Ҫ����C���@��*����@�Ѫ����@�Ҫ����C��@��*����@�Ѫ����@�Ҫ����C��^@��*����@�ժ����@�֪����C�W�@��*����@�ժ����@�֪����C���@��*����@�ժ����@�֪����C��@��*����@�ժ����@�֪����C�$�@��*����@�ժ����@�֪����C�	F@��*����@�ժ����@�֪����C���@��*����@�٪����@�ڪ����C�l�@��*����@�٪����@�ڪ����C�ٶ@��*����@�٪����@�ڪ����C�Dj@��*����@�٪����@�ڪ����C���@��*����@�٪����@�ڪ����C�@��*����@�٪����@�ڪ����C�E@��*����@�ݪ����@�ު����C�,�@��*����@�ݪ����@�ު����C�@��*����@�ݪ����@�ު����C�1
@��*����@�ݪ����@�ު����C���@��*����@�ݪ����@�ު����C��\@��*����@�ݪ����@�ު����C�b$@��*����@�᪪���@�⪪���C�>7@��*����@�᪪���@�⪪���C��@��*����@�᪪���@�⪪���C��q@��*����@�᪪���@�⪪���C��@��*����@�᪪���@�⪪���C��@��*����@�᪪���@�⪪���C��@��*����@�媪���@�檪���C���@��*����@�媪���@�檪���C�vb@��*����@�媪���@�檪���C���@��*����@�媪���@�檪���C��@��*����@�媪���@�檪���C��T@��*����@�媪���@�檪���C�A�@��*����@�骪���@�ꪪ���C�G]@��*����@�骪���@�ꪪ���C�=�@��*����@�骪���@�ꪪ���C���@��*����@�骪���@�ꪪ���C�� @��*����@�骪���@�ꪪ���C�X�@��*����@�骪���@�ꪪ���C���@��*����@�������@����C��@��*����@�������@����C��M@��*����@�������@����C��@��*����@�������@����C�Q�@��*����@�������@����C�-�@��*����@�������@����C���@��*����@�񪪪��@�򪪪��C��Z@��*����@�񪪪��@�򪪪��C��y@��*����@�񪪪��@�򪪪��C��f@��*����@�񪪪��@�򪪪��C��@��*����@�񪪪��@�򪪪��C�<'@��*����@�񪪪��@�򪪪��C���@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C�X�@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C�J�@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C���@��*����@�������@�������C�܄@��*����@�������@�������C��@��*����@�������@�������C�EF@��*����@�������@�������C�@��*����@�������@�������C���@��*����@�������@�������C�|@��*����@�������@�������C��)@��*����@�������@�������C�'�@��*����@�������@�������C��8@�*����@������@������C�_k@�*����@������@������C���@�*����@������@������C�R@�*����@������@������C��:@�*����@������@������C���@�*����@������@������C�I�@�*����@������@������C�nV@�*����@������@������C�L�@�*����@������@������C�O�@�*����@������@������C�w�@�*����@������@������C�@�*����@������@������C�-f@�
*����@�	�����@�
�����C�@�
*����@�	�����@�
�����C�@�
*����@�	�����@�
�����C���@�
*����@�	�����@�
�����C�w@�
*����@�	�����@�
�����C�gy@�
*����@�	�����@�
�����C�o�@�*����@������@������C��+@�*����@������@������C�NY@�*����@������@������C��=@�*����@������@������C�gS@�*����@������@������C�[{@�*����@������@������C��|@�*����@������@������C�,�@�*����@������@������C��@�*����@������@������C�e�@�*����@������@������C�@�*����@������@������C�>@�*����@������@������C���@�*����@������@������C�H@�*����@������@������C�J�@�*����@������@������C�Ҳ@�*����@������@������C�uh@�*����@������@������C���@�*����@������@������C��^@�*����@������@������C�R�@�*����@������@������C�.=@�*����@������@������C�Yg@�*����@������@������C��@�*����@������@������C�`S@�*����@������@������C���@�*����@������@������C���@�*����@������@������C��-@�*����@������@������C��\@�*����@������@������C�@�*����@������@������C�V{@�*����@������@������C�d�@�"*����@�!�����@�"�����C��G@�"*����@�!�����@�"�����C�b<@�"*����@�!�����@�"�����C�|Q@�"*����@�!�����@�"�����C�f�@�"*����@�!�����@�"�����C�0 @�"*����@�!�����@�"�����C�5@�&*����@�%�����@�&�����C��n@�&*����@�%�����@�&�����C�^@�&*����@�%�����@�&�����C� �@�&*����@�%�����@�&�����C�U\@�&*����@�%�����@�&�����C���@�&*����@�%�����@�&�����C�2�@�**����@�)�����@�*�����C���@�**����@�)�����@�*�����C�N�@�**����@�)�����@�*�����C�Ζ@�**����@�)�����@�*�����C���@�**����@�)�����@�*�����C���@�**����@�)�����@�*�����C��y@�.*����@�-�����@�.�����C��j@�.*����@�-�����@�.�����C�+z@�.*����@�-�����@�.�����C��?@�.*����@�-�����@�.�����C���@�.*����@�-�����@�.�����C�e@�.*����@�-�����@�.�����C�<�@�2*����@�1�����@�2�����C��$@�2*����@�1�����@�2�����C��Z@�2*����@�1�����@�2�����C�C6@�2*����@�1�����@�2�����C��W@�2*����@�1�����@�2�����C�%@�2*����@�1�����@�2�����C��&@�6*����@�5�����@�6�����C�=�@�6*����@�5�����@�6�����C��u@�6*����@�5�����@�6�����C���@�6*����@�5�����@�6�����C��@�6*����@�5�����@�6�����C��@�6*����@�5�����@�6�����C�
�@�:*����@�9�����@�:�����C���@�:*����@�9�����@�:�����C��@�:*����@�9�����@�:�����C�|@�:*����@�9�����@�:�����C�:@�:*����@�9�����@�:�����C��@�:*����@�9�����@�:�����C��@�>*����@�=�����@�>�����C���@�>*����@�=�����@�>�����C�G@�>*����@�=�����@�>�����C�5@�>*����@�=�����@�>�����C���@�>*����@�=�����@�>�����C�T�@�>*����@�=�����@�>�����C��@�B*����@�A�����@�B�����C�Ǻ@�B*����@�A�����@�B�����C��@�B*����@�A�����@�B�����C��w@�B*����@�A�����@�B�����C�e�@�B*����@�A�����@�B�����C�lj@�B*����@�A�����@�B�����C�':@�F*����@�E�����@�F�����C��T@�F*����@�E�����@�F�����C��@�F*����@�E�����@�F�����C��@�F*����@�E�����@�F�����C�4@�F*����@�E�����@�F�����C�^f@�F*����@�E�����@�F�����C�@4@�J*����@�I�����@�J�����C�T$@�J*����@�I�����@�J�����C��3@�J*����@�I�����@�J�����C�Ɓ@�J*����@�I�����@�J�����C��7@�J*����@�I�����@�J�����C�2+@�J*����@�I�����@�J�����C� @�N*����@�M�����@�N�����C�ō@�N*����@�M�����@�N�����C�%N@�N*����@�M�����@�N�����C�h�@�N*����@�M�����@�N�����C��@�N*����@�M�����@�N�����C�C,@�N*����@�M�����@�N�����C���@�R*����@�Q�����@�R�����C�+N@�R*����@�Q�����@�R�����C���@�R*����@�Q�����@�R�����C�r�@�R*����@�Q�����@�R�����C��8@�R*����@�Q�����@�R�����C��\@�R*����@�Q�����@�R�����C�X9@�V*����@�U�����@�V�����C���@�V*����@�U�����@�V�����C���@�V*����@�U�����@�V�����C�ڮ@�V*����@�U�����@�V�����C�x�@�V*����@�U�����@�V�����C�ɛ@�V*����@�U�����@�V�����C��@�Z*����@�Y�����@�Z�����C�Q(@�Z*����@�Y�����@�Z�����C�tz@�Z*����@�Y�����@�Z�����C��@�Z*����@�Y�����@�Z�����C�2@�Z*����@�Y�����@�Z�����C���@�Z*����@�Y�����@�Z�����C���@�^*����@�]�����@�^�����C�I.@�^*����@�]�����@�^�����C��@�^*����@�]�����@�^�����C��@�^*����@�]�����@�^�����C�%�@�^*����@�]�����@�^�����C�۸@�^*����@�]�����@�^�����C�
�@�b*����@�a�����@�b�����C��2@�b*����@�a�����@�b�����C�,j@�b*����@�a�����@�b�����C�N�@�b*����@�a�����@�b�����C���@�b*����@�a�����@�b�����C��c@�b*����@�a�����@�b�����C���@�f*����@�e�����@�f�����C�5�@�f*����@�e�����@�f�����C�<!@�f*����@�e�����@�f�����C��H@�f*����@�e�����@�f�����C��U@�f*����@�e�����@�f�����C���@�f*����@�e�����@�f�����C��@�j*����@�i�����@�j�����C�,@�j*����@�i�����@�j�����C��$@�j*����@�i�����@�j�����C���@�j*����@�i�����@�j�����C��@�j*����@�i�����@�j�����C��@�j*����@�i�����@�j�����C�0�@�n*����@�m�����@�n�����C���@�n*����@�m�����@�n�����C�Cj@�n*����@�m�����@�n�����C��@�n*����@�m�����@�n�����C�2�@�n*����@�m�����@�n�����C�օ@�n*����@�m�����@�n�����C�@�r*����@�q�����@�r�����C���@�r*����@�q�����@�r�����C�gA@�r*����@�q�����@�r�����C�Fm@�r*����@�q�����@�r�����C�D�@�r*����@�q�����@�r�����C�=6@�r*����@�q�����@�r�����C�ٸ@�v*����@�u�����@�v�����C��@�v*����@�u�����@�v�����C�/g@�v*����@�u�����@�v�����C�6�@�v*����@�u�����@�v�����C�e�@�v*����@�u�����@�v�����C�<a@�v*����@�u�����@�v�����C��@�z*����@�y�����@�z�����C�l@�z*����@�y�����@�z�����C���@�z*����@�y�����@�z�����C�K@�z*����@�y�����@�z�����C�Q@�z*����@�y�����@�z�����C���@�z*����@�y�����@�z�����C���@�~*����@�}�����@�~�����C�Y�@�~*����@�}�����@�~�����C��@�~*����@�}�����@�~�����C�*@�~*����@�}�����@�~�����C�w@�~*����@�}�����@�~�����C���@�~*����@�}�����@�~�����C� @��*����@�������@�������C���@��*����@�������@�������C�WX@��*����@�������@�������C�B}@��*����@�������@�������C��,@��*����@�������@�������C��@��*����@�������@�������C�z�@��*����@�������@�������C��@��*����@�������@�������C�d�@��*����@�������@�������C��v@��*����@�������@�������C�X@��*����@�������@�������C�MS@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C�Q@��*����@�������@�������C�v!@��*����@�������@�������C��@��*����@�������@�������C�N@@��*����@�������@�������C��@��*����@�������@�������C��J@��*����@�������@�������C���@��*����@�������@�������C� �@��*����@�������@�������C��@��*����@�������@�������C�9c@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C�~�@��*����@�������@�������C���@��*����@�������@�������C�@��*����@�������@�������C�F�@��*����@�������@�������C���@��*����@�������@�������C��*@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C���@��*����@�������@�������C��]@��*����@�������@�������C�[w@��*����@�������@�������C�6@��*����@�������@�������C�vs@��*����@�������@�������C�
�@��*����@�������@�������C�@��*����@�������@�������C��@��*����@�������@�������C�cZ@��*����@�������@�������C�T�@��*����@�������@�������C�b/@��*����@�������@�������C��h@��*����@�������@�������C�M�@��*����@�������@�������C���@��*����@�������@�������C�u�@��*����@�������@�������C�q]@��*����@�������@�������C�/;@��*����@�������@�������C�4a@��*����@�������@�������C�s@��*����@�������@�������C�&k@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C���@��*����@�������@�������C�-\@��*����@�������@�������C�g'@��*����@�������@�������C��|@��*����@�������@�������C�ŧ@��*����@�������@�������C��Y@��*����@�������@�������C�OV@��*����@�������@�������C��h@��*����@�������@�������C�L@��*����@�������@�������C���@��*����@�������@�������C�Dq@��*����@�������@�������C���@��*����@�������@�������C��L@��*����@�������@�������C�7l@��*����@�������@�������C�7@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C�v�@��*����@�������@�������C��@��*����@�������@�������C��i@��*����@�������@�������C��@��*����@�������@�������C��P@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C��@��*����@�������@�������C�	'@��*����@�������@�������C���@��*����@�������@�������C�H@��*����@�������@�������C�71@��*����@�������@�������C���@��*����@�������@�������C�$ @��*����@�������@�������C���@��*����@�������@�������C��=@��*����@�������@�������C�d�@��*����@�������@�������C��@��*����@�������@�������C��R@��*����@�������@�ª����C���@��*����@�������@�ª����C��@��*����@�������@�ª����C���@��*����@�������@�ª����C���@��*����@�������@�ª����C�n�@��*����@�������@�ª����C�e@��*����@�Ū����@�ƪ����C�;@��*����@�Ū����@�ƪ����C��J@��*����@�Ū����@�ƪ����C�)�@��*����@�Ū����@�ƪ����C�ڻ@��*����@�Ū����@�ƪ����C��W@��*����@�Ū����@�ƪ����C��@��*����@�ɪ����@�ʪ����C��Y@��*����@�ɪ����@�ʪ����C��4@��*����@�ɪ����@�ʪ����C�ϓ@��*����@�ɪ����@�ʪ����C�-@��*����@�ɪ����@�ʪ����C���@��*����@�ɪ����@�ʪ����C��|@��*����@�ͪ����@�Ϊ����C�8�@��*����@�ͪ����@�Ϊ����C�E@��*����@�ͪ����@�Ϊ����C�^@��*����@�ͪ����@�Ϊ����C�a@��*����@�ͪ����@�Ϊ����C��@��*����@�ͪ����@�Ϊ����C���@��*����@�Ѫ����@�Ҫ����C�c�@��*����@�Ѫ����@�Ҫ����C�~@��*����@�Ѫ����@�Ҫ����C���@��*����@�Ѫ����@�Ҫ����C�g�@��*����@�Ѫ����@�Ҫ����C���@��*����@�Ѫ����@�Ҫ����C��+@��*����@�ժ����@�֪����C�^@��*����@�ժ����@�֪����C���@��*����@�ժ����@�֪����C���@��*����@�ժ����@�֪����C�C@��*����@�ժ����@�֪����C�I�@��*����@�ժ����@�֪����C�w�@��*����@�٪����@�ڪ����C���@��*����@�٪����@�ڪ����C�Et@��*����@�٪����@�ڪ����C���@��*����@�٪����@�ڪ����C�rC@��*����@�٪����@�ڪ����C���@��*����@�٪����@�ڪ����C���@��*����@�ݪ����@�ު����C�@�@��*����@�ݪ����@�ު����C��G@��*����@�ݪ����@�ު����C�z8@��*����@�ݪ����@�ު����C��@��*����@�ݪ����@�ު����C�[b@��*����@�ݪ����@�ު����C�y�@��*����@�᪪���@�⪪���C�
@��*����@�᪪���@�⪪���C�^�@��*����@�᪪���@�⪪���C���@��*����@�᪪���@�⪪���C���@��*����@�᪪���@�⪪���C��@��*����@�᪪���@�⪪���C�!a@��*����@�媪���@�檪���C�:�@��*����@�媪���@�檪���C��@��*����@�媪���@�檪���C��;@��*����@�媪���@�檪���C���@��*����@�媪���@�檪���C��x@��*����@�媪���@�檪���C�yw@��*����@�骪���@�ꪪ���C�d@��*����@�骪���@�ꪪ���C��@��*����@�骪���@�ꪪ���C�g@��*����@�骪���@�ꪪ���C���@��*����@�骪���@�ꪪ���C�%�@��*����@�骪���@�ꪪ���C�*Y@��*����@�������@����C�@��*����@�������@����C��@��*����@�������@����C�*�@��*����@�������@����C�;�@��*����@�������@����C���@��*����@�������@����C�[�@��*����@�񪪪��@�򪪪��C���@��*����@�񪪪��@�򪪪��C��#@��*����@�񪪪��@�򪪪��C�4�@��*����@�񪪪��@�򪪪��C�"�@��*����@�񪪪��@�򪪪��C�}�@��*����@�񪪪��@�򪪪��C�D�@��*����@�������@�������C�k�@��*����@�������@�������C�L>@��*����@�������@�������C�q�@��*����@�������@�������C�G@��*����@�������@�������C��l@��*����@�������@�������C��T@��*����@�������@�������C�|@��*����@�������@�������C�ũ@��*����@�������@�������C�s�@��*����@�������@�������C���@��*����@�������@�������C�̿@��*����@�������@�������C�]\@��*����@�������@�������C�H=@��*����@�������@�������C�>�@��*����@�������@�������C�8@��*����@�������@�������C�ۯ@��*����@�������@�������C��E@��*����@�������@�������C�c�@�*����@������@������C���@�*����@������@������C��e@�*����@������@������C��@�*����@������@������C�2\@�*����@������@������C�Qg@�*����@������@������C��S@�*����@������@������C��z@�*����@������@������C���@�*����@������@������C���@�*����@������@������C��o@�*����@������@������C��@�*����@������@������C��@�
*����@�	�����@�
�����C�f�@�
*����@�	�����@�
�����C���@�
*����@�	�����@�
�����C���@�
*����@�	�����@�
�����C���@�
*����@�	�����@�
�����C���@�
*����@�	�����@�
�����C�N�@�*����@������@������C��@�*����@������@������C�9@�*����@������@������C��`@�*����@������@������C�(@�*����@������@������C�ʈ@�*����@������@������C�g�@�*����@������@������C�]G@�*����@������@������C�tC@�*����@������@������C�\�@�*����@������@������C��@�*����@������@������C��|@�*����@������@������C���@�*����@������@������C���@�*����@������@������C�$b@�*����@������@������C���@�*����@������@������C��`@�*����@������@������C�b�@�*����@������@������C�@�*����@������@������C��2@�*����@������@������C���@�*����@������@������C��O@�*����@������@������C��F@�*����@������@������C��>@�*����@������@������C�@�*����@������@������C�a@�*����@������@������C��:@�*����@������@������C���@�*����@������@������C��@�*����@������@������C��g@�*����@������@������C��Y@�"*����@�!�����@�"�����C���@�"*����@�!�����@�"�����C�%�@�"*����@�!�����@�"�����C�B%@�"*����@�!�����@�"�����C�a@�"*����@�!�����@�"�����C���@�"*����@�!�����@�"�����C��@�&*����@�%�����@�&�����C��@�&*����@�%�����@�&�����C�0m@�&*����@�%�����@�&�����C��@�&*����@�%�����@�&�����C��@�&*����@�%�����@�&�����C���@�&*����@�%�����@�&�����C�J9@�**����@�)�����@�*�����C�m�@�**����@�)�����@�*�����C��@�**����@�)�����@�*�����C�`@�**����@�)�����@�*�����C�I�@�**����@�)�����@�*�����C�w�@�**����@�)�����@�*�����C�o@�.*����@�-�����@�.�����C���@�.*����@�-�����@�.�����C���@�.*����@�-�����@�.�����C���@�.*����@�-�����@�.�����C��@�.*����@�-�����@�.�����C��d@�.*����@�-�����@�.�����C�9_@�2*����@�1�����@�2�����C�t@�2*����@�1�����@�2�����C�,�@�2*����@�1�����@�2�����C�'�@�2*����@�1�����@�2�����C�5�@�2*����@�1�����@�2�����C�g�@�2*����@�1�����@�2�����C�g�@�6*����@�5�����@�6�����C���@�6*����@�5�����@�6�����C�1�@�6*����@�5�����@�6�����C�g@�6*����@�5�����@�6�����C���@�6*����@�5�����@�6�����C��`@�6*����@�5�����@�6�����C�g|@�:*����@�9�����@�:�����C���@�:*����@�9�����@�:�����C�g�@�:*����@�9�����@�:�����C��@�:*����@�9�����@�:�����C���@�:*����@�9�����@�:�����C�}o@�:*����@�9�����@�:�����C��)@�>*����@�=�����@�>�����C�{@�>*����@�=�����@�>�����C���@�>*����@�=�����@�>�����C�n>@�>*����@�=�����@�>�����C��l@�>*����@�=�����@�>�����C�� @�>*����@�=�����@�>�����C�w@�B*����@�A�����@�B�����C�,)@�B*����@�A�����@�B�����C���@�B*����@�A�����@�B�����C��@�B*����@�A�����@�B�����C���@�B*����@�A�����@�B�����C�t]@�B*����@�A�����@�B�����C���@�F*����@�E�����@�F�����C��@�F*����@�E�����@�F�����C���@�F*����@�E�����@�F�����C�&@�F*����@�E�����@�F�����C���@�F*����@�E�����@�F�����C���@�F*����@�E�����@�F�����C���@�J*����@�I�����@�J�����C��@�J*����@�I�����@�J�����C�(N@�J*����@�I�����@�J�����C�O'@�J*����@�I�����@�J�����C�H�@�J*����@�I�����@�J�����C��@�J*����@�I�����@�J�����C��@�N*����@�M�����@�N�����C���@�N*����@�M�����@�N�����C�I�@�N*����@�M�����@�N�����C�d�@�N*����@�M�����@�N�����C��@�N*����@�M�����@�N�����C�*�@�N*����@�M�����@�N�����C���@�R*����@�Q�����@�R�����C��@�R*����@�Q�����@�R�����C�k�@�R*����@�Q�����@�R�����C��@�R*����@�Q�����@�R�����C���@�R*����@�Q�����@�R�����C�	 @�R*����@�Q�����@�R�����C��%@�V*����@�U�����@�V�����C��O@�V*����@�U�����@�V�����C���@�V*����@�U�����@�V�����C�J�@�V*����@�U�����@�V�����C��7@�V*����@�U�����@�V�����C��@�V*����@�U�����@�V�����C���@�Z*����@�Y�����@�Z�����C�N@�Z*����@�Y�����@�Z�����C��@�Z*����@�Y�����@�Z�����C�)%@�Z*����@�Y�����@�Z�����C�D�@�Z*����@�Y�����@�Z�����C�0�@�Z*����@�Y�����@�Z�����C�4�@�Z*����@�Y�����@�Z�����C�eb@�Z*����@�Y�����@�Z�����C�F@�^*����@�]�����@�^�����C�q�@�^*����@�]�����@�^�����C��@�^*����@�]�����@�^�����C���@�^*����@�]�����@�^�����C�.�@�^*����@�]�����@�^�����C��{@�^*����@�]�����@�^�����C��e@�^*����@�]�����@�^�����C�q�@�^*����@�]�����@�^�����C�%@@�b*����@�a�����@�b�����C�'@�b*����@�a�����@�b�����C��<@�b*����@�a�����@�b�����C�M@�b*����@�a�����@�b�����C�b@�b*����@�a�����@�b�����C���@�b*����@�a�����@�b�����C���@�b*����@�a�����@�b�����C�j
@�b*����@�a�����@�b�����C��@�f*����@�e�����@�f�����C���@�f*����@�e�����@�f�����C��a@�f*����@�e�����@�f�����C�^~@�f*����@�e�����@�f�����C�+}@�f*����@�e�����@�f�����C��@�f*����@�e�����@�f�����C��g@�f*����@�e�����@�f�����C���@�f*����@�e�����@�f�����C��d@�j*����@�i�����@�j�����C��9@�j*����@�i�����@�j�����C���@�j*����@�i�����@�j�����C�N}@�j*����@�i�����@�j�����C�y�@�j*����@�i�����@�j�����C���@�j*����@�i�����@�j�����C�x>@�n*����@�m�����@�n�����C�Ԉ@�n*����@�m�����@�n�����C�$�@�n*����@�m�����@�n�����C�H�@�n*����@�m�����@�n�����C���@�n*����@�m�����@�n�����C���@�n*����@�m�����@�n�����C��{@�r*����@�q�����@�r�����C��@�r*����@�q�����@�r�����C���@�r*����@�q�����@�r�����C�O�@�r*����@�q�����@�r�����C�i�@�r*����@�q�����@�r�����C�(�@�r*����@�q�����@�r�����C�ʳ@�v*����@�u�����@�v�����C���@�v*����@�u�����@�v�����C�,~@�z*����@�y�����@�z�����C�m�@�z*����@�y�����@�z�����C��B@�~*����@�}�����@�~�����C��i@�~*����@�}�����@�~�����C�J9@��*����@�������@�������C�3�@��*����@�������@�������C�<T@��*����@�������@�������C�sb@��*����@�������@�������C��@��*����@�������@�������C���@��*����@�������@�������C�zs@��*����@�������@�������C���@��*����@�������@�������C��1@��*����@�������@�������C��S@��*����@�������@�������C��h@��*����@�������@�������C�U@��*����@�������@�������C�È@��*����@�������@�������C��1@��*����@�������@�������C�J�@��*����@�������@�������C�!?@��*����@�������@�������C�u@��*����@�������@�������C�\@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C��L@��*����@�������@�������C��a@��*����@�������@�������C� p@��*����@�������@�������C��]@��*����@�������@�������C�@��*����@�������@�������C�ٍ@��*����@�������@�������C�@��*����@�������@�������C��@��*����@�������@�������C�
�@��*����@�������@�������C���@��*����@�������@�������C��0@��*����@�������@�������C�T�@��*����@�������@�ª����C�A=@��*����@�������@�ª����C�@��*����@�Ū����@�ƪ����C�s�@��*����@�Ū����@�ƪ����C�@��*����@�ɪ����@�ʪ����C�-�@��*����@�ɪ����@�ʪ����C���@��*����@�ͪ����@�Ϊ����C��@��*����@�ͪ����@�Ϊ����C�
�@��*����@�Ѫ����@�Ҫ����C�ǯ@��*����@�Ѫ����@�Ҫ����C�<,@��*����@�ժ����@�֪����C��@��*����@�ժ����@�֪����C���@��*����@�٪����@�ڪ����C�Ml@��*����@�٪����@�ڪ����C��@��*����@�ݪ����@�ު����C���@��*����@�ݪ����@�ު����C�-	@��*����@�᪪���@�⪪���C��u@��*����@�᪪���@�⪪���C�2@��*����@�媪���@�檪���C�	q@��*����@�媪���@�檪���C��Z@��*����@�骪���@�ꪪ���C�Gf@��*����@�骪���@�ꪪ���C��@��*����@�������@����C�6�@��*����@�������@����C�n�@��*����@�񪪪��@�򪪪��C��@��*����@�񪪪��@�򪪪��C��?@��*����@�������@�������C�HS@��*����@�������@�������C���@��*����@�������@�������C��+@��*����@�������@�������C�b8@��*����@�������@�������C���@��*����@�������@�������C�On@�UUUU@� �UUUU@�UUUUUC�>@�UUUU@� �UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC�U@�UUUU@��UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC�V~@�UUUU@��UUUU@�UUUUUC�uV@�UUUU@��UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC�Y�@�	UUUU@��UUUU@�	UUUUUC��O@�	UUUU@��UUUU@�	UUUUUC��
@�	UUUU@��UUUU@�	UUUUUC�@�@�UUUU@�
�UUUU@�UUUUUC�}@�UUUU@�
�UUUU@�UUUUUC���@�UUUU@�
�UUUU@�UUUUUC�؎@�UUUU@��UUUU@�UUUUUC�٢@�UUUU@��UUUU@�UUUUUC�j@�UUUU@��UUUU@�UUUUUC��4@�UUUU@��UUUU@�UUUUUC��]@�UUUU@��UUUU@�UUUUUC�$�@�UUUU@��UUUU@�UUUUUC�E�@�UUUU@��UUUU@�UUUUUC�$�@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC��;@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC��B@�UUUU@��UUUU@�UUUUUC�b@�UUUU@��UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC�@�UUUU@��UUUU@�UUUUUC�m;@�UUUU@��UUUU@�UUUUUC� �@�UUUU@��UUUU@�UUUUUC�)@�UUUU@��UUUU@�UUUUUC�D�@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC�t@�UUUU@��UUUU@�UUUUUC�X�@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC�Q�@�UUUU@��UUUU@�UUUUUC�4_@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC�ם@�UUUU@��UUUU@�UUUUUC��`@�UUUU@��UUUU@�UUUUUC��@�!UUUU@� �UUUU@�!UUUUUC�c�@�!UUUU@� �UUUU@�!UUUUUC�[�@�!UUUU@� �UUUU@�!UUUUUC���@�#UUUU@�"�UUUU@�#UUUUUC��d@�#UUUU@�"�UUUU@�#UUUUUC���@�#UUUU@�"�UUUU@�#UUUUUC�H@�%UUUU@�$�UUUU@�%UUUUUC�ܜ@�%UUUU@�$�UUUU@�%UUUUUC���@�%UUUU@�$�UUUU@�%UUUUUC���@�'UUUU@�&�UUUU@�'UUUUUC�?+@�'UUUU@�&�UUUU@�'UUUUUC�'�@�'UUUU@�&�UUUU@�'UUUUUC��@�)UUUU@�(�UUUU@�)UUUUUC�&�@�)UUUU@�(�UUUU@�)UUUUUC��G@�)UUUU@�(�UUUU@�)UUUUUC���@�+UUUU@�*�UUUU@�+UUUUUC��n@�+UUUU@�*�UUUU@�+UUUUUC��1@�+UUUU@�*�UUUU@�+UUUUUC��h@�-UUUU@�,�UUUU@�-UUUUUC�$?@�-UUUU@�,�UUUU@�-UUUUUC��@�-UUUU@�,�UUUU@�-UUUUUC�E}@�/UUUU@�.�UUUU@�/UUUUUC��%@�/UUUU@�.�UUUU@�/UUUUUC��@�/UUUU@�.�UUUU@�/UUUUUC��@�1UUUU@�0�UUUU@�1UUUUUC�D@�1UUUU@�0�UUUU@�1UUUUUC�z@�1UUUU@�0�UUUU@�1UUUUUC���@�3UUUU@�2�UUUU@�3UUUUUC�.�@�3UUUU@�2�UUUU@�3UUUUUC� Q@�3UUUU@�2�UUUU@�3UUUUUC�܏@�5UUUU@�4�UUUU@�5UUUUUC�;k@�5UUUU@�4�UUUU@�5UUUUUC�y?@�5UUUU@�4�UUUU@�5UUUUUC�@�7UUUU@�6�UUUU@�7UUUUUC�o�@�7UUUU@�6�UUUU@�7UUUUUC��|@�7UUUU@�6�UUUU@�7UUUUUC���@�9UUUU@�8�UUUU@�9UUUUUC�y�@�9UUUU@�8�UUUU@�9UUUUUC��}@�9UUUU@�8�UUUU@�9UUUUUC���@�;UUUU@�:�UUUU@�;UUUUUC�6@�;UUUU@�:�UUUU@�;UUUUUC�~n@�;UUUU@�:�UUUU@�;UUUUUC��@�=UUUU@�<�UUUU@�=UUUUUC��G@�=UUUU@�<�UUUU@�=UUUUUC���@�=UUUU@�<�UUUU@�=UUUUUC���@�?UUUU@�>�UUUU@�?UUUUUC��@�?UUUU@�>�UUUU@�?UUUUUC��:@�?UUUU@�>�UUUU@�?UUUUUC��M@�AUUUU@�@�UUUU@�AUUUUUC��b@�AUUUU@�@�UUUU@�AUUUUUC���@�AUUUU@�@�UUUU@�AUUUUUC�\T@�CUUUU@�B�UUUU@�CUUUUUC�S4@�CUUUU@�B�UUUU@�CUUUUUC�s@�CUUUU@�B�UUUU@�CUUUUUC�@�EUUUU@�D�UUUU@�EUUUUUC��@�EUUUU@�D�UUUU@�EUUUUUC���@�EUUUU@�D�UUUU@�EUUUUUC�~�@�GUUUU@�F�UUUU@�GUUUUUC���@�GUUUU@�F�UUUU@�GUUUUUC�b�@�GUUUU@�F�UUUU@�GUUUUUC�k&@�IUUUU@�H�UUUU@�IUUUUUC�u
@�IUUUU@�H�UUUU@�IUUUUUC�+�@�IUUUU@�H�UUUU@�IUUUUUC���@�KUUUU@�J�UUUU@�KUUUUUC��@�KUUUU@�J�UUUU@�KUUUUUC��@�KUUUU@�J�UUUU@�KUUUUUC��$@�MUUUU@�L�UUUU@�MUUUUUC�x�@�MUUUU@�L�UUUU@�MUUUUUC���@�MUUUU@�L�UUUU@�MUUUUUC���@�OUUUU@�N�UUUU@�OUUUUUC���@�OUUUU@�N�UUUU@�OUUUUUC�=a@�OUUUU@�N�UUUU@�OUUUUUC�,@�QUUUU@�P�UUUU@�QUUUUUC�B1@�QUUUU@�P�UUUU@�QUUUUUC��m@�QUUUU@�P�UUUU@�QUUUUUC�VF@�SUUUU@�R�UUUU@�SUUUUUC��@�SUUUU@�R�UUUU@�SUUUUUC���@�SUUUU@�R�UUUU@�SUUUUUC��@�UUUUU@�T�UUUU@�UUUUUUC��<@�UUUUU@�T�UUUU@�UUUUUUC���@�UUUUU@�T�UUUU@�UUUUUUC���@�WUUUU@�V�UUUU@�WUUUUUC���@�WUUUU@�V�UUUU@�WUUUUUC��#@�WUUUU@�V�UUUU@�WUUUUUC�ȝ@�YUUUU@�X�UUUU@�YUUUUUC�g@�YUUUU@�X�UUUU@�YUUUUUC��@�YUUUU@�X�UUUU@�YUUUUUC��@�[UUUU@�Z�UUUU@�[UUUUUC�݀@�[UUUU@�Z�UUUU@�[UUUUUC�bj@�[UUUU@�Z�UUUU@�[UUUUUC�V#@�]UUUU@�\�UUUU@�]UUUUUC�0	@�]UUUU@�\�UUUU@�]UUUUUC���@�]UUUU@�\�UUUU@�]UUUUUC�H�@�_UUUU@�^�UUUU@�_UUUUUC��q@�_UUUU@�^�UUUU@�_UUUUUC�1@�_UUUU@�^�UUUU@�_UUUUUC��9@�aUUUU@�`�UUUU@�aUUUUUC��@�aUUUU@�`�UUUU@�aUUUUUC��#@�aUUUU@�`�UUUU@�aUUUUUC�(�@�cUUUU@�b�UUUU@�cUUUUUC���@�cUUUU@�b�UUUU@�cUUUUUC�:@�cUUUU@�b�UUUU@�cUUUUUC���@�eUUUU@�d�UUUU@�eUUUUUC�L�@�eUUUU@�d�UUUU@�eUUUUUC�!@�eUUUU@�d�UUUU@�eUUUUUC�7@�gUUUU@�f�UUUU@�gUUUUUC���@�gUUUU@�f�UUUU@�gUUUUUC��@�gUUUU@�f�UUUU@�gUUUUUC�	]@�iUUUU@�h�UUUU@�iUUUUUC��'@�iUUUU@�h�UUUU@�iUUUUUC�n�@�iUUUU@�h�UUUU@�iUUUUUC��