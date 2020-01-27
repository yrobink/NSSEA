CDF   �   
      lon       lat       time       bnds         &   CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      	+Sat Oct 19 17:39:59 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_CESM1-BGC_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HW03/rcp85/X//tas_mon_NSF-DOE-NCAR_CESM1-BGC.nc
Sat Oct 19 17:39:59 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_CESM1-BGC_seldate.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_CESM1-BGC_setref.nc
Sat Oct 19 17:39:59 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_CESM1-BGC_merge.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_CESM1-BGC_seldate.nc
Sat Oct 19 17:39:59 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_historical_r1i1p1_185001-200512_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_CESM1-BGC_merge.nc
Sat Oct 19 17:39:59 2019: cdo yearmean /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_yearmean.nc
Sat Oct 19 17:39:59 2019: cdo selseas,JJA /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_seltime.nc
Sat Oct 19 17:39:59 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_var.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_fldmean.nc
Sat Oct 19 17:39:59 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_box.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_var.nc
Sat Oct 19 17:39:57 2019: cdo sellonlatbox,-10,30,35,70 /bdd/CMIP5/main/NSF-DOE-NCAR/CESM1-BGC/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012_tmp_box.nc
2012-05-11T18:02:06Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.    source        	CESM1-BGC      institution       HNSF/DOE NCAR (National Center for Atmospheric Research) Boulder, CO, USA   institute_id      NSF-DOE-NCAR   experiment_id         
historical     model_id      	CESM1-BGC      forcing       $Sl GHG Vl SS Ds SD BC MD OC Oz AA LU   parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       ?�         contact       cesm_data@ucar.edu     comment       (CESM home page: http://www.cesm.ucar.edu   
references        3TBD
 See also http://www.cesm.ucar.edu/publications    initialization_method               physics_version             tracking_id       $110a947b-500f-4e60-a789-c23393123f5a   acknowledgements     �The CESM project is supported by the National Science Foundation and the Office of Science (BER) of the U.S. Department of Energy. NCAR is sponsored by the National Science Foundation. Computing resources were provided by the Climate Simulation Laboratory at the NCAR Computational and Information Systems Laboratory (CISL), sponsored by the National Science Foundation and other agencies.      cesm_casename         b40.20th.1deg.bdrd.001     cesm_repotag      unknown    cesm_compset      unknown    
resolution        f09_g16 (0.9x1.25_gx1v6)   forcing_note      �Additional information on the external forcings used in this experiment can be found at http://www.cesm.ucar.edu/CMIP5/forcing_information     processed_by      8strandwg on silver.cgd.ucar.edu at 20120516  -113148.887   processing_code_information       �Last Changed Rev: 760 Last Changed Date: 2012-05-15 15:48:26 -0600 (Tue, 15 May 2012) Repository UUID: d2181dbe-5796-6825-dc7f-cbd98591f93d    product       output     
experiment        
historical     	frequency         mon    creation_date         2012-05-16T17:32:00Z   
project_id        CMIP5      table_id      =Table Amon (12 January 2012) 4996d487f7a65749098d9cc0dccb4f8d      title         4CESM1-BGC model output prepared for CMIP5 historical   parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.8.1      CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      365_day         �   	time_bnds                            �   tas                    
   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   original_name         TREFHT     comment       TREFHT no change       cell_methods      time: mean (interval: 30 days)     history      2012-05-16T17:31:48Z altered by CMOR: Treated scalar dimension: 'height'. 2012-05-16T17:31:48Z altered by CMOR: Reordered dimensions, original order: lat lon time. 2012-05-16T17:31:48Z altered by CMOR: replaced missing value flag (-1e+32) with standard missing value (1e+20).    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CESM1-BGC_historical_r0i0p0.nc areacella: areacella_fx_CESM1-BGC_historical_r0i0p0.nc            �                @��*����@�骪���@�ꪪ���C���@��*����@�������@����C��@��*����@�񪪪��@�򪪪��C��@��*����@�������@�������C���@��*����@�������@�������C��b@��*����@�������@�������C�Fk@�*����@������@������C��q@�*����@������@������C��@�
*����@�	�����@�
�����C���@�*����@������@������C�r�@�*����@������@������C��@�*����@������@������C��O@�*����@������@������C���@�*����@������@������C��A@�"*����@�!�����@�"�����C��u@�&*����@�%�����@�&�����C�yA@�**����@�)�����@�*�����C�ó@�.*����@�-�����@�.�����C��@�2*����@�1�����@�2�����C���@�6*����@�5�����@�6�����C�~S@�:*����@�9�����@�:�����C���@�>*����@�=�����@�>�����C�?�@�B*����@�A�����@�B�����C�E�@�F*����@�E�����@�F�����C��\@�J*����@�I�����@�J�����C��@�N*����@�M�����@�N�����C��=@�R*����@�Q�����@�R�����C��@�V*����@�U�����@�V�����C���@�Z*����@�Y�����@�Z�����C���@�^*����@�]�����@�^�����C��K@�b*����@�a�����@�b�����C���@�f*����@�e�����@�f�����C�l-@�j*����@�i�����@�j�����C���@�n*����@�m�����@�n�����C�٣@�r*����@�q�����@�r�����C�w@�v*����@�u�����@�v�����C�[�@�z*����@�y�����@�z�����C���@�~*����@�}�����@�~�����C�NK@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C��d@��*����@�������@�������C�@��*����@�������@�������C�ã@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C��@��*����@�������@�������C�̾@��*����@�������@�������C��@��*����@�������@�������C�ڲ@��*����@�������@�������C�D:@��*����@�������@�������C���@��*����@�������@�������C�f@��*����@�������@�������C���@��*����@�������@�������C�yi@��*����@�������@�ª����C���@��*����@�Ū����@�ƪ����C��C@��*����@�ɪ����@�ʪ����C���@��*����@�ͪ����@�Ϊ����C���@��*����@�Ѫ����@�Ҫ����C�T�@��*����@�ժ����@�֪����C��@��*����@�٪����@�ڪ����C���@��*����@�ݪ����@�ު����C���@��*����@�᪪���@�⪪���C�V@��*����@�媪���@�檪���C��v@��*����@�骪���@�ꪪ���C�t�@��*����@�������@����C��(@��*����@�񪪪��@�򪪪��C��@��*����@�������@�������C��$@��*����@�������@�������C��@��*����@�������@�������C���@�*����@������@������C��
@�*����@������@������C��@�
*����@�	�����@�
�����C��t@�*����@������@������C�@�*����@������@������C��j@�*����@������@������C��n@�*����@������@������C�k�@�*����@������@������C�f�@�"*����@�!�����@�"�����C���@�&*����@�%�����@�&�����C��@�**����@�)�����@�*�����C���@�.*����@�-�����@�.�����C�O�@�2*����@�1�����@�2�����C��$@�6*����@�5�����@�6�����C�Q�@�:*����@�9�����@�:�����C��@�>*����@�=�����@�>�����C���@�B*����@�A�����@�B�����C�kf@�F*����@�E�����@�F�����C���@�J*����@�I�����@�J�����C��r@�N*����@�M�����@�N�����C�Ë@�R*����@�Q�����@�R�����C�)@�V*����@�U�����@�V�����C�|�@�Z*����@�Y�����@�Z�����C��@�^*����@�]�����@�^�����C���@�b*����@�a�����@�b�����C���@�f*����@�e�����@�f�����C�ˁ@�j*����@�i�����@�j�����C���@�n*����@�m�����@�n�����C�=@�r*����@�q�����@�r�����C���@�v*����@�u�����@�v�����C�fr@�z*����@�y�����@�z�����C�	_@�~*����@�}�����@�~�����C��/@��*����@�������@�������C��@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C�o�@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C�ߧ@��*����@�������@�������C��!@��*����@�������@�������C�z@��*����@�������@�������C��}@��*����@�������@�������C��@@��*����@�������@�������C�}�@��*����@�������@�������C��q@��*����@�������@�������C�&�@��*����@�������@�������C��L@��*����@�������@�ª����C���@��*����@�Ū����@�ƪ����C��p@��*����@�ɪ����@�ʪ����C��@��*����@�ͪ����@�Ϊ����C��h@��*����@�Ѫ����@�Ҫ����C��w@��*����@�ժ����@�֪����C��x@��*����@�٪����@�ڪ����C���@��*����@�ݪ����@�ު����C�І@��*����@�᪪���@�⪪���C��@��*����@�媪���@�檪���C�>6@��*����@�骪���@�ꪪ���C��l@��*����@�������@����C��>@��*����@�񪪪��@�򪪪��C�+5@��*����@�������@�������C�֥@��*����@�������@�������C���@��*����@�������@�������C���@�*����@������@������C��6@�*����@������@������C�A@�
*����@�	�����@�
�����C��@�*����@������@������C���@�*����@������@������C�:@�*����@������@������C��@�*����@������@������C�
�@�*����@������@������C��@�"*����@�!�����@�"�����C�*@�&*����@�%�����@�&�����C���@�**����@�)�����@�*�����C���@�.*����@�-�����@�.�����C��@�2*����@�1�����@�2�����C��N@�6*����@�5�����@�6�����C���@�:*����@�9�����@�:�����C�\S@�>*����@�=�����@�>�����C�)
@�B*����@�A�����@�B�����C��@�F*����@�E�����@�F�����C���@�J*����@�I�����@�J�����C��@�N*����@�M�����@�N�����C���@�R*����@�Q�����@�R�����C��@�V*����@�U�����@�V�����C�R@�Z*����@�Y�����@�Z�����C���@�^*����@�]�����@�^�����C��@�b*����@�a�����@�b�����C��@�f*����@�e�����@�f�����C�K3@�j*����@�i�����@�j�����C�Ec@�n*����@�m�����@�n�����C�J�@�r*����@�q�����@�r�����C�`@�v*����@�u�����@�v�����C��@�z*����@�y�����@�z�����C�h�@�~*����@�}�����@�~�����C��@��*����@�������@�������C��@��*����@�������@�������C��k@��*����@�������@�������C�@E@��*����@�������@�������C�}@��*����@�������@�������C���@��*����@�������@�������C�}>@��*����@�������@�������C�]�@��*����@�������@�������C�� @��*����@�������@�������C���@��*����@�������@�������C�\�@��*����@�������@�������C���@��*����@�������@�������C���@��*����@�������@�������C�'H@��*����@�������@�������C�&@��*����@�������@�������C�%�@��*����@�������@�������C�}X@��*����@�������@�ª����C���@��*����@�Ū����@�ƪ����C�?@��*����@�ɪ����@�ʪ����C�d@��*����@�ͪ����@�Ϊ����C�aT@��*����@�Ѫ����@�Ҫ����C���@��*����@�ժ����@�֪����C��C@��*����@�٪����@�ڪ����C�ۭ@��*����@�ݪ����@�ު����C��W@��*����@�᪪���@�⪪���C���@��*����@�媪���@�檪���C��K@��*����@�骪���@�ꪪ���C�8�@��*����@�������@����C��F@��*����@�񪪪��@�򪪪��C��3@��*����@�������@�������C��>@��*����@�������@�������C���@��*����@�������@�������C���@�UUUU@� �UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC��@�	UUUU@��UUUU@�	UUUUUC�0@�UUUU@�
�UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC��q@�UUUU@��UUUU@�UUUUUC��|@�UUUU@��UUUU@�UUUUUC�"�@�UUUU@��UUUU@�UUUUUC�LR@�UUUU@��UUUU@�UUUUUC��@�UUUU@��UUUU@�UUUUUC�+b@�UUUU@��UUUU@�UUUUUC�L@�UUUU@��UUUU@�UUUUUC��x@�UUUU@��UUUU@�UUUUUC���@�UUUU@��UUUU@�UUUUUC�G\@�!UUUU@� �UUUU@�!UUUUUC��@�#UUUU@�"�UUUU@�#UUUUUC�s�@�%UUUU@�$�UUUU@�%UUUUUC�~%@�'UUUU@�&�UUUU@�'UUUUUC�uV@�)UUUU@�(�UUUU@�)UUUUUC�A@�+UUUU@�*�UUUU@�+UUUUUC���@�-UUUU@�,�UUUU@�-UUUUUC�i�@�/UUUU@�.�UUUU@�/UUUUUC�IX@�1UUUU@�0�UUUU@�1UUUUUC��.@�3UUUU@�2�UUUU@�3UUUUUC�`�@�5UUUU@�4�UUUU@�5UUUUUC��@�7UUUU@�6�UUUU@�7UUUUUC��o@�9UUUU@�8�UUUU@�9UUUUUC��@�;UUUU@�:�UUUU@�;UUUUUC��@�=UUUU@�<�UUUU@�=UUUUUC��)@�?UUUU@�>�UUUU@�?UUUUUC���@�AUUUU@�@�UUUU@�AUUUUUC��@�CUUUU@�B�UUUU@�CUUUUUC���@�EUUUU@�D�UUUU@�EUUUUUC��[@�GUUUU@�F�UUUU@�GUUUUUC�`@�IUUUU@�H�UUUU@�IUUUUUC���@�KUUUU@�J�UUUU@�KUUUUUC�@�MUUUU@�L�UUUU@�MUUUUUC��'@�OUUUU@�N�UUUU@�OUUUUUC�v@�QUUUU@�P�UUUU@�QUUUUUC��@�SUUUU@�R�UUUU@�SUUUUUC�,@�UUUUU@�T�UUUU@�UUUUUUC�)�@�WUUUU@�V�UUUU@�WUUUUUC�2�@�YUUUU@�X�UUUU@�YUUUUUC�_N@�[UUUU@�Z�UUUU@�[UUUUUC��@�]UUUU@�\�UUUU@�]UUUUUC���@�_UUUU@�^�UUUU@�_UUUUUC�!@�aUUUU@�`�UUUU@�aUUUUUC�Ur@�cUUUU@�b�UUUU@�cUUUUUC�!@�eUUUU@�d�UUUU@�eUUUUUC�H�@�gUUUU@�f�UUUU@�gUUUUUC�@�@�iUUUU@�h�UUUU@�iUUUUUC��O