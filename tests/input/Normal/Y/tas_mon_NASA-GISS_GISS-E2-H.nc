CDF  f   
      lon       lat       time       bnds            CDI       ?Climate Data Interface version 1.6.8 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      ¼Sat Oct 19 17:57:20 2019: cdo settunits,years /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_GISS-E2-H_setref.nc /data/yrobin//Projets/EUPHEME/input/event/HW03/rcp85/Y//tas_mon_NASA-GISS_GISS-E2-H.nc
Sat Oct 19 17:57:19 2019: cdo setreftime,0000-01-01,00:00 /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_GISS-E2-H_seldate.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_GISS-E2-H_setref.nc
Sat Oct 19 17:57:19 2019: cdo seldate,1850-01-01,2100-12-31 /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_GISS-E2-H_merge.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268//tas_mon_GISS-E2-H_seldate.nc
Sat Oct 19 17:57:19 2019: cdo mergetime /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r1i1p1_185001-190012_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r1i1p1_190101-195012_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r1i1p1_195101-200512_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r2i1p1_185001-190012_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r2i1p1_190101-195012_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r2i1p1_195101-200512_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r3i1p1_185001-190012_tmp_yearmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_historical_r3i1p1_190101-195012_tmp_yearmean.nc
Sat Oct 19 17:57:19 2019: cdo yearmean /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_seltime.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_yearmean.nc
Sat Oct 19 17:57:19 2019: cdo selseas,JJA /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_fldmean.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_seltime.nc
Sat Oct 19 17:57:19 2019: cdo fldmean /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_var.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_fldmean.nc
Sat Oct 19 17:57:19 2019: cdo ifthen /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/land_sea_mask.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_var.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_var_mask.nc
Sat Oct 19 17:57:17 2019: cdo selname,tas /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_box.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_var.nc
Sat Oct 19 17:57:17 2019: cdo sellonlatbox,-5,10,42,51 /bdd/CMIP5/main/NASA-GISS/GISS-E2-H/rcp85/mon/atmos/Amon/r2i1p1/latest/tas/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012.nc /data/yrobin//tmp/EUPHEME_HW03_rcp85.sh585268/tas_Amon_GISS-E2-H_rcp85_r2i1p1_205101-210012_tmp_box.nc
2014-02-18T19:29:08Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.   source        0GISS-E2-H-Eh135f9a Atmosphere: GISS-E2; Ocean: H   institution       <NASA/GISS (Goddard Institute for Space Studies) New York, NY   institute_id      	NASA-GISS      experiment_id         
historical     model_id      	GISS-E2-H      forcing       ^GHG, LU, Sl, Vl, BC, OC, SA, Oz (also includes orbital change - BC on snow - Nitrate aerosols)     parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @¢Ô        contact        Kenneth Lo (cdkkl@giss.nasa.gov)   
references        #www.giss.nasa.gov/research/modeling    initialization_method               physics_version             tracking_id       $f8d45b41-e94a-45cb-a248-1bca028fa256   product       output     
experiment        
historical     	frequency         mon    creation_date         2011-04-12T14:27:26Z   
project_id        CMIP5      table_id      =Table Amon (31 January 2011) 53b766a395ac41696af40aab76a49ae5      title         4GISS-E2-H model output prepared for CMIP5 historical   parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.5.7      CDO       @Climate Data Operators version 1.6.8 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           ¨   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           °   time               standard_name         time   	long_name         time   bounds        	time_bnds      units         years since 0-01-01 00:00:00   calendar      365_day         ¸   	time_bnds                            À   tas                    	   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `­xì   missing_value         `­xì   original_name         dummy      cell_methods      time: mean     history       º2011-04-12T14:27:26Z altered by CMOR: Treated scalar dimension: 'height'. 2011-04-12T14:27:26Z altered by CMOR: replaced missing value flag (-1e+30) with standard missing value (1e+20).      associated_files      ­baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_GISS-E2-H_historical_r0i0p0.nc areacella: areacella_fx_GISS-E2-H_historical_r0i0p0.nc            Ð                @ê*ªªª«@éªªªª«@êªªªª«Cî@ê*ªªª«@éªªªª«@êªªªª«Cµÿ@ê*ªªª«@éªªªª«@êªªªª«C0²@ê*ªªª«@éªªªª«@êªªªª«Cä@ê*ªªª«@éªªªª«@êªªªª«CK`@ê*ªªª«@éªªªª«@êªªªª«CD@î*ªªª«@íªªªª«@îªªªª«C­J@î*ªªª«@íªªªª«@îªªªª«Cù²@î*ªªª«@íªªªª«@îªªªª«CbÂ@î*ªªª«@íªªªª«@îªªªª«CWF@î*ªªª«@íªªªª«@îªªªª«Cÿ@î*ªªª«@íªªªª«@îªªªª«Co@ò*ªªª«@ñªªªª«@òªªªª«C)ï@ò*ªªª«@ñªªªª«@òªªªª«Cì @ò*ªªª«@ñªªªª«@òªªªª«Ch¯@ò*ªªª«@ñªªªª«@òªªªª«CÞÍ@ò*ªªª«@ñªªªª«@òªªªª«C8«@ò*ªªª«@ñªªªª«@òªªªª«CU?@ö*ªªª«@õªªªª«@öªªªª«CY@ö*ªªª«@õªªªª«@öªªªª«CÃA@ö*ªªª«@õªªªª«@öªªªª«C%@ö*ªªª«@õªªªª«@öªªªª«C@ö*ªªª«@õªªªª«@öªªªª«C[ã@ö*ªªª«@õªªªª«@öªªªª«CEU@ú*ªªª«@ùªªªª«@úªªªª«CT0@ú*ªªª«@ùªªªª«@úªªªª«C8Ü@ú*ªªª«@ùªªªª«@úªªªª«Cö@ú*ªªª«@ùªªªª«@úªªªª«C=ý@ú*ªªª«@ùªªªª«@úªªªª«Cìð@ú*ªªª«@ùªªªª«@úªªªª«CÔ@þ*ªªª«@ýªªªª«@þªªªª«C´ñ@þ*ªªª«@ýªªªª«@þªªªª«Cg@þ*ªªª«@ýªªªª«@þªªªª«Cèð@þ*ªªª«@ýªªªª«@þªªªª«C @þ*ªªª«@ýªªªª«@þªªªª«CS@þ*ªªª«@ýªªªª«@þªªªª«C¿Û@*ªªª«@ªªªª«@ªªªª«C{	@*ªªª«@ªªªª«@ªªªª«C"V@*ªªª«@ªªªª«@ªªªª«CD@*ªªª«@ªªªª«@ªªªª«Cc@*ªªª«@ªªªª«@ªªªª«C6ð@*ªªª«@ªªªª«@ªªªª«C9N@*ªªª«@ªªªª«@ªªªª«C×J@*ªªª«@ªªªª«@ªªªª«CIR@*ªªª«@ªªªª«@ªªªª«CèÙ@*ªªª«@ªªªª«@ªªªª«C­Ã@*ªªª«@ªªªª«@ªªªª«CHJ@*ªªª«@ªªªª«@ªªªª«C"@
*ªªª«@	ªªªª«@
ªªªª«C^«@
*ªªª«@	ªªªª«@
ªªªª«CD8@
*ªªª«@	ªªªª«@
ªªªª«C`@
*ªªª«@	ªªªª«@
ªªªª«CbÍ@
*ªªª«@	ªªªª«@
ªªªª«Cýó@
*ªªª«@	ªªªª«@
ªªªª«CÛu@*ªªª«@ªªªª«@ªªªª«CÌ?@*ªªª«@ªªªª«@ªªªª«CKÚ@*ªªª«@ªªªª«@ªªªª«Cu@*ªªª«@ªªªª«@ªªªª«C9ø@*ªªª«@ªªªª«@ªªªª«C{Ù@*ªªª«@ªªªª«@ªªªª«C·@*ªªª«@ªªªª«@ªªªª«CØ@*ªªª«@ªªªª«@ªªªª«Cc@*ªªª«@ªªªª«@ªªªª«C @*ªªª«@ªªªª«@ªªªª«CS@*ªªª«@ªªªª«@ªªªª«CG @*ªªª«@ªªªª«@ªªªª«CcY@*ªªª«@ªªªª«@ªªªª«CG»@*ªªª«@ªªªª«@ªªªª«C4C@*ªªª«@ªªªª«@ªªªª«Cü#@*ªªª«@ªªªª«@ªªªª«C¼á@*ªªª«@ªªªª«@ªªªª«CP@*ªªª«@ªªªª«@ªªªª«CQå@*ªªª«@ªªªª«@ªªªª«CW@*ªªª«@ªªªª«@ªªªª«C¢@*ªªª«@ªªªª«@ªªªª«C£æ@*ªªª«@ªªªª«@ªªªª«CÍÔ@*ªªª«@ªªªª«@ªªªª«CÈò@*ªªª«@ªªªª«@ªªªª«Cï@*ªªª«@ªªªª«@ªªªª«CÝi@*ªªª«@ªªªª«@ªªªª«CÜç@*ªªª«@ªªªª«@ªªªª«C¼ë@*ªªª«@ªªªª«@ªªªª«CML@*ªªª«@ªªªª«@ªªªª«CHM@*ªªª«@ªªªª«@ªªªª«C|Ð@"*ªªª«@!ªªªª«@"ªªªª«Cà@"*ªªª«@!ªªªª«@"ªªªª«CÌú@"*ªªª«@!ªªªª«@"ªªªª«C6÷@"*ªªª«@!ªªªª«@"ªªªª«C¿@"*ªªª«@!ªªªª«@"ªªªª«CÜU@"*ªªª«@!ªªªª«@"ªªªª«C ô@&*ªªª«@%ªªªª«@&ªªªª«Cke@&*ªªª«@%ªªªª«@&ªªªª«CÕg@&*ªªª«@%ªªªª«@&ªªªª«C@&*ªªª«@%ªªªª«@&ªªªª«Cþ@&*ªªª«@%ªªªª«@&ªªªª«C¤±@&*ªªª«@%ªªªª«@&ªªªª«C2)@**ªªª«@)ªªªª«@*ªªªª«CßÈ@**ªªª«@)ªªªª«@*ªªªª«CNõ@**ªªª«@)ªªªª«@*ªªªª«C£ì@**ªªª«@)ªªªª«@*ªªªª«C¸f@**ªªª«@)ªªªª«@*ªªªª«Cªz@**ªªª«@)ªªªª«@*ªªªª«Cg*@.*ªªª«@-ªªªª«@.ªªªª«CV*@.*ªªª«@-ªªªª«@.ªªªª«CÒ@.*ªªª«@-ªªªª«@.ªªªª«Cõ@.*ªªª«@-ªªªª«@.ªªªª«Cï@.*ªªª«@-ªªªª«@.ªªªª«C*·@.*ªªª«@-ªªªª«@.ªªªª«C @2*ªªª«@1ªªªª«@2ªªªª«CLM@2*ªªª«@1ªªªª«@2ªªªª«Côÿ@2*ªªª«@1ªªªª«@2ªªªª«Cs¬@2*ªªª«@1ªªªª«@2ªªªª«C¥.@2*ªªª«@1ªªªª«@2ªªªª«Cî&@2*ªªª«@1ªªªª«@2ªªªª«Cp£@6*ªªª«@5ªªªª«@6ªªªª«CÕ@6*ªªª«@5ªªªª«@6ªªªª«Cê@6*ªªª«@5ªªªª«@6ªªªª«CÍ@6*ªªª«@5ªªªª«@6ªªªª«C@6*ªªª«@5ªªªª«@6ªªªª«CK6@6*ªªª«@5ªªªª«@6ªªªª«Ct@:*ªªª«@9ªªªª«@:ªªªª«Cÿ@:*ªªª«@9ªªªª«@:ªªªª«C!@:*ªªª«@9ªªªª«@:ªªªª«C^$@:*ªªª«@9ªªªª«@:ªªªª«Cþ@:*ªªª«@9ªªªª«@:ªªªª«Cc'@:*ªªª«@9ªªªª«@:ªªªª«C­@>*ªªª«@=ªªªª«@>ªªªª«CW8@>*ªªª«@=ªªªª«@>ªªªª«C&k@>*ªªª«@=ªªªª«@>ªªªª«Cì@>*ªªª«@=ªªªª«@>ªªªª«CïÊ@>*ªªª«@=ªªªª«@>ªªªª«C\@>*ªªª«@=ªªªª«@>ªªªª«Cùo@B*ªªª«@Aªªªª«@Bªªªª«Ch@B*ªªª«@Aªªªª«@Bªªªª«C@B*ªªª«@Aªªªª«@Bªªªª«C5@B*ªªª«@Aªªªª«@Bªªªª«Cùg@B*ªªª«@Aªªªª«@Bªªªª«C1P@B*ªªª«@Aªªªª«@Bªªªª«Ck@F*ªªª«@Eªªªª«@Fªªªª«C¿@F*ªªª«@Eªªªª«@Fªªªª«CPË@F*ªªª«@Eªªªª«@Fªªªª«CÖG@F*ªªª«@Eªªªª«@Fªªªª«CÙö@F*ªªª«@Eªªªª«@Fªªªª«CrÍ@F*ªªª«@Eªªªª«@Fªªªª«Ce£@J*ªªª«@Iªªªª«@Jªªªª«CÙk@J*ªªª«@Iªªªª«@Jªªªª«Cÿ´@J*ªªª«@Iªªªª«@Jªªªª«CÑÊ@J*ªªª«@Iªªªª«@Jªªªª«C.ú@J*ªªª«@Iªªªª«@Jªªªª«CIµ@J*ªªª«@Iªªªª«@Jªªªª«C@@N*ªªª«@Mªªªª«@Nªªªª«Ca­@N*ªªª«@Mªªªª«@Nªªªª«C;ë@N*ªªª«@Mªªªª«@Nªªªª«Cz6@N*ªªª«@Mªªªª«@Nªªªª«CGþ@N*ªªª«@Mªªªª«@Nªªªª«Cl§@N*ªªª«@Mªªªª«@Nªªªª«CÊ@R*ªªª«@Qªªªª«@Rªªªª«CÇ@R*ªªª«@Qªªªª«@Rªªªª«C·ý@R*ªªª«@Qªªªª«@Rªªªª«CFC@R*ªªª«@Qªªªª«@Rªªªª«C#@R*ªªª«@Qªªªª«@Rªªªª«C>1@R*ªªª«@Qªªªª«@Rªªªª«C
$@V*ªªª«@Uªªªª«@Vªªªª«C@V*ªªª«@Uªªªª«@Vªªªª«C×@@V*ªªª«@Uªªªª«@Vªªªª«CÆÆ@V*ªªª«@Uªªªª«@Vªªªª«Ci£@V*ªªª«@Uªªªª«@Vªªªª«Cba@V*ªªª«@Uªªªª«@Vªªªª«C¢@Z*ªªª«@Yªªªª«@Zªªªª«C,E@Z*ªªª«@Yªªªª«@Zªªªª«Cí§@Z*ªªª«@Yªªªª«@Zªªªª«CFã@Z*ªªª«@Yªªªª«@Zªªªª«Cy@Z*ªªª«@Yªªªª«@Zªªªª«Cx@Z*ªªª«@Yªªªª«@Zªªªª«C¢@^*ªªª«@]ªªªª«@^ªªªª«C>Z@^*ªªª«@]ªªªª«@^ªªªª«Cª@^*ªªª«@]ªªªª«@^ªªªª«C °@^*ªªª«@]ªªªª«@^ªªªª«Cp²@^*ªªª«@]ªªªª«@^ªªªª«CûJ@^*ªªª«@]ªªªª«@^ªªªª«CÜ@b*ªªª«@aªªªª«@bªªªª«C\@b*ªªª«@aªªªª«@bªªªª«CóA@b*ªªª«@aªªªª«@bªªªª«Cü@b*ªªª«@aªªªª«@bªªªª«C,@b*ªªª«@aªªªª«@bªªªª«C®ç@b*ªªª«@aªªªª«@bªªªª«CÓ/@f*ªªª«@eªªªª«@fªªªª«CG>@f*ªªª«@eªªªª«@fªªªª«C74@f*ªªª«@eªªªª«@fªªªª«C2k@f*ªªª«@eªªªª«@fªªªª«C¿@f*ªªª«@eªªªª«@fªªªª«Ck@f*ªªª«@eªªªª«@fªªªª«Cc@j*ªªª«@iªªªª«@jªªªª«CÁ@j*ªªª«@iªªªª«@jªªªª«Cô@j*ªªª«@iªªªª«@jªªªª«CíÁ@j*ªªª«@iªªªª«@jªªªª«CûÈ@j*ªªª«@iªªªª«@jªªªª«C@j*ªªª«@iªªªª«@jªªªª«Cé@n*ªªª«@mªªªª«@nªªªª«C @n*ªªª«@mªªªª«@nªªªª«Cý@n*ªªª«@mªªªª«@nªªªª«Cv@n*ªªª«@mªªªª«@nªªªª«CT@n*ªªª«@mªªªª«@nªªªª«Cr@n*ªªª«@mªªªª«@nªªªª«C=A@r*ªªª«@qªªªª«@rªªªª«C@r*ªªª«@qªªªª«@rªªªª«C#@r*ªªª«@qªªªª«@rªªªª«Cp8@r*ªªª«@qªªªª«@rªªªª«Cûò@r*ªªª«@qªªªª«@rªªªª«C'@r*ªªª«@qªªªª«@rªªªª«Cwó@v*ªªª«@uªªªª«@vªªªª«C'@v*ªªª«@uªªªª«@vªªªª«CæØ@v*ªªª«@uªªªª«@vªªªª«CÞo@v*ªªª«@uªªªª«@vªªªª«C1@v*ªªª«@uªªªª«@vªªªª«CrN@v*ªªª«@uªªªª«@vªªªª«C¯n@z*ªªª«@yªªªª«@zªªªª«C8å@z*ªªª«@yªªªª«@zªªªª«CTß@z*ªªª«@yªªªª«@zªªªª«C¶@z*ªªª«@yªªªª«@zªªªª«C@z*ªªª«@yªªªª«@zªªªª«Cj@z*ªªª«@yªªªª«@zªªªª«CÍÓ@~*ªªª«@}ªªªª«@~ªªªª«C	@~*ªªª«@}ªªªª«@~ªªªª«Câ@~*ªªª«@}ªªªª«@~ªªªª«C@~*ªªª«@}ªªªª«@~ªªªª«C¶ª@~*ªªª«@}ªªªª«@~ªªªª«Cö@~*ªªª«@}ªªªª«@~ªªªª«CG@*ªªª«@ªªªª«@ªªªª«C3@*ªªª«@ªªªª«@ªªªª«CãÛ@*ªªª«@ªªªª«@ªªªª«C;å@*ªªª«@ªªªª«@ªªªª«CP]@*ªªª«@ªªªª«@ªªªª«CÉj@*ªªª«@ªªªª«@ªªªª«CÊ£@*ªªª«@ªªªª«@ªªªª«Cáy@*ªªª«@ªªªª«@ªªªª«CÓm@*ªªª«@ªªªª«@ªªªª«Cû@*ªªª«@ªªªª«@ªªªª«Cÿ@*ªªª«@ªªªª«@ªªªª«Cò@*ªªª«@ªªªª«@ªªªª«C[@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«CF@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«C°1@*ªªª«@ªªªª«@ªªªª«C½ï@*ªªª«@ªªªª«@ªªªª«C$@*ªªª«@ªªªª«@ªªªª«CsJ@*ªªª«@ªªªª«@ªªªª«Cv`@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«CS+@*ªªª«@ªªªª«@ªªªª«C]@*ªªª«@ªªªª«@ªªªª«C5@*ªªª«@ªªªª«@ªªªª«C/@*ªªª«@ªªªª«@ªªªª«C³@*ªªª«@ªªªª«@ªªªª«C	k@*ªªª«@ªªªª«@ªªªª«C#$@*ªªª«@ªªªª«@ªªªª«C{¹@*ªªª«@ªªªª«@ªªªª«C§@*ªªª«@ªªªª«@ªªªª«C'õ@*ªªª«@ªªªª«@ªªªª«Cq@*ªªª«@ªªªª«@ªªªª«C0Ï@*ªªª«@ªªªª«@ªªªª«C}@*ªªª«@ªªªª«@ªªªª«C ¦@*ªªª«@ªªªª«@ªªªª«Cù@*ªªª«@ªªªª«@ªªªª«Ck@*ªªª«@ªªªª«@ªªªª«C^E@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«CÃÝ@*ªªª«@ªªªª«@ªªªª«Cù@*ªªª«@ªªªª«@ªªªª«C´@*ªªª«@ªªªª«@ªªªª«Cgo@*ªªª«@ªªªª«@ªªªª«C_@*ªªª«@ªªªª«@ªªªª«Ce @*ªªª«@ªªªª«@ªªªª«CÌÙ@*ªªª«@ªªªª«@ªªªª«CWÚ@*ªªª«@ªªªª«@ªªªª«Cq@¢*ªªª«@¡ªªªª«@¢ªªªª«C %@¢*ªªª«@¡ªªªª«@¢ªªªª«C×@¢*ªªª«@¡ªªªª«@¢ªªªª«C×@¢*ªªª«@¡ªªªª«@¢ªªªª«Cä@¢*ªªª«@¡ªªªª«@¢ªªªª«C®=@¢*ªªª«@¡ªªªª«@¢ªªªª«Cít@¦*ªªª«@¥ªªªª«@¦ªªªª«CAÂ@¦*ªªª«@¥ªªªª«@¦ªªªª«Cu@¦*ªªª«@¥ªªªª«@¦ªªªª«C<Õ@¦*ªªª«@¥ªªªª«@¦ªªªª«C%Ü@¦*ªªª«@¥ªªªª«@¦ªªªª«CÞ·@¦*ªªª«@¥ªªªª«@¦ªªªª«CÊ@ª*ªªª«@©ªªªª«@ªªªªª«C@|@ª*ªªª«@©ªªªª«@ªªªªª«Cì@ª*ªªª«@©ªªªª«@ªªªªª«CN@ª*ªªª«@©ªªªª«@ªªªªª«C¸@ª*ªªª«@©ªªªª«@ªªªªª«CAi@ª*ªªª«@©ªªªª«@ªªªªª«C@ñ@®*ªªª«@­ªªªª«@®ªªªª«Cö	@®*ªªª«@­ªªªª«@®ªªªª«C@®*ªªª«@­ªªªª«@®ªªªª«Cy@®*ªªª«@­ªªªª«@®ªªªª«CP-@®*ªªª«@­ªªªª«@®ªªªª«C6@®*ªªª«@­ªªªª«@®ªªªª«CB@²*ªªª«@±ªªªª«@²ªªªª«Cç@²*ªªª«@±ªªªª«@²ªªªª«CX3@²*ªªª«@±ªªªª«@²ªªªª«C/²@²*ªªª«@±ªªªª«@²ªªªª«Cnþ@²*ªªª«@±ªªªª«@²ªªªª«Cîµ@²*ªªª«@±ªªªª«@²ªªªª«CÙ7@¶*ªªª«@µªªªª«@¶ªªªª«C°]@¶*ªªª«@µªªªª«@¶ªªªª«C·@¶*ªªª«@µªªªª«@¶ªªªª«C\ò@¶*ªªª«@µªªªª«@¶ªªªª«CÅ@¶*ªªª«@µªªªª«@¶ªªªª«CÍ@¶*ªªª«@µªªªª«@¶ªªªª«CçÄ@º*ªªª«@¹ªªªª«@ºªªªª«CA@º*ªªª«@¹ªªªª«@ºªªªª«C'P@º*ªªª«@¹ªªªª«@ºªªªª«CóF@º*ªªª«@¹ªªªª«@ºªªªª«C@º*ªªª«@¹ªªªª«@ºªªªª«CoÑ@º*ªªª«@¹ªªªª«@ºªªªª«C&½@¾*ªªª«@½ªªªª«@¾ªªªª«C"@¾*ªªª«@½ªªªª«@¾ªªªª«C@¾*ªªª«@½ªªªª«@¾ªªªª«C@¾*ªªª«@½ªªªª«@¾ªªªª«CD@¾*ªªª«@½ªªªª«@¾ªªªª«CP@@¾*ªªª«@½ªªªª«@¾ªªªª«C¹X@Â*ªªª«@Áªªªª«@Âªªªª«C¢@Â*ªªª«@Áªªªª«@Âªªªª«CÝ@Â*ªªª«@Áªªªª«@Âªªªª«C-a@Â*ªªª«@Áªªªª«@Âªªªª«CÁÏ@Â*ªªª«@Áªªªª«@Âªªªª«CÄ@Â*ªªª«@Áªªªª«@Âªªªª«C(x@Æ*ªªª«@Åªªªª«@Æªªªª«C @Æ*ªªª«@Åªªªª«@Æªªªª«CAÞ@Æ*ªªª«@Åªªªª«@Æªªªª«Cðd@Æ*ªªª«@Åªªªª«@Æªªªª«C@Æ*ªªª«@Åªªªª«@Æªªªª«Cwn@Æ*ªªª«@Åªªªª«@Æªªªª«Ct@Ê*ªªª«@Éªªªª«@Êªªªª«C©@Ê*ªªª«@Éªªªª«@Êªªªª«Ca@Ê*ªªª«@Éªªªª«@Êªªªª«C=@Ê*ªªª«@Éªªªª«@Êªªªª«CÕE@Ê*ªªª«@Éªªªª«@Êªªªª«CÜ@Ê*ªªª«@Éªªªª«@Êªªªª«C@Î*ªªª«@Íªªªª«@Îªªªª«Cp7@Î*ªªª«@Íªªªª«@Îªªªª«Cæ@Î*ªªª«@Íªªªª«@Îªªªª«Cè*@Î*ªªª«@Íªªªª«@Îªªªª«C_Ø@Î*ªªª«@Íªªªª«@Îªªªª«CZ@Î*ªªª«@Íªªªª«@Îªªªª«Cb@Ò*ªªª«@Ñªªªª«@Òªªªª«CÙ@Ò*ªªª«@Ñªªªª«@Òªªªª«Cl¡@Ò*ªªª«@Ñªªªª«@Òªªªª«Cu@Ò*ªªª«@Ñªªªª«@Òªªªª«CØS@Ò*ªªª«@Ñªªªª«@Òªªªª«C @Ò*ªªª«@Ñªªªª«@Òªªªª«CÏ@Ö*ªªª«@Õªªªª«@Öªªªª«Cè&@Ö*ªªª«@Õªªªª«@Öªªªª«C(@Ö*ªªª«@Õªªªª«@Öªªªª«CJø@Ö*ªªª«@Õªªªª«@Öªªªª«CÍ@Ö*ªªª«@Õªªªª«@Öªªªª«C^@Ö*ªªª«@Õªªªª«@Öªªªª«Cª@Ú*ªªª«@Ùªªªª«@Úªªªª«CÍ@Ú*ªªª«@Ùªªªª«@Úªªªª«Cj@Ú*ªªª«@Ùªªªª«@Úªªªª«Ccû@Ú*ªªª«@Ùªªªª«@Úªªªª«C'×@Ú*ªªª«@Ùªªªª«@Úªªªª«C9@Ú*ªªª«@Ùªªªª«@Úªªªª«C¤M@Þ*ªªª«@Ýªªªª«@Þªªªª«C8Ò@Þ*ªªª«@Ýªªªª«@Þªªªª«CÃj@Þ*ªªª«@Ýªªªª«@Þªªªª«Cï8@Þ*ªªª«@Ýªªªª«@Þªªªª«Cä®@Þ*ªªª«@Ýªªªª«@Þªªªª«C	@Þ*ªªª«@Ýªªªª«@Þªªªª«C¶Ø@â*ªªª«@áªªªª«@âªªªª«Cæ#@â*ªªª«@áªªªª«@âªªªª«CD@â*ªªª«@áªªªª«@âªªªª«CÜ@â*ªªª«@áªªªª«@âªªªª«Cµö@â*ªªª«@áªªªª«@âªªªª«C­£@â*ªªª«@áªªªª«@âªªªª«Cã@æ*ªªª«@åªªªª«@æªªªª«C @æ*ªªª«@åªªªª«@æªªªª«CÄ@æ*ªªª«@åªªªª«@æªªªª«CcQ@æ*ªªª«@åªªªª«@æªªªª«Cms@æ*ªªª«@åªªªª«@æªªªª«CA@æ*ªªª«@åªªªª«@æªªªª«C2è@ê*ªªª«@éªªªª«@êªªªª«Cõ§@ê*ªªª«@éªªªª«@êªªªª«Cñ7@ê*ªªª«@éªªªª«@êªªªª«C±ó@ê*ªªª«@éªªªª«@êªªªª«Cín@ê*ªªª«@éªªªª«@êªªªª«C°+@ê*ªªª«@éªªªª«@êªªªª«CÂµ@î*ªªª«@íªªªª«@îªªªª«C@î*ªªª«@íªªªª«@îªªªª«CK@î*ªªª«@íªªªª«@îªªªª«C§@î*ªªª«@íªªªª«@îªªªª«Cø@î*ªªª«@íªªªª«@îªªªª«Cº@î*ªªª«@íªªªª«@îªªªª«C@ò*ªªª«@ñªªªª«@òªªªª«Cì@ò*ªªª«@ñªªªª«@òªªªª«C¥@ò*ªªª«@ñªªªª«@òªªªª«Cðû@ò*ªªª«@ñªªªª«@òªªªª«C@ò*ªªª«@ñªªªª«@òªªªª«Ci£@ò*ªªª«@ñªªªª«@òªªªª«CÍ@ö*ªªª«@õªªªª«@öªªªª«CH×@ö*ªªª«@õªªªª«@öªªªª«CáÊ@ö*ªªª«@õªªªª«@öªªªª«C¯@ö*ªªª«@õªªªª«@öªªªª«CÒo@ö*ªªª«@õªªªª«@öªªªª«C¨@ö*ªªª«@õªªªª«@öªªªª«Cå@ú*ªªª«@ùªªªª«@úªªªª«C¬Î@ú*ªªª«@ùªªªª«@úªªªª«CJÄ@ú*ªªª«@ùªªªª«@úªªªª«CpN@ú*ªªª«@ùªªªª«@úªªªª«CJ@ú*ªªª«@ùªªªª«@úªªªª«CD*@ú*ªªª«@ùªªªª«@úªªªª«CEÅ@þ*ªªª«@ýªªªª«@þªªªª«C Ì@þ*ªªª«@ýªªªª«@þªªªª«C@þ*ªªª«@ýªªªª«@þªªªª«CZ@þ*ªªª«@ýªªªª«@þªªªª«C-3@þ*ªªª«@ýªªªª«@þªªªª«CzB@þ*ªªª«@ýªªªª«@þªªªª«C-<@*ªªª«@ªªªª«@ªªªª«C5¡@*ªªª«@ªªªª«@ªªªª«CQ@*ªªª«@ªªªª«@ªªªª«Cý/@*ªªª«@ªªªª«@ªªªª«C¬O@*ªªª«@ªªªª«@ªªªª«C]K@*ªªª«@ªªªª«@ªªªª«C2Â@*ªªª«@ªªªª«@ªªªª«CE´@*ªªª«@ªªªª«@ªªªª«C}u@*ªªª«@ªªªª«@ªªªª«CLº@*ªªª«@ªªªª«@ªªªª«C6ï@*ªªª«@ªªªª«@ªªªª«CzV@*ªªª«@ªªªª«@ªªªª«C®D@
*ªªª«@	ªªªª«@
ªªªª«C·a@
*ªªª«@	ªªªª«@
ªªªª«C¿x@
*ªªª«@	ªªªª«@
ªªªª«CÙh@
*ªªª«@	ªªªª«@
ªªªª«C5@
*ªªª«@	ªªªª«@
ªªªª«C³æ@
*ªªª«@	ªªªª«@
ªªªª«Cü@*ªªª«@ªªªª«@ªªªª«C§@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«C²@*ªªª«@ªªªª«@ªªªª«CA&@*ªªª«@ªªªª«@ªªªª«CZ@*ªªª«@ªªªª«@ªªªª«Côû@*ªªª«@ªªªª«@ªªªª«CMî@*ªªª«@ªªªª«@ªªªª«C	·@*ªªª«@ªªªª«@ªªªª«C(þ@*ªªª«@ªªªª«@ªªªª«Cg»@*ªªª«@ªªªª«@ªªªª«C!×@*ªªª«@ªªªª«@ªªªª«C.?@*ªªª«@ªªªª«@ªªªª«CC.@*ªªª«@ªªªª«@ªªªª«CÜq@*ªªª«@ªªªª«@ªªªª«C¼@*ªªª«@ªªªª«@ªªªª«Cä@*ªªª«@ªªªª«@ªªªª«Câ@*ªªª«@ªªªª«@ªªªª«CW@*ªªª«@ªªªª«@ªªªª«Cò!@*ªªª«@ªªªª«@ªªªª«CÊ@*ªªª«@ªªªª«@ªªªª«Ca@*ªªª«@ªªªª«@ªªªª«CQf@*ªªª«@ªªªª«@ªªªª«C@`@*ªªª«@ªªªª«@ªªªª«Câ@*ªªª«@ªªªª«@ªªªª«CÕ@*ªªª«@ªªªª«@ªªªª«Cgè@*ªªª«@ªªªª«@ªªªª«C¦·@*ªªª«@ªªªª«@ªªªª«CöÊ@*ªªª«@ªªªª«@ªªªª«C!,@*ªªª«@ªªªª«@ªªªª«CK@"*ªªª«@!ªªªª«@"ªªªª«C²ð@"*ªªª«@!ªªªª«@"ªªªª«C>¦@"*ªªª«@!ªªªª«@"ªªªª«C9Ý@"*ªªª«@!ªªªª«@"ªªªª«CúÙ@"*ªªª«@!ªªªª«@"ªªªª«CÚJ@"*ªªª«@!ªªªª«@"ªªªª«CLp@&*ªªª«@%ªªªª«@&ªªªª«Cä@&*ªªª«@%ªªªª«@&ªªªª«Cõ@&*ªªª«@%ªªªª«@&ªªªª«C
"@&*ªªª«@%ªªªª«@&ªªªª«CØ8@&*ªªª«@%ªªªª«@&ªªªª«CÑ@&*ªªª«@%ªªªª«@&ªªªª«C¶@**ªªª«@)ªªªª«@*ªªªª«CU@**ªªª«@)ªªªª«@*ªªªª«C@**ªªª«@)ªªªª«@*ªªªª«CM@**ªªª«@)ªªªª«@*ªªªª«Cf@**ªªª«@)ªªªª«@*ªªªª«C$@**ªªª«@)ªªªª«@*ªªªª«C}@.*ªªª«@-ªªªª«@.ªªªª«Cý»@.*ªªª«@-ªªªª«@.ªªªª«CH@.*ªªª«@-ªªªª«@.ªªªª«Cüû@.*ªªª«@-ªªªª«@.ªªªª«C¤@.*ªªª«@-ªªªª«@.ªªªª«Cû@.*ªªª«@-ªªªª«@.ªªªª«Ciú@2*ªªª«@1ªªªª«@2ªªªª«C@2*ªªª«@1ªªªª«@2ªªªª«CÒw@2*ªªª«@1ªªªª«@2ªªªª«C&È@2*ªªª«@1ªªªª«@2ªªªª«CB@2*ªªª«@1ªªªª«@2ªªªª«CÐq@2*ªªª«@1ªªªª«@2ªªªª«C n@6*ªªª«@5ªªªª«@6ªªªª«Cþ@6*ªªª«@5ªªªª«@6ªªªª«C@6*ªªª«@5ªªªª«@6ªªªª«CN@6*ªªª«@5ªªªª«@6ªªªª«CHd@6*ªªª«@5ªªªª«@6ªªªª«C·©@6*ªªª«@5ªªªª«@6ªªªª«CÏÃ@:*ªªª«@9ªªªª«@:ªªªª«CØd@:*ªªª«@9ªªªª«@:ªªªª«C·@:*ªªª«@9ªªªª«@:ªªªª«CÔ@:*ªªª«@9ªªªª«@:ªªªª«CD
@:*ªªª«@9ªªªª«@:ªªªª«Ce@:*ªªª«@9ªªªª«@:ªªªª«C~@>*ªªª«@=ªªªª«@>ªªªª«Cð*@>*ªªª«@=ªªªª«@>ªªªª«CoÞ@>*ªªª«@=ªªªª«@>ªªªª«C#@>*ªªª«@=ªªªª«@>ªªªª«Co@>*ªªª«@=ªªªª«@>ªªªª«Cú@>*ªªª«@=ªªªª«@>ªªªª«CB@B*ªªª«@Aªªªª«@Bªªªª«C´@B*ªªª«@Aªªªª«@Bªªªª«CÃ@B*ªªª«@Aªªªª«@Bªªªª«C¦K@B*ªªª«@Aªªªª«@Bªªªª«CÁÎ@B*ªªª«@Aªªªª«@Bªªªª«C@Ï@B*ªªª«@Aªªªª«@Bªªªª«C¹@F*ªªª«@Eªªªª«@Fªªªª«Ciq@F*ªªª«@Eªªªª«@Fªªªª«CÁ@F*ªªª«@Eªªªª«@Fªªªª«C@F*ªªª«@Eªªªª«@Fªªªª«C]Æ@F*ªªª«@Eªªªª«@Fªªªª«C	@F*ªªª«@Eªªªª«@Fªªªª«CÞ@J*ªªª«@Iªªªª«@Jªªªª«CRT@J*ªªª«@Iªªªª«@Jªªªª«C¯«@J*ªªª«@Iªªªª«@Jªªªª«CÇ@J*ªªª«@Iªªªª«@Jªªªª«C@J*ªªª«@Iªªªª«@Jªªªª«Cy)@J*ªªª«@Iªªªª«@Jªªªª«C Ñ@N*ªªª«@Mªªªª«@Nªªªª«CHô@N*ªªª«@Mªªªª«@Nªªªª«Cé¸@N*ªªª«@Mªªªª«@Nªªªª«C%·@N*ªªª«@Mªªªª«@Nªªªª«Cec@N*ªªª«@Mªªªª«@Nªªªª«CEø@N*ªªª«@Mªªªª«@Nªªªª«C¥ß@R*ªªª«@Qªªªª«@Rªªªª«C¨ê@R*ªªª«@Qªªªª«@Rªªªª«CÎ5@R*ªªª«@Qªªªª«@Rªªªª«CuË@R*ªªª«@Qªªªª«@Rªªªª«CsÕ@R*ªªª«@Qªªªª«@Rªªªª«C÷K@R*ªªª«@Qªªªª«@Rªªªª«Cÿ@V*ªªª«@Uªªªª«@Vªªªª«C°%@V*ªªª«@Uªªªª«@Vªªªª«CúA@V*ªªª«@Uªªªª«@Vªªªª«CR@V*ªªª«@Uªªªª«@Vªªªª«Cö@V*ªªª«@Uªªªª«@Vªªªª«C)@V*ªªª«@Uªªªª«@Vªªªª«C¥ò@Z*ªªª«@Yªªªª«@Zªªªª«C^i@Z*ªªª«@Yªªªª«@Zªªªª«C#E@Z*ªªª«@Yªªªª«@Zªªªª«C5s@Z*ªªª«@Yªªªª«@Zªªªª«C¬@Z*ªªª«@Yªªªª«@Zªªªª«CNÐ@Z*ªªª«@Yªªªª«@Zªªªª«C@^*ªªª«@]ªªªª«@^ªªªª«C@^*ªªª«@]ªªªª«@^ªªªª«Có¬@^*ªªª«@]ªªªª«@^ªªªª«Cs@^*ªªª«@]ªªªª«@^ªªªª«CÝ@^*ªªª«@]ªªªª«@^ªªªª«CaÕ@^*ªªª«@]ªªªª«@^ªªªª«C¼Æ@b*ªªª«@aªªªª«@bªªªª«C"ö@b*ªªª«@aªªªª«@bªªªª«CÆ@b*ªªª«@aªªªª«@bªªªª«C{®@b*ªªª«@aªªªª«@bªªªª«CO(@b*ªªª«@aªªªª«@bªªªª«C	*@b*ªªª«@aªªªª«@bªªªª«CýL@f*ªªª«@eªªªª«@fªªªª«C×y@f*ªªª«@eªªªª«@fªªªª«C;Õ@f*ªªª«@eªªªª«@fªªªª«C%@f*ªªª«@eªªªª«@fªªªª«C? @f*ªªª«@eªªªª«@fªªªª«C@f*ªªª«@eªªªª«@fªªªª«CÜ"@j*ªªª«@iªªªª«@jªªªª«C+@j*ªªª«@iªªªª«@jªªªª«CT@j*ªªª«@iªªªª«@jªªªª«Cæó@j*ªªª«@iªªªª«@jªªªª«CWC@j*ªªª«@iªªªª«@jªªªª«CÍ@j*ªªª«@iªªªª«@jªªªª«CÜ.@n*ªªª«@mªªªª«@nªªªª«CÞx@n*ªªª«@mªªªª«@nªªªª«CÁ@n*ªªª«@mªªªª«@nªªªª«Cc?@n*ªªª«@mªªªª«@nªªªª«Cç@n*ªªª«@mªªªª«@nªªªª«C0@n*ªªª«@mªªªª«@nªªªª«C+@r*ªªª«@qªªªª«@rªªªª«Clá@r*ªªª«@qªªªª«@rªªªª«CâD@r*ªªª«@qªªªª«@rªªªª«Cº·@r*ªªª«@qªªªª«@rªªªª«Câ@r*ªªª«@qªªªª«@rªªªª«CÓ@r*ªªª«@qªªªª«@rªªªª«C>3@v*ªªª«@uªªªª«@vªªªª«Cûj@v*ªªª«@uªªªª«@vªªªª«C+@v*ªªª«@uªªªª«@vªªªª«C@v*ªªª«@uªªªª«@vªªªª«Cú@v*ªªª«@uªªªª«@vªªªª«Cg@v*ªªª«@uªªªª«@vªªªª«C¬n@z*ªªª«@yªªªª«@zªªªª«C3È@z*ªªª«@yªªªª«@zªªªª«C @z*ªªª«@yªªªª«@zªªªª«Cÿ@z*ªªª«@yªªªª«@zªªªª«C{@z*ªªª«@yªªªª«@zªªªª«Cë@z*ªªª«@yªªªª«@zªªªª«CKd@~*ªªª«@}ªªªª«@~ªªªª«C7ã@~*ªªª«@}ªªªª«@~ªªªª«Cæ¿@~*ªªª«@}ªªªª«@~ªªªª«CÙÌ@~*ªªª«@}ªªªª«@~ªªªª«Cï³@~*ªªª«@}ªªªª«@~ªªªª«C®A@~*ªªª«@}ªªªª«@~ªªªª«Cµ@*ªªª«@ªªªª«@ªªªª«C|$@*ªªª«@ªªªª«@ªªªª«C*@*ªªª«@ªªªª«@ªªªª«Cï@*ªªª«@ªªªª«@ªªªª«Cý@*ªªª«@ªªªª«@ªªªª«CÈØ@*ªªª«@ªªªª«@ªªªª«C6@*ªªª«@ªªªª«@ªªªª«CpË@*ªªª«@ªªªª«@ªªªª«Co@*ªªª«@ªªªª«@ªªªª«Cëá@*ªªª«@ªªªª«@ªªªª«C[@*ªªª«@ªªªª«@ªªªª«C­k@*ªªª«@ªªªª«@ªªªª«C;j@*ªªª«@ªªªª«@ªªªª«C|³@*ªªª«@ªªªª«@ªªªª«Cé\@*ªªª«@ªªªª«@ªªªª«C^-@*ªªª«@ªªªª«@ªªªª«C¥@*ªªª«@ªªªª«@ªªªª«CV@*ªªª«@ªªªª«@ªªªª«Cò-@*ªªª«@ªªªª«@ªªªª«Cö`@*ªªª«@ªªªª«@ªªªª«CPÁ@*ªªª«@ªªªª«@ªªªª«Cg@*ªªª«@ªªªª«@ªªªª«C2@*ªªª«@ªªªª«@ªªªª«Ci@*ªªª«@ªªªª«@ªªªª«Ci¤@*ªªª«@ªªªª«@ªªªª«CáÉ@*ªªª«@ªªªª«@ªªªª«Cf@*ªªª«@ªªªª«@ªªªª«C+j@*ªªª«@ªªªª«@ªªªª«Ca@*ªªª«@ªªªª«@ªªªª«CØæ@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«C 
@*ªªª«@ªªªª«@ªªªª«Cø@*ªªª«@ªªªª«@ªªªª«CÌ1@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«CcX@*ªªª«@ªªªª«@ªªªª«CË®@*ªªª«@ªªªª«@ªªªª«Cæ@*ªªª«@ªªªª«@ªªªª«Ca«@*ªªª«@ªªªª«@ªªªª«C+I@*ªªª«@ªªªª«@ªªªª«Cz4@*ªªª«@ªªªª«@ªªªª«CÆ @*ªªª«@ªªªª«@ªªªª«C@Ú@*ªªª«@ªªªª«@ªªªª«C)@*ªªª«@ªªªª«@ªªªª«C¶@*ªªª«@ªªªª«@ªªªª«CïF@*ªªª«@ªªªª«@ªªªª«CÖ1@*ªªª«@ªªªª«@ªªªª«C×Ä@*ªªª«@ªªªª«@ªªªª«C@¢*ªªª«@¡ªªªª«@¢ªªªª«Cg@¢*ªªª«@¡ªªªª«@¢ªªªª«Cè@¢*ªªª«@¡ªªªª«@¢ªªªª«CÓz@¢*ªªª«@¡ªªªª«@¢ªªªª«CÒ@¢*ªªª«@¡ªªªª«@¢ªªªª«Cã@¢*ªªª«@¡ªªªª«@¢ªªªª«CüÛ@¦*ªªª«@¥ªªªª«@¦ªªªª«C£^@¦*ªªª«@¥ªªªª«@¦ªªªª«Cãm@¦*ªªª«@¥ªªªª«@¦ªªªª«Cõ@¦*ªªª«@¥ªªªª«@¦ªªªª«C8@@¦*ªªª«@¥ªªªª«@¦ªªªª«C9é@¦*ªªª«@¥ªªªª«@¦ªªªª«C.ð@ª*ªªª«@©ªªªª«@ªªªªª«Cbú@ª*ªªª«@©ªªªª«@ªªªªª«Cøµ@ª*ªªª«@©ªªªª«@ªªªªª«C{@ª*ªªª«@©ªªªª«@ªªªªª«C@ª*ªªª«@©ªªªª«@ªªªªª«C&@ª*ªªª«@©ªªªª«@ªªªªª«C@®*ªªª«@­ªªªª«@®ªªªª«C@®*ªªª«@­ªªªª«@®ªªªª«Cwå@®*ªªª«@­ªªªª«@®ªªªª«Còü@®*ªªª«@­ªªªª«@®ªªªª«C4!@®*ªªª«@­ªªªª«@®ªªªª«C1]@®*ªªª«@­ªªªª«@®ªªªª«CR@²*ªªª«@±ªªªª«@²ªªªª«CÄè@²*ªªª«@±ªªªª«@²ªªªª«Cw@²*ªªª«@±ªªªª«@²ªªªª«CH@²*ªªª«@±ªªªª«@²ªªªª«Cç@²*ªªª«@±ªªªª«@²ªªªª«C'é@²*ªªª«@±ªªªª«@²ªªªª«CÐª@¶*ªªª«@µªªªª«@¶ªªªª«C*@¶*ªªª«@µªªªª«@¶ªªªª«C#¡@¶*ªªª«@µªªªª«@¶ªªªª«Câ@¶*ªªª«@µªªªª«@¶ªªªª«CU@¶*ªªª«@µªªªª«@¶ªªªª«CÀ@¶*ªªª«@µªªªª«@¶ªªªª«C4@º*ªªª«@¹ªªªª«@ºªªªª«C/A@º*ªªª«@¹ªªªª«@ºªªªª«CÂ@º*ªªª«@¹ªªªª«@ºªªªª«CFÌ@º*ªªª«@¹ªªªª«@ºªªªª«Cø¦@º*ªªª«@¹ªªªª«@ºªªªª«CÙì@º*ªªª«@¹ªªªª«@ºªªªª«C4@¾*ªªª«@½ªªªª«@¾ªªªª«Cì:@¾*ªªª«@½ªªªª«@¾ªªªª«CHÉ@¾*ªªª«@½ªªªª«@¾ªªªª«C/ø@¾*ªªª«@½ªªªª«@¾ªªªª«C@¾*ªªª«@½ªªªª«@¾ªªªª«C @¾*ªªª«@½ªªªª«@¾ªªªª«C@Â*ªªª«@Áªªªª«@Âªªªª«Cq@Â*ªªª«@Áªªªª«@Âªªªª«C´@Â*ªªª«@Áªªªª«@Âªªªª«CY@Â*ªªª«@Áªªªª«@Âªªªª«Cäz@Â*ªªª«@Áªªªª«@Âªªªª«C	#@Â*ªªª«@Áªªªª«@Âªªªª«C§@Æ*ªªª«@Åªªªª«@Æªªªª«CõF@Æ*ªªª«@Åªªªª«@Æªªªª«CÎ­@Æ*ªªª«@Åªªªª«@Æªªªª«CÎ@Æ*ªªª«@Åªªªª«@Æªªªª«C!©@Æ*ªªª«@Åªªªª«@Æªªªª«C7)@Æ*ªªª«@Åªªªª«@Æªªªª«C¥@Ê*ªªª«@Éªªªª«@Êªªªª«CSq@Ê*ªªª«@Éªªªª«@Êªªªª«Cá@Ê*ªªª«@Éªªªª«@Êªªªª«CXð@Ê*ªªª«@Éªªªª«@Êªªªª«C,ì@Ê*ªªª«@Éªªªª«@Êªªªª«C@Ê*ªªª«@Éªªªª«@Êªªªª«Cn@Î*ªªª«@Íªªªª«@Îªªªª«C{û@Î*ªªª«@Íªªªª«@Îªªªª«C&)@Î*ªªª«@Íªªªª«@Îªªªª«C&í@Î*ªªª«@Íªªªª«@Îªªªª«C×P@Î*ªªª«@Íªªªª«@Îªªªª«Cco@Î*ªªª«@Íªªªª«@Îªªªª«Cûô@Ò*ªªª«@Ñªªªª«@Òªªªª«C@Ò*ªªª«@Ñªªªª«@Òªªªª«C¶O@Ò*ªªª«@Ñªªªª«@Òªªªª«Cí@Ò*ªªª«@Ñªªªª«@Òªªªª«C@Ò*ªªª«@Ñªªªª«@Òªªªª«C@Ò*ªªª«@Ñªªªª«@Òªªªª«CFe@Ö*ªªª«@Õªªªª«@Öªªªª«CÇÇ@Ö*ªªª«@Õªªªª«@Öªªªª«Cè÷@Ö*ªªª«@Õªªªª«@Öªªªª«Cúü@Ö*ªªª«@Õªªªª«@Öªªªª«C9@Ö*ªªª«@Õªªªª«@Öªªªª«CÁ9@Ö*ªªª«@Õªªªª«@Öªªªª«CÊ@Ú*ªªª«@Ùªªªª«@Úªªªª«C)@Ú*ªªª«@Ùªªªª«@Úªªªª«CÏ @Ú*ªªª«@Ùªªªª«@Úªªªª«C~/@Ú*ªªª«@Ùªªªª«@Úªªªª«C`@Ú*ªªª«@Ùªªªª«@Úªªªª«CK@Ú*ªªª«@Ùªªªª«@Úªªªª«CÈh@Þ*ªªª«@Ýªªªª«@Þªªªª«C/.@Þ*ªªª«@Ýªªªª«@Þªªªª«C¹@Þ*ªªª«@Ýªªªª«@Þªªªª«C1@Þ*ªªª«@Ýªªªª«@Þªªªª«Cx@Þ*ªªª«@Ýªªªª«@Þªªªª«Cùè@Þ*ªªª«@Ýªªªª«@Þªªªª«C!@â*ªªª«@áªªªª«@âªªªª«C·@â*ªªª«@áªªªª«@âªªªª«C©@â*ªªª«@áªªªª«@âªªªª«CÌ3@â*ªªª«@áªªªª«@âªªªª«C²!@â*ªªª«@áªªªª«@âªªªª«C®@â*ªªª«@áªªªª«@âªªªª«C=ÿ@æ*ªªª«@åªªªª«@æªªªª«CI@æ*ªªª«@åªªªª«@æªªªª«C¥C@æ*ªªª«@åªªªª«@æªªªª«CKÇ@æ*ªªª«@åªªªª«@æªªªª«CæW@æ*ªªª«@åªªªª«@æªªªª«C@æ*ªªª«@åªªªª«@æªªªª«C+@ê*ªªª«@éªªªª«@êªªªª«C[¼@ê*ªªª«@éªªªª«@êªªªª«Cys@ê*ªªª«@éªªªª«@êªªªª«C=i@ê*ªªª«@éªªªª«@êªªªª«Cê*@ê*ªªª«@éªªªª«@êªªªª«C%¸@ê*ªªª«@éªªªª«@êªªªª«CÒÕ@î*ªªª«@íªªªª«@îªªªª«CÛ6@î*ªªª«@íªªªª«@îªªªª«CÔD@î*ªªª«@íªªªª«@îªªªª«C<¹@î*ªªª«@íªªªª«@îªªªª«C3@î*ªªª«@íªªªª«@îªªªª«CÍõ@î*ªªª«@íªªªª«@îªªªª«C°²@ò*ªªª«@ñªªªª«@òªªªª«Co#@ò*ªªª«@ñªªªª«@òªªªª«Cx@ò*ªªª«@ñªªªª«@òªªªª«Ct@ò*ªªª«@ñªªªª«@òªªªª«C-%@ò*ªªª«@ñªªªª«@òªªªª«C>ª@ò*ªªª«@ñªªªª«@òªªªª«Cî÷@ö*ªªª«@õªªªª«@öªªªª«Cû1@ö*ªªª«@õªªªª«@öªªªª«C@ö*ªªª«@õªªªª«@öªªªª«Cc@ö*ªªª«@õªªªª«@öªªªª«CÓî@ö*ªªª«@õªªªª«@öªªªª«C,6@ö*ªªª«@õªªªª«@öªªªª«CÜö@ú*ªªª«@ùªªªª«@úªªªª«C¥ª@ú*ªªª«@ùªªªª«@úªªªª«C¼û@ú*ªªª«@ùªªªª«@úªªªª«CÍÍ@ú*ªªª«@ùªªªª«@úªªªª«Cæ²@ú*ªªª«@ùªªªª«@úªªªª«Câ@ú*ªªª«@ùªªªª«@úªªªª«C¡@þ*ªªª«@ýªªªª«@þªªªª«CÕ{@þ*ªªª«@ýªªªª«@þªªªª«CÛ¯@þ*ªªª«@ýªªªª«@þªªªª«CHø@þ*ªªª«@ýªªªª«@þªªªª«CÞ@þ*ªªª«@ýªªªª«@þªªªª«C-Ó@þ*ªªª«@ýªªªª«@þªªªª«C}M@*ªªª«@ªªªª«@ªªªª«Ct@*ªªª«@ªªªª«@ªªªª«C"p@*ªªª«@ªªªª«@ªªªª«C9Ð@*ªªª«@ªªªª«@ªªªª«C%q@*ªªª«@ªªªª«@ªªªª«Cj@*ªªª«@ªªªª«@ªªªª«C§Ð@*ªªª«@ªªªª«@ªªªª«C¬7@*ªªª«@ªªªª«@ªªªª«C&ø@*ªªª«@ªªªª«@ªªªª«Cñ @*ªªª«@ªªªª«@ªªªª«C4Q@*ªªª«@ªªªª«@ªªªª«CðÚ@*ªªª«@ªªªª«@ªªªª«Ca@
*ªªª«@	ªªªª«@
ªªªª«CÂW@
*ªªª«@	ªªªª«@
ªªªª«CU@
*ªªª«@	ªªªª«@
ªªªª«C@
*ªªª«@	ªªªª«@
ªªªª«C¼@
*ªªª«@	ªªªª«@
ªªªª«Ca[@
*ªªª«@	ªªªª«@
ªªªª«CIÕ@*ªªª«@ªªªª«@ªªªª«CS¿@*ªªª«@ªªªª«@ªªªª«Cs`@*ªªª«@ªªªª«@ªªªª«Cù¾@*ªªª«@ªªªª«@ªªªª«Cgÿ@*ªªª«@ªªªª«@ªªªª«Cäs@*ªªª«@ªªªª«@ªªªª«CþN@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«C¢@*ªªª«@ªªªª«@ªªªª«C%t@*ªªª«@ªªªª«@ªªªª«C]@*ªªª«@ªªªª«@ªªªª«CD@*ªªª«@ªªªª«@ªªªª«C"ò@*ªªª«@ªªªª«@ªªªª«C:7@*ªªª«@ªªªª«@ªªªª«CL@*ªªª«@ªªªª«@ªªªª«Cø@*ªªª«@ªªªª«@ªªªª«C­@*ªªª«@ªªªª«@ªªªª«CÌb@*ªªª«@ªªªª«@ªªªª«Caã@*ªªª«@ªªªª«@ªªªª«C<ü@*ªªª«@ªªªª«@ªªªª«C~@*ªªª«@ªªªª«@ªªªª«C°Ä@*ªªª«@ªªªª«@ªªªª«C~ù@*ªªª«@ªªªª«@ªªªª«C>å@*ªªª«@ªªªª«@ªªªª«C±@*ªªª«@ªªªª«@ªªªª«CÄ@*ªªª«@ªªªª«@ªªªª«C+@*ªªª«@ªªªª«@ªªªª«Cü@*ªªª«@ªªªª«@ªªªª«C8@*ªªª«@ªªªª«@ªªªª«Cw@*ªªª«@ªªªª«@ªªªª«CUÑ@"*ªªª«@!ªªªª«@"ªªªª«C @"*ªªª«@!ªªªª«@"ªªªª«C?Í@"*ªªª«@!ªªªª«@"ªªªª«C¨×@"*ªªª«@!ªªªª«@"ªªªª«C2@"*ªªª«@!ªªªª«@"ªªªª«C_@"*ªªª«@!ªªªª«@"ªªªª«C¢ñ@&*ªªª«@%ªªªª«@&ªªªª«C==@&*ªªª«@%ªªªª«@&ªªªª«CùÊ@&*ªªª«@%ªªªª«@&ªªªª«Cí@&*ªªª«@%ªªªª«@&ªªªª«Cn@&*ªªª«@%ªªªª«@&ªªªª«Cý¯@&*ªªª«@%ªªªª«@&ªªªª«C=>@**ªªª«@)ªªªª«@*ªªªª«C`¶@**ªªª«@)ªªªª«@*ªªªª«C->@**ªªª«@)ªªªª«@*ªªªª«C*Ã@**ªªª«@)ªªªª«@*ªªªª«C@**ªªª«@)ªªªª«@*ªªªª«C_¸@**ªªª«@)ªªªª«@*ªªªª«C¹>@.*ªªª«@-ªªªª«@.ªªªª«Cª@.*ªªª«@-ªªªª«@.ªªªª«Co\@.*ªªª«@-ªªªª«@.ªªªª«C@.*ªªª«@-ªªªª«@.ªªªª«CzÜ@.*ªªª«@-ªªªª«@.ªªªª«Cü@.*ªªª«@-ªªªª«@.ªªªª«C¾s@2*ªªª«@1ªªªª«@2ªªªª«CA¦@2*ªªª«@1ªªªª«@2ªªªª«CP×@2*ªªª«@1ªªªª«@2ªªªª«CIy@2*ªªª«@1ªªªª«@2ªªªª«C8^@2*ªªª«@1ªªªª«@2ªªªª«C9@2*ªªª«@1ªªªª«@2ªªªª«Cc@6*ªªª«@5ªªªª«@6ªªªª«CÔ@6*ªªª«@5ªªªª«@6ªªªª«C¦;@6*ªªª«@5ªªªª«@6ªªªª«C[%@6*ªªª«@5ªªªª«@6ªªªª«C¿ç@6*ªªª«@5ªªªª«@6ªªªª«C}Þ@6*ªªª«@5ªªªª«@6ªªªª«C7Õ@:*ªªª«@9ªªªª«@:ªªªª«Cñ@:*ªªª«@9ªªªª«@:ªªªª«C(,@:*ªªª«@9ªªªª«@:ªªªª«C^ø@:*ªªª«@9ªªªª«@:ªªªª«C´W@:*ªªª«@9ªªªª«@:ªªªª«CQ§@:*ªªª«@9ªªªª«@:ªªªª«C?g@>*ªªª«@=ªªªª«@>ªªªª«C»ú@>*ªªª«@=ªªªª«@>ªªªª«Cq@>*ªªª«@=ªªªª«@>ªªªª«Cu®@>*ªªª«@=ªªªª«@>ªªªª«C@>*ªªª«@=ªªªª«@>ªªªª«C«@>*ªªª«@=ªªªª«@>ªªªª«Cqf@B*ªªª«@Aªªªª«@Bªªªª«Cs@B*ªªª«@Aªªªª«@Bªªªª«C@B*ªªª«@Aªªªª«@Bªªªª«C @B*ªªª«@Aªªªª«@Bªªªª«C®¿@B*ªªª«@Aªªªª«@Bªªªª«CÓÀ@B*ªªª«@Aªªªª«@Bªªªª«C&Ö@F*ªªª«@Eªªªª«@Fªªªª«C'@F*ªªª«@Eªªªª«@Fªªªª«Ci@F*ªªª«@Eªªªª«@Fªªªª«CEÍ@F*ªªª«@Eªªªª«@Fªªªª«CGÁ@F*ªªª«@Eªªªª«@Fªªªª«Cì^@F*ªªª«@Eªªªª«@Fªªªª«Cb@J*ªªª«@Iªªªª«@Jªªªª«Ciä@J*ªªª«@Iªªªª«@Jªªªª«C@J*ªªª«@Iªªªª«@Jªªªª«Cp@J*ªªª«@Iªªªª«@Jªªªª«C±8@J*ªªª«@Iªªªª«@Jªªªª«CA@J*ªªª«@Iªªªª«@Jªªªª«C%·@N*ªªª«@Mªªªª«@Nªªªª«C:Á@N*ªªª«@Mªªªª«@Nªªªª«C@N*ªªª«@Mªªªª«@Nªªªª«C[E@N*ªªª«@Mªªªª«@Nªªªª«Cv@N*ªªª«@Mªªªª«@Nªªªª«CK*@N*ªªª«@Mªªªª«@Nªªªª«CD©@R*ªªª«@Qªªªª«@Rªªªª«CD@R*ªªª«@Qªªªª«@Rªªªª«C)@R*ªªª«@Qªªªª«@Rªªªª«CÐè@R*ªªª«@Qªªªª«@Rªªªª«C
e@R*ªªª«@Qªªªª«@Rªªªª«Cù1@R*ªªª«@Qªªªª«@Rªªªª«C-+@V*ªªª«@Uªªªª«@Vªªªª«C°9@V*ªªª«@Uªªªª«@Vªªªª«Cq@V*ªªª«@Uªªªª«@Vªªªª«C(+@V*ªªª«@Uªªªª«@Vªªªª«C@V*ªªª«@Uªªªª«@Vªªªª«C(r@V*ªªª«@Uªªªª«@Vªªªª«CF@Z*ªªª«@Yªªªª«@Zªªªª«C«±@Z*ªªª«@Yªªªª«@Zªªªª«CSR@^*ªªª«@]ªªªª«@^ªªªª«C$Ã@^*ªªª«@]ªªªª«@^ªªªª«C¡-@b*ªªª«@aªªªª«@bªªªª«Cû/@b*ªªª«@aªªªª«@bªªªª«Cr1@f*ªªª«@eªªªª«@fªªªª«CÖW@f*ªªª«@eªªªª«@fªªªª«C"¬@j*ªªª«@iªªªª«@jªªªª«Cæ<@j*ªªª«@iªªªª«@jªªªª«Cáh@n*ªªª«@mªªªª«@nªªªª«C^@n*ªªª«@mªªªª«@nªªªª«C%µ@r*ªªª«@qªªªª«@rªªªª«C-@r*ªªª«@qªªªª«@rªªªª«C@v*ªªª«@uªªªª«@vªªªª«C}¹@v*ªªª«@uªªªª«@vªªªª«Cøv@z*ªªª«@yªªªª«@zªªªª«CØ@z*ªªª«@yªªªª«@zªªªª«Cø¶@~*ªªª«@}ªªªª«@~ªªªª«C	@~*ªªª«@}ªªªª«@~ªªªª«C@*ªªª«@ªªªª«@ªªªª«CÒÑ@*ªªª«@ªªªª«@ªªªª«C¼&@*ªªª«@ªªªª«@ªªªª«C%k@*ªªª«@ªªªª«@ªªªª«C¾@@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«CG*@*ªªª«@ªªªª«@ªªªª«CÃÂ@*ªªª«@ªªªª«@ªªªª«C5@*ªªª«@ªªªª«@ªªªª«C¡@*ªªª«@ªªªª«@ªªªª«Cn@*ªªª«@ªªªª«@ªªªª«Cï@*ªªª«@ªªªª«@ªªªª«C@*ªªª«@ªªªª«@ªªªª«CB@*ªªª«@ªªªª«@ªªªª«C*È@*ªªª«@ªªªª«@ªªªª«C£@*ªªª«@ªªªª«@ªªªª«CÎ	@¢*ªªª«@¡ªªªª«@¢ªªªª«C1@¢*ªªª«@¡ªªªª«@¢ªªªª«C@¦*ªªª«@¥ªªªª«@¦ªªªª«C^Ì@¦*ªªª«@¥ªªªª«@¦ªªªª«CUc@ª*ªªª«@©ªªªª«@ªªªªª«C÷h@ª*ªªª«@©ªªªª«@ªªªªª«C«û@®*ªªª«@­ªªªª«@®ªªªª«C5@®*ªªª«@­ªªªª«@®ªªªª«C¸@²*ªªª«@±ªªªª«@²ªªªª«C/@²*ªªª«@±ªªªª«@²ªªªª«C¯F@¶*ªªª«@µªªªª«@¶ªªªª«Cµµ@¶*ªªª«@µªªªª«@¶ªªªª«Cí¹@º*ªªª«@¹ªªªª«@ºªªªª«C/}@º*ªªª«@¹ªªªª«@ºªªªª«C­@¾*ªªª«@½ªªªª«@¾ªªªª«Co§@¾*ªªª«@½ªªªª«@¾ªªªª«CÂQ@Â*ªªª«@Áªªªª«@Âªªªª«C>@Â*ªªª«@Áªªªª«@Âªªªª«C8ì@Æ*ªªª«@Åªªªª«@Æªªªª«CV@Æ*ªªª«@Åªªªª«@Æªªªª«C Ê@Ê*ªªª«@Éªªªª«@Êªªªª«CCI@Ê*ªªª«@Éªªªª«@Êªªªª«C@Î*ªªª«@Íªªªª«@Îªªªª«CXÇ@Î*ªªª«@Íªªªª«@Îªªªª«CJò@Ò*ªªª«@Ñªªªª«@Òªªªª«C3&@Ò*ªªª«@Ñªªªª«@Òªªªª«C`@Ö*ªªª«@Õªªªª«@Öªªªª«C½c@Ö*ªªª«@Õªªªª«@Öªªªª«CÖ&@Ú*ªªª«@Ùªªªª«@Úªªªª«CbG@Ú*ªªª«@Ùªªªª«@Úªªªª«CÄ@Þ*ªªª«@Ýªªªª«@Þªªªª«CÙ¬@Þ*ªªª«@Ýªªªª«@Þªªªª«Cõ@â*ªªª«@áªªªª«@âªªªª«C%!@â*ªªª«@áªªªª«@âªªªª«Cg@æ*ªªª«@åªªªª«@æªªªª«CÔÅ@æ*ªªª«@åªªªª«@æªªªª«C* @ê*ªªª«@éªªªª«@êªªªª«C÷@ê*ªªª«@éªªªª«@êªªªª«Cª@î*ªªª«@íªªªª«@îªªªª«CÍ@î*ªªª«@íªªªª«@îªªªª«Cµí@ò*ªªª«@ñªªªª«@òªªªª«CëÞ@ò*ªªª«@ñªªªª«@òªªªª«C#@ö*ªªª«@õªªªª«@öªªªª«C¶Á@ö*ªªª«@õªªªª«@öªªªª«C@ú*ªªª«@ùªªªª«@úªªªª«CHb@ú*ªªª«@ùªªªª«@úªªªª«C!Â@þ*ªªª«@ýªªªª«@þªªªª«CuÏ@þ*ªªª«@ýªªªª«@þªªªª«C¨<@ UUUU@  ÕUUUU@ UUUUUCÆ@ UUUU@  ÕUUUU@ UUUUUCÏW@ UUUU@ ÕUUUU@ UUUUUCA©@ UUUU@ ÕUUUU@ UUUUUC§Ð@ UUUU@ ÕUUUU@ UUUUUCÝ8@ UUUU@ ÕUUUU@ UUUUUC"@ UUUU@ ÕUUUU@ UUUUUC¹@ UUUU@ ÕUUUU@ UUUUUC²É@ 	UUUU@ ÕUUUU@ 	UUUUUCFÇ@ 	UUUU@ ÕUUUU@ 	UUUUUCjÙ@ UUUU@ 
ÕUUUU@ UUUUUCK@ UUUU@ 
ÕUUUU@ UUUUUCK@ UUUU@ ÕUUUU@ UUUUUC[ @ UUUU@ ÕUUUU@ UUUUUCV@ UUUU@ ÕUUUU@ UUUUUCì²@ UUUU@ ÕUUUU@ UUUUUC?@ UUUU@ ÕUUUU@ UUUUUCÏ³@ UUUU@ ÕUUUU@ UUUUUCM@ UUUU@ ÕUUUU@ UUUUUC°@ UUUU@ ÕUUUU@ UUUUUCÉ@ UUUU@ ÕUUUU@ UUUUUC@ UUUU@ ÕUUUU@ UUUUUC	@ UUUU@ ÕUUUU@ UUUUUCkb@ UUUU@ ÕUUUU@ UUUUUCÿ­@ UUUU@ ÕUUUU@ UUUUUC²@ UUUU@ ÕUUUU@ UUUUUCxü@ UUUU@ ÕUUUU@ UUUUUC@ UUUU@ ÕUUUU@ UUUUUC£@ UUUU@ ÕUUUU@ UUUUUCQp@ UUUU@ ÕUUUU@ UUUUUCs~@ UUUU@ ÕUUUU@ UUUUUC¢@ UUUU@ ÕUUUU@ UUUUUC§@ !UUUU@  ÕUUUU@ !UUUUUC|u@ !UUUU@  ÕUUUU@ !UUUUUCÎ@ #UUUU@ "ÕUUUU@ #UUUUUC(Ý@ #UUUU@ "ÕUUUU@ #UUUUUC'W@ %UUUU@ $ÕUUUU@ %UUUUUCÁ@ %UUUU@ $ÕUUUU@ %UUUUUC,¸@ 'UUUU@ &ÕUUUU@ 'UUUUUCN}@ 'UUUU@ &ÕUUUU@ 'UUUUUCøò@ )UUUU@ (ÕUUUU@ )UUUUUCR@ )UUUU@ (ÕUUUU@ )UUUUUCt1@ +UUUU@ *ÕUUUU@ +UUUUUC)¢@ +UUUU@ *ÕUUUU@ +UUUUUCnÐ@ -UUUU@ ,ÕUUUU@ -UUUUUC÷'@ -UUUU@ ,ÕUUUU@ -UUUUUC8L@ /UUUU@ .ÕUUUU@ /UUUUUCØX@ /UUUU@ .ÕUUUU@ /UUUUUC®ã@ 1UUUU@ 0ÕUUUU@ 1UUUUUC°@ 1UUUU@ 0ÕUUUU@ 1UUUUUC'Ù@ 3UUUU@ 2ÕUUUU@ 3UUUUUCS=@ 3UUUU@ 2ÕUUUU@ 3UUUUUCj.@ 5UUUU@ 4ÕUUUU@ 5UUUUUC°ô@ 5UUUU@ 4ÕUUUU@ 5UUUUUCb@ 7UUUU@ 6ÕUUUU@ 7UUUUUCµÙ@ 7UUUU@ 6ÕUUUU@ 7UUUUUCª@ 9UUUU@ 8ÕUUUU@ 9UUUUUC"Ñ@ 9UUUU@ 8ÕUUUU@ 9UUUUUCl£@ ;UUUU@ :ÕUUUU@ ;UUUUUCW@ ;UUUU@ :ÕUUUU@ ;UUUUUCKE@ =UUUU@ <ÕUUUU@ =UUUUUCÉ5@ =UUUU@ <ÕUUUU@ =UUUUUCú@ ?UUUU@ >ÕUUUU@ ?UUUUUC¾Ø@ ?UUUU@ >ÕUUUU@ ?UUUUUC¾]@ AUUUU@ @ÕUUUU@ AUUUUUC_@ AUUUU@ @ÕUUUU@ AUUUUUC`°@ CUUUU@ BÕUUUU@ CUUUUUC,h@ CUUUU@ BÕUUUU@ CUUUUUC@ EUUUU@ DÕUUUU@ EUUUUUCûW@ EUUUU@ DÕUUUU@ EUUUUUCÕ@ GUUUU@ FÕUUUU@ GUUUUUCµM@ GUUUU@ FÕUUUU@ GUUUUUC@ IUUUU@ HÕUUUU@ IUUUUUCû@ IUUUU@ HÕUUUU@ IUUUUUCÜ@ KUUUU@ JÕUUUU@ KUUUUUCáï@ KUUUU@ JÕUUUU@ KUUUUUCM7@ MUUUU@ LÕUUUU@ MUUUUUC4@ MUUUU@ LÕUUUU@ MUUUUUCå@ OUUUU@ NÕUUUU@ OUUUUUCòë@ OUUUU@ NÕUUUU@ OUUUUUCÖi@ QUUUU@ PÕUUUU@ QUUUUUCt@ QUUUU@ PÕUUUU@ QUUUUUCï2@ SUUUU@ RÕUUUU@ SUUUUUC"ß@ SUUUU@ RÕUUUU@ SUUUUUC©@ UUUUU@ TÕUUUU@ UUUUUUCÚ9@ UUUUU@ TÕUUUU@ UUUUUUCD@ WUUUU@ VÕUUUU@ WUUUUUCÜç@ WUUUU@ VÕUUUU@ WUUUUUCD@ YUUUU@ XÕUUUU@ YUUUUUCú@ YUUUU@ XÕUUUU@ YUUUUUC^ý@ [UUUU@ ZÕUUUU@ [UUUUUC]@ [UUUU@ ZÕUUUU@ [UUUUUC!@ ]UUUU@ \ÕUUUU@ ]UUUUUC@ ]UUUU@ \ÕUUUU@ ]UUUUUC@ _UUUU@ ^ÕUUUU@ _UUUUUC5@ _UUUU@ ^ÕUUUU@ _UUUUUCe§@ aUUUU@ `ÕUUUU@ aUUUUUCE¹@ aUUUU@ `ÕUUUU@ aUUUUUC1#@ cUUUU@ bÕUUUU@ cUUUUUCÊÇ@ cUUUU@ bÕUUUU@ cUUUUUC@ eUUUU@ dÕUUUU@ eUUUUUC@ eUUUU@ dÕUUUU@ eUUUUUC¢¹@ gUUUU@ fÕUUUU@ gUUUUUC'@ gUUUU@ fÕUUUU@ gUUUUUCÛ@ iUUUU@ hÕUUUU@ iUUUUUC0@ iUUUU@ hÕUUUU@ iUUUUUC ;