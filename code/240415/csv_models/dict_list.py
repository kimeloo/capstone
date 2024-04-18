
# 폴더별 전체 컬럼 (train_replaced.csv)
# Administrative
administrative = ['nsrrid']

# Administrative/SHHS1
administrative_shhs1 = ['lang15']

# Anthropometry
anthropometry = ['hip', 'neck20', 'bmi_s1', 'weight', 'waist', 'height', 'weight20']
print(f'anthropometry : {len(anthropometry)}')

# Clinical Data/Diagnostic Studies/Electrocardiogram
clinical_data_electrocardiogram = ['lvh3_1', 'lvh3_3', 'st4_1_3', 'st5_1_3', 'lvhst', 'mob1', 'part2deg', 'mob2', 'av3deg', 'av1deg', 'lbbb', 'rbbb', 'ilbbb', 'irbbb', 'lah', 'iventblk', 'wpw', 'antsepmi', 'infmi', 'antlatmi', 'nonsp_st', 'nonsp_tw', 'rtrial', 'rvh', 'afib', 'paced']
print(f'clinical_data_electrocardiogram : {len(clinical_data_electrocardiogram)}, True')
clinical_data_electrocardiogram_top_30 =['nonsp_tw', 'lvh3_1', 'rbbb', 'st5_1_3', 'av1deg', 'infmi', 'lvh3_3', 'nonsp_st', 'ilbbb', 'st4_1_3', 'lvhst', 'irbbb', 'wpw', 'mob2', 'rvh', 'rtrial', 'mob1', 'part2deg', 'av3deg', 'iventblk', 'antlatmi', 'lbbb', 'antsepmi', 'afib', 'lah', 'paced']

# Clinical Data/Diagnostic Studies/Lung Function
clinical_data_lung_function = ['fev1', 'fvc']

# Clinical Data/Laboratory Tests
clinical_data_laboratory_tests = ['chol', 'hdl', 'trig']

# Clinical Data/Vital Signs/Seated Blood Pressure/SHHS1
clinical_data_seated_blood_pressure = ['aai']

# Demographics
demographics = ['gender', 'race', 'ethnicity']

# Demographics/SHHS1
demographics_shhs1 = ['mstat', 'age_s1']

# General Health/Medical Outcomes Study Short Form (SF-36)/SHHS1
general_health_medical_outcomes_study = ['cmp1yr25', 'vigact25', 'modact25', 'lift25', 'climbs25', 'climb125', 'bend25', 'wk1ml25', 'wksblk25', 'wk1blk25', 'bathe25', 'phctdn25', 'phacls25', 'limit25', 'exefrt25', 'emctdn25', 'emacls25', 'carful25', 'probsa25', 'bdpain25', 'painin25', 'pep25', 'nrvous25', 'down25', 'calm25', 'energ25', 'blue25', 'worn25', 'happy25', 'tired25', 'hlthlm25', 'sickez25', 'hlthy25', 'worse25', 'pf_s1', 'rp_s1', 'bp_s1', 'vt_s1', 'sf_s1', 're_s1', 'mh_s1', 'pcs_s1', 'mcs_s1']
general_health_medical_outcomes_study_top_30 = ['pcs_s1', 'mcs_s1', 'vt_s1', 'pf_s1', 'mh_s1', 'bp_s1', 'nrvous25', 'pep25', 'worse25', 'hlthy25', 'calm25', 'worn25', 'tired25', 'sf_s1', 'bend25', 'blue25', 'cmp1yr25', 'climbs25', 'hlthlm25', 'wk1ml25', 'energ25', 'happy25', 'sickez25', 'vigact25', 'bdpain25', 'painin25', 're_s1', 'rp_s1', 'down25', 'modact25']
print(f'general_health_medical_outcomes_study : {len(general_health_medical_outcomes_study)}, True')

# General Health/SHHS1
general_healt_shhs1 = ['stress15']

# Lifestyle and Behavioral Health/SHHS1
lifestyle = ['coffee15', 'tea15', 'soda15', 'evsmok15', 'smknow15', 'asa15', 'smokstat_s1']

# Medical History/Medications/SHHS1
medications = ['nitro15', 'estrgn1', 'progst1', 'anar1a1', 'lipid1', 'sympth1', 'tca1', 'asa1', 'nsaid1', 'benzod1', 'premar1', 'pdei1', 'ntca1', 'warf1', 'loop1', 'hctz1', 'hctzk1', 'ccbir1', 'ccbsr1', 'alpha1', 'alphad1', 'anar1b1', 'anar1c1', 'anar31', 'pvdl1', 'basq1', 'niac1', 'thry1', 'istrd1', 'ostrd1', 'beta1', 'betad1', 'ccb1', 'ace1', 'aced1', 'vaso1', 'vasod1', 'diuret1', 'dig1', 'ntg1']
medications_top30 = ['asa1', 'nsaid1', 'lipid1', 'alpha1', 'diuret1', 'nitro15', 'premar1', 'thry1', 'progst1', 'ntca1', 'estrgn1', 'benzod1', 'ace1', 'ccb1', 'beta1', 'hctz1', 'niac1', 'ostrd1', 'dig1', 'sympth1', 'hctzk1', 'vaso1', 'loop1', 'pdei1', 'ntg1', 'tca1', 'ccbsr1', 'warf1', 'aced1', 'istrd1']
print(f'medications : {len(medications)}, True')

# Medical History/SHHS1
medical_history_shhs1 = ['parrptdiab', 'cgpkyr', 'alcoh', 'angina15', 'mi15', 'stroke15', 'hf15', 'cabg15', 'ca15', 'othrcs15', 'pacem15', 'sa15', 'emphys15', 'crbron15', 'copd15', 'asthma15', 'asth1215', 'cough315', 'phlegm15', 'runny15', 'sinus15']
medical_history_shhs1_top_30 = ['cgpkyr', 'alcoh', 'sinus15', 'runny15', 'cough315', 'phlegm15', 'angina15', 'asth1215', 'mi15', 'crbron15', 'stroke15', 'asthma15', 'othrcs15', 'ca15', 'parrptdiab', 'cabg15', 'copd15', 'hf15', 'emphys15', 'sa15', 'pacem15']
print(f'medical_history_shhs1 : {len(medical_history_shhs1)}, True')

# Medical History/SHHS2
medical_history_shhs2 = ['htnderv_s1']

# Sleep Monitoring/Polysomnography/Administrative
sleep_administrative = ['rdisn', 'oximet51']

# Sleep Monitoring/Polysomnography/Apnea-Hypopnea Indices
sleep_apena_hypopnea_indices = ['rdi0p', 'rdi2p', 'rdi3p', 'rdi4p', 'rdi5p', 'rdi0pa', 'rdi2pa', 'rdi3pa', 'rdi4pa', 'rdi5pa', 'rdi0ps', 'rdi2ps', 'rdi3ps', 'rdi4ps', 'rdi5ps', 'rdi0pns', 'rdi2pns', 'rdi3pns', 'rdi4pns', 'rdi5pns', 'rdirem0p', 'rdirem2p', 'rdirem3p', 'rdirem4p', 'rdirem5p', 'rdinr0p', 'rdinr2p', 'rdinr3p', 'rdinr4p', 'rdinr5p', 'oai0p', 'oai4p', 'oai4pa', 'cai0p', 'cai4p', 'cai4pa', 'rdirbp', 'rdirop', 'rdinbp', 'rdinop', 'cardrbp', 'cardrop', 'cardnbp', 'cardnop', 'oardrbp', 'oardrop', 'oardnbp', 'oardnop', 'rdirba', 'rdiroa', 'rdinba', 'rdinoa', 'cardrba', 'cardroa', 'cardnba', 'cardnoa', 'oardrba', 'oardroa', 'oardnba', 'oardnoa', 'rdirbp2', 'rdirop2', 'rdinbp2', 'rdinop2', 'cardrbp2', 'cardrop2', 'cardnbp2', 'cardnop2', 'oardrbp2', 'oardrop2', 'oardnbp2', 'oardnop2', 'rdirba2', 'rdiroa2', 'rdinba2', 'rdinoa2', 'cardrba2', 'cardroa2', 'cardnba2', 'cardnoa2', 'oardrba2', 'oardroa2', 'oardnba2', 'oardnoa2', 'rdirbp3', 'rdirop3', 'rdinbp3', 'rdinop3', 'cardrbp3', 'cardrop3', 'cardnbp3', 'cardnop3', 'oardrbp3', 'oardrop3', 'oardnbp3', 'oardnop3', 'rdirba3', 'rdiroa3', 'rdinba3', 'rdinoa3', 'cardrba3', 'cardroa3', 'cardnba3', 'cardnoa3', 'oardrba3', 'oardroa3', 'oardnba3', 'oardnoa3', 'rdirbp4', 'rdirop4', 'rdinbp4', 'rdinop4', 'cardrbp4', 'cardrop4', 'cardnbp4', 'cardnop4', 'oardrbp4', 'oardrop4', 'oardnbp4', 'oardnop4', 'rdirba4', 'rdiroa4', 'rdinba4', 'rdinoa4', 'cardrba4', 'cardroa4', 'cardnba4', 'cardnoa4', 'oardrba4', 'oardroa4', 'oardnba4', 'oardnoa4', 'rdirbp5', 'rdirop5', 'rdinbp5', 'rdinop5', 'cardrbp5', 'cardrop5', 'cardnbp5', 'cardnop5', 'oardrbp5', 'oardrop5', 'oardnbp5', 'oardnop5', 'rdirba5', 'rdiroa5', 'rdinba5', 'rdinoa5', 'cardrba5', 'cardroa5', 'cardnba5', 'cardnoa5', 'oardrba5', 'oardroa5', 'oardnba5', 'oardnoa5', 'ahi_a0h3', 'ahi_a0h4', 'ahi_a0h3a', 'ahi_a0h4a', 'ahi_o0h3', 'ahi_o0h4', 'ahi_o0h3a', 'ahi_o0h4a', 'ahi_c0h3', 'ahi_c0h4', 'ahi_c0h3a', 'ahi_c0h4a', 'cent_obs_ratio', 'cent_obs_ratioa']
sleep_apena_hypopnea_indices_top_30 = ['pcs_s1', 'mcs_s1', 'vt_s1', 'pf_s1', 'mh_s1', 'bp_s1', 'nrvous25', 'pep25', 'worse25', 'hlthy25', 'calm25', 'worn25', 'tired25', 'sf_s1', 'bend25', 'blue25', 'cmp1yr25', 'climbs25', 'hlthlm25', 'wk1ml25', 'energ25', 'happy25', 'sickez25', 'vigact25', 'bdpain25', 'painin25', 'probsa25', 'rp_s1', 'down25', 'modact25']
print(f'sleep_apena_hypopnea_indices : {len(sleep_apena_hypopnea_indices)}, True')

# Sleep Monitoring/Polysomnography/Arousals
sleep_arousals = ['ai_all', 'ai_nrem', 'ai_rem', 'arrembp', 'arremop', 'arnrembp', 'arnremop', 'ahrembp', 'ahremop', 'ahnrembp', 'ahnremop']
print(f'sleep_arousals : {len(sleep_arousals)}')

# Sleep Monitoring/Polysomnography/Evening and Morning Survey/SHHS1
sleep_evening_and_morning_survey = ['hwlghr10', 'hwlgmn10', 'ltdp10', 'shlg10', 'rest10', 'hwwell10', 'diffa10', 'minfa10', 'wine10', 'shots10', 'beer10', 'coffee10', 'tea10', 'soda10', 'cgrtts10', 'pipe10', 'cigars10', 'wrhead10', 'wrface10', 'plstc10', 'vest10']
sleep_evening_and_morning_survey_top_30 = ['hstg342p', 'timest1p', 'pctsa95h', 'remlaiip', 'timest2', 'times34p', 'waso', 'timest34', 'pcstahar', 'timest1', 'timerem', 'nremepop', 'stonsetp', 'hslptawp', 'pslp_hp3', 'remlaip', 'slpprdp', 'hremt2p', 'timest2p', 'timebedp', 'slpeffp', 'remepop', 'stloutp', 'pslp_ap0hp3a', 'pcstah3d', 'timeremp', 'pcstahda', 'pctsa90h', 'slplatp', 'remepbp']
print(f'sleep_evening_and_morning_survey : {len(sleep_evening_and_morning_survey)}, True')

# Sleep Monitoring/Polysomnography/Heart Rate
sleep_heart_rate = ['savbrbh', 'smnbrbh', 'smxbrbh', 'savbroh', 'smnbroh', 'smxbroh', 'savbnbh', 'smnbnbh', 'smxbnbh', 'savbnoh', 'smnbnoh', 'smxbnoh', 'aavbrbh', 'amnbrbh', 'amxbrbh', 'aavbroh', 'amnbroh', 'amxbroh', 'aavbnbh', 'amnbnbh', 'amxbnbh', 'aavbnoh', 'amnbnoh', 'amxbnoh', 'havbroh', 'hmnbroh', 'hmxbroh', 'havbnbh', 'hmnbnbh', 'hmxbnbh', 'havbnoh', 'hmnbnoh', 'hmxbnoh', 'davbrbh', 'dmnbrbh', 'dmxbrbh', 'davbroh', 'dmnbroh', 'dmxbroh', 'davbnbh', 'dmnbnbh', 'dmxbnbh', 'davbnoh', 'dmnbnoh', 'dmxbnoh']
sleep_heart_rate_top_30 = ['davbroh', 'hmxbnoh', 'amnbnoh', 'havbnoh', 'davbnbh', 'aavbroh', 'havbnbh', 'amxbroh', 'dmxbroh', 'aavbnbh', 'savbnbh', 'davbnoh', 'amxbnoh', 'smxbnoh', 'smnbnoh', 'smnbroh', 'aavbnoh', 'havbroh', 'dmnbnoh', 'hmxbroh', 'hmnbroh', 'amnbroh', 'dmnbroh', 'smxbroh', 'savbnoh', 'smnbnbh', 'aavbrbh', 'hmnbnoh', 'dmxbnoh', 'smxbnbh']
print(f'sleep_heart_rate : {len(sleep_heart_rate)}, True')

# Sleep Monitoring/Polysomnography/Oxygen Saturation
sleep_oxygen_saturation = ['ndes2ph', 'ndes3ph', 'ndes4ph', 'ndes5ph', 'avsao2rh', 'avsao2nh', 'mnsao2rh', 'mnsao2nh', 'mxsao2rh', 'mxsao2nh', 'mxdrbp', 'mxdrop', 'mxdnbp', 'mxdnop', 'avdrbp', 'avdrop', 'avdnbp', 'avdnop', 'mndrbp', 'mndrop', 'mndnbp', 'mndnop', 'mxdroa', 'mxdnba', 'mxdnoa', 'avdroa', 'avdnba', 'avdnoa', 'mndroa', 'mndnba', 'mndnoa', 'mxdrbp2', 'mxdrop2', 'mxdnbp2', 'mxdnop2', 'avdrbp2', 'avdrop2', 'avdnbp2', 'avdnop2', 'mndrbp2', 'mndrop2', 'mndnbp2', 'mndnop2', 'mxdrba2', 'mxdroa2', 'mxdnba2', 'mxdnoa2', 'avdrba2', 'avdroa2', 'avdnba2', 'avdnoa2', 'mndrba2', 'mndroa2', 'mndnba2', 'mndnoa2', 'mxdrop3', 'mxdnbp3', 'mxdnop3', 'avdrop3', 'avdnbp3', 'avdnop3', 'mndrop3', 'mndnbp3', 'mndnop3', 'mxdrba3', 'mxdroa3', 'mxdnba3', 'mxdnoa3', 'avdrba3', 'avdroa3', 'avdnba3', 'avdnoa3', 'mndrba3', 'mndroa3', 'mndnba3', 'mndnoa3', 'mxdrop4', 'mxdnbp4', 'mxdnop4', 'avdrop4', 'avdnbp4', 'avdnop4', 'mndrop4', 'mndnbp4', 'mndnop4', 'mxdrba4', 'mxdroa4', 'mxdnba4', 'mxdnoa4', 'avdrba4', 'avdroa4', 'avdnba4', 'avdnoa4', 'mndrba4', 'mndroa4', 'mndnba4', 'mndnoa4', 'mxdrop5', 'mxdnop5', 'avdrop5', 'avdnop5', 'mndrop5', 'mndnop5', 'mxdroa5', 'mxdnba5', 'mxdnoa5', 'avdroa5', 'avdnba5', 'avdnoa5', 'mndroa5', 'mndnba5', 'mndnoa5', 'avgsat', 'minsat', 'mxdrbp3', 'avdrbp3', 'mndrbp3', 'mxdrba5', 'avdrba5', 'mndrba5']
sleep_oxygen_saturation_top_30 = ['avgsat', 'avdnoa3', 'ndes2ph', 'avsao2rh', 'avsao2nh', 'avdrop', 'avdnop', 'avdnoa5', 'ndes3ph', 'avdroa5', 'avdnoa', 'ndes4ph', 'avdroa', 'avdrop2', 'avdnop5', 'ndes5ph', 'avdnop2', 'avdnbp3', 'avdnoa2', 'avdroa3', 'avdrop3', 'mnsao2nh', 'avdnop3', 'avdnbp', 'avdrba5', 'avdnba5', 'avdrop4', 'avdnoa4', 'avdrbp', 'avdroa2']
print(f'sleep_oxygen_saturation : {len(sleep_oxygen_saturation)}, True')

# Sleep Monitoring/Polysomnography/Respiratory Event Counts
sleep_respiratory_event_counts = ['hrembp', 'hrop', 'hnrbp', 'hnrop', 'carbp', 'carop', 'canbp', 'canop', 'oarbp', 'oarop', 'oanbp', 'oanop', 'hremba', 'hroa', 'hnrba', 'hnroa', 'carba', 'caroa', 'canba', 'canoa', 'oarba', 'oaroa', 'oanba', 'oanoa', 'hrembp2', 'hrop2', 'hnrbp2', 'hnrop2', 'carbp2', 'carop2', 'canbp2', 'canop2', 'oarbp2', 'oarop2', 'oanbp2', 'oanop2', 'hremba2', 'hroa2', 'hnrba2', 'hnroa2', 'carba2', 'caroa2', 'canba2', 'canoa2', 'oarba2', 'oaroa2', 'oanba2', 'oanoa2', 'hrembp3', 'hrop3', 'hnrbp3', 'hnrop3', 'carbp3', 'carop3', 'canbp3', 'canop3', 'oarbp3', 'oarop3', 'oanbp3', 'oanop3', 'hremba3', 'hroa3', 'hnrba3', 'hnroa3', 'carba3', 'caroa3', 'canba3', 'canoa3', 'oarba3', 'oaroa3', 'oanba3', 'oanoa3', 'hrembp4', 'hrop4', 'hnrbp4', 'hnrop4', 'carbp4', 'carop4', 'canbp4', 'canop4', 'oarbp4', 'oarop4', 'oanbp4', 'oanop4', 'hremba4', 'hroa4', 'hnrba4', 'hnroa4', 'carba4', 'caroa4', 'canba4', 'canoa4', 'oarba4', 'oaroa4', 'oanba4', 'oanoa4', 'hrembp5', 'hrop5', 'hnrbp5', 'hnrop5', 'carbp5', 'carop5', 'canbp5', 'canop5', 'oarbp5', 'oarop5', 'oanbp5', 'oanop5', 'hremba5', 'hroa5', 'hnrba5', 'hnroa5', 'carba5', 'caroa5', 'canba5', 'canoa5', 'oarba5', 'oaroa5', 'oanba5', 'oanoa5']
sleep_respiratory_event_counts_top_30 = ['hnroa3', 'hnroa5', 'hnrbp4', 'hnrop', 'hrop', 'hnrop2', 'hnrop3', 'hrop2', 'hnroa2', 'hnrbp', 'hnrbp2', 'hroa3', 'hnroa', 'hroa2', 'hrembp', 'hnrba', 'hroa4', 'hroa', 'hrop3', 'hnroa4', 'hroa5', 'hnrba3', 'hrop4', 'hnrop5', 'hnrba2', 'hrop5', 'oanop5', 'hnrbp5', 'hrembp5', 'hnrop4']
print(f'sleep_respiratory_event_counts : {len(sleep_respiratory_event_counts)}, True')

# Sleep Monitoring/Polysomnography/Respiratory Event Lengths
sleep_respiratory_event_lengths = ['avhrbp', 'mnhrbp', 'mxhrbp', 'avhrop', 'mnhrop', 'mxhrop', 'avhnbp', 'mnhnbp', 'mxhnbp', 'avhnop', 'mnhnop', 'mxhnop', 'avhroa', 'mnhroa', 'mxhroa', 'avhnba', 'mnhnba', 'mxhnba', 'avhnoa', 'mnhnoa', 'mxhnoa', 'avhrbp2', 'mnhrbp2', 'mxhrbp2', 'avhrop2', 'mnhrop2', 'mxhrop2', 'avhnbp2', 'mnhnbp2', 'mxhnbp2', 'avhnop2', 'mnhnop2', 'mxhnop2', 'avhrba2', 'mnhrba2', 'mxhrba2', 'avhroa2', 'mnhroa2', 'mxhroa2', 'avhnba2', 'mnhnba2', 'mxhnba2', 'avhnoa2', 'mnhnoa2', 'mxhnoa2', 'avhrop3', 'mnhrop3', 'mxhrop3', 'avhnbp3', 'mnhnbp3', 'mxhnbp3', 'avhnop3', 'mnhnop3', 'mxhnop3', 'avhrba3', 'mnhrba3', 'mxhrba3', 'avhroa3', 'mnhroa3', 'mxhroa3', 'avhnba3', 'mnhnba3', 'mxhnba3', 'avhnoa3', 'mnhnoa3', 'mxhnoa3', 'avhrop4', 'mnhrop4', 'mxhrop4', 'avhnbp4', 'mnhnbp4', 'mxhnbp4', 'avhnop4', 'mnhnop4', 'mxhnop4', 'avhroa4', 'mnhroa4', 'mxhroa4', 'avhnba4', 'mnhnba4', 'mxhnba4', 'avhnoa4', 'mnhnoa4', 'mxhnoa4', 'avhrop5', 'mnhrop5', 'mxhrop5', 'avhnop5', 'mnhnop5', 'mxhnop5', 'avhroa5', 'mnhroa5', 'mxhroa5', 'avhnba5', 'mnhnba5', 'mxhnba5', 'avhnoa5', 'mnhnoa5', 'mxhnoa5', 'avhrbp3', 'mnhrbp3', 'mxhrbp3', 'avhrba4', 'mnhrba4', 'mxhrba4']
sleep_respiratory_event_lengths_top_30 = ['avhrop3', 'avhroa4', 'avhrop2', 'mxhnop3', 'avhnop', 'avhnop2', 'avhnop4', 'avhrop4', 'avhnoa', 'avhrop', 'avhnop3', 'avhnop5', 'mxhnop2', 'avhnoa2', 'mxhroa', 'mxhnop', 'avhnoa3', 'avhroa2', 'avhnoa4', 'avhnba5', 'avhroa', 'avhnbp3', 'avhroa5', 'mxhrop4', 'avhnoa5', 'mxhrop', 'mxhrop2', 'mxhnba', 'avhnba', 'avhrop5']
print(f'sleep_respiratory_event_lengths : {len(sleep_respiratory_event_lengths)}, True')

# Sleep Monitoring/Polysomnography/Signal Quality
sleep_signal_quality = ['abnoreeg', 'abnoreye', 'rcrdtime']

# Sleep Monitoring/Polysomnography/Signal Quality/SHHS1
sleep_signal_quality_shhs1 = ['lgbreath', 'respscch', 'sleep_latency', 'outliercheck1', 'outliercheck2']

# Sleep Monitoring/Polysomnography/Signal Quality/SHHS2
sleep_signal_quality_shhs2 = ['alpdel', 'period']

# Sleep Monitoring/Polysomnography/Sleep Architecture
sleep_architecture = ['waso', 'timest1p', 'timest2p', 'times34p', 'timeremp', 'timest1', 'timest2', 'timest34', 'timerem', 'supinep', 'nsupinep', 'pctstapn', 'pctsthyp', 'pcstahar', 'pcstah3d', 'pcstahda', 'pctsa95h', 'pctsa90h', 'pctsa85h', 'pctsa80h', 'pctsa75h', 'pctsa70h', 'remepbp', 'remepop', 'nremepbp', 'nremepop', 'stloutp', 'stonsetp', 'slplatp', 'remlaip', 'remlaiip', 'timebedp', 'slpprdp', 'slpeffp', 'stg2t1p', 'stg34t2p', 'remt1p', 'remt2p', 'remt34p', 'slptawp', 'hstg2t1p', 'hstg342p', 'hremt1p', 'hremt2p', 'hremt34p', 'hslptawp', 'pslp_ca0', 'pslp_oa0', 'pslp_ap0', 'pslp_ap3', 'pslp_hp0', 'pslp_hp3', 'pslp_hp3a', 'pslp_ap0hp3', 'pslp_ap0hp3a']
sleep_architecture_top_30 = ['hstg342p', 'timest1p', 'pctsa95h', 'remlaiip', 'timest2', 'times34p', 'waso', 'timest34', 'pcstahar', 'timest1', 'timerem', 'nremepop', 'stonsetp', 'hslptawp', 'pslp_hp3', 'remlaip', 'slpprdp', 'timest2p', 'timebedp', 'slpeffp', 'remepop', 'stloutp', 'pslp_ap0hp3a', 'pcstah3d', 'timeremp', 'pcstahda', 'hremt2p', 'pctsa90h', 'slplatp', 'remepbp']
print(f'sleep_architecture : {len(sleep_architecture)}, True')

# Sleep Questionnaires/Hypersomnia
sleep_question_hypersomnia = ['sleepy02']

# Sleep Questionnaires/Hypersomnia/Epworth Sleepiness Scale (ESS)/SHHS1
sleep_question_hypersomnia_epworth = ['sitrd02', 'watv02', 'sitpub02', 'pgrcar02', 'lydwn02', 'sittlk02', 'sitlch02', 'incar02', 'attabl02', 'drive02', 'ess_s1']
print(f'sleep_question_hypersomnia_epworth : {len(sleep_question_hypersomnia_epworth)}')

# Sleep Questionnaires/Sleep Disorder
sleep_question_sleep_disorder = ['mdsa02']

# Sleep Questionnaires/Sleep Disordered Breathing
sleep_question_sleep_disorder_breathing = ['hvsnrd02', 'hosnr02', 'loudsn02', 'issnor02', 'stpbrt02']

# Sleep Questionnaires/Sleep Disturbance
sleep_question_sleep_disturbance = ['slpill15', 'tfa02', 'wudnrs02', 'wu2em02', 'funres02', 'tkpill02', 'nges02', 'cough02', 'cp02', 'sob02', 'sweats02', 'noise02', 'painjt02', 'hb02', 'legcrp02', 'needbr02']
print(f'sleep_question_sleep_disturbance : {len(sleep_question_sleep_disturbance)}')

# Sleep Questionnaires/Sleep Habits
sleep_question_sleep_habits = ['napshr15', 'napsmn15', 'tfawdh02', 'tfawdm02', 'tfawda02', 'tfaweh02', 'tfawem02', 'tfawea02', 'mi2slp02', 'twuwdh02', 'twuwdm02', 'twuwda02', 'twuweh02', 'twuwem02', 'twuwea02', 'hrswd02', 'hrswe02', 'naps02', 'membhh02']
print(f'sleep_question_sleep_habits : {len(sleep_question_sleep_habits)}')

# Sleep Treatment
sleep_treatment = ['surgtr02', 'o2thpy02']