import pathlib
import glob

from pkg_utils import load_pkg_data, average_pkg_data
from night_form_utils import load_night_forms, interp_night_forms
from plotting import plot_PKG_NF_time_series, plot_bar_comp, plot_bar_adaptive_comp


# hard-coded path to the data file
p_night_forms = pathlib.Path('/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/Data/RCS02/Step6_at_home aDBS_short/')
vec_night_forms = glob.glob(str(p_night_forms / 'RCS02_*analysis.xlsx'))
assert len(vec_night_forms) == 1

# read the data file
df_night_forms = load_night_forms(vec_night_forms[0], '2023/02/20', '2023/02/26')

# get the path to the PKG data
p_pkg = pathlib.Path('/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/Data/RCS02/Step6_at_home aDBS_short/sixth round self_switch/pkg/')
vec_pkg_L = glob.glob(str(p_pkg / 'L' / 'RCS02L*.csv'))
vec_pkg_R = glob.glob(str(p_pkg / 'R' / 'RCS02R*.csv'))

# read the PKG data
df_pkg_L, df_pkg_valid_L, bool_valid_L = load_pkg_data(vec_pkg_L[0])
df_pkg_R, df_pkg_valid_R, bool_valid_R = load_pkg_data(vec_pkg_R[0])

# perform averaging in the PKG table also
df_pkg_avg_L, df_pkg_avg_valid_L, bool_avg_valid_L = average_pkg_data(df_pkg_L, avg_min=20)
df_pkg_avg_R, df_pkg_avg_valid_R, bool_avg_valid_R = average_pkg_data(df_pkg_R, avg_min=20)

# interpolate the night forms to have same time stamps as the PKG data
df_night_forms_interp_valid = interp_night_forms(df_night_forms, df_pkg_valid_L['Date_Time'], bool_valid_L)
df_night_forms_interp_avg_valid = interp_night_forms(df_night_forms, df_pkg_avg_valid_L['Date_Time'], bool_avg_valid_L)

# make some quick plots
# plot_PKG_NF_time_series(df_pkg_valid_L, df_pkg_avg_valid_L, df_night_forms_interp_avg_valid,
#                         str_side='L', f_output='PKG_NF_time_series_L.png')
# plot_PKG_NF_time_series(df_pkg_valid_R, df_pkg_avg_valid_R, df_night_forms_interp_avg_valid,
#                         str_side='R', f_output='PKG_NF_time_series_R.png')

# # then plot bar plot for comparison
# plot_bar_comp(df_pkg_valid_L, bool_valid_L, df_night_forms, str_side='L', str_mode='brady',
#               bool_smooth=False, f_output='PKG_NF_brady_comp_L.png')
# plot_bar_comp(df_pkg_valid_R, bool_valid_R, df_night_forms, str_side='R', str_mode='brady',
#               bool_smooth=False, f_output='PKG_NF_brady_comp_R.png')
#
# # next plot bar plot for smoothed data for comparison
# plot_bar_comp(df_pkg_avg_valid_L, bool_avg_valid_L, df_night_forms, str_side='L', str_mode='brady',
#               bool_smooth=True, f_output='PKG_NF_brady_comp_smooth_L.png')
# plot_bar_comp(df_pkg_avg_valid_R, bool_avg_valid_R, df_night_forms, str_side='R', str_mode='brady',
#               bool_smooth=True, f_output='PKG_NF_brady_comp_smooth_R.png')
#
# # also plot the ones with thresholding
# plot_bar_comp(df_pkg_avg_valid_L, bool_avg_valid_L, df_night_forms, str_side='L', str_mode='brady',
#               bool_smooth=True, bool_thresh=True, thresh=45, f_output='PKG_NF_brady_comp_smooth_thresh45_L.png')
# plot_bar_comp(df_pkg_avg_valid_R, bool_avg_valid_R, df_night_forms, str_side='R', str_mode='brady',
#               bool_smooth=True, bool_thresh=True, thresh=45, f_output='PKG_NF_brady_comp_smooth_thresh45_R.png')


############################################################################################################
# # plot dyskinesia
#
# plot_bar_comp(df_pkg_valid_L, bool_valid_L, df_night_forms, str_side='L', str_mode='brady',
#               bool_smooth=False, f_output='PKG_NF_brady_comp_L.png')
# plot_bar_comp(df_pkg_valid_R, bool_valid_R, df_night_forms, str_side='R', str_mode='brady',
#               bool_smooth=False, f_output='PKG_NF_brady_comp_R.png')
#
# # next plot bar plot for smoothed data for comparison
# plot_bar_comp(df_pkg_avg_valid_L, bool_avg_valid_L, df_night_forms, str_side='L', str_mode='dysk',
#               bool_smooth=True, f_output='PKG_NF_dysk_comp_smooth_L.png')
# plot_bar_comp(df_pkg_avg_valid_R, bool_avg_valid_R, df_night_forms, str_side='R', str_mode='dysk',
#               bool_smooth=True, f_output='PKG_NF_dysk_comp_smooth_R.png')
#
# also plot the ones with thresholding
# plot_bar_comp(df_pkg_avg_valid_L, bool_avg_valid_L, df_night_forms, str_side='L', str_mode='dysk',
#               bool_smooth=True, bool_thresh=True, thresh=7, f_output='PKG_NF_dysk_comp_smooth_thresh7_L.png')
# plot_bar_comp(df_pkg_avg_valid_R, bool_avg_valid_R, df_night_forms, str_side='R', str_mode='dysk',
#               bool_smooth=True, bool_thresh=True, thresh=7, f_output='PKG_NF_dysk_comp_smooth_thresh7_R.png')

############################################################################################################
# plot cDBS vs aDBS

# compare bradykinesia
plot_bar_adaptive_comp(df_pkg_avg_valid_L, bool_avg_valid_L, df_night_forms, str_side='L', str_mode='brady',
                       f_output='PKG_NF_brady_adaptive_comp_L.png')
plot_bar_adaptive_comp(df_pkg_avg_valid_R, bool_avg_valid_R, df_night_forms, str_side='R', str_mode='brady',
                       f_output='PKG_NF_brady_adaptive_comp_R.png')

# compare dyskinesia
plot_bar_adaptive_comp(df_pkg_avg_valid_L, bool_avg_valid_L, df_night_forms, str_side='L', str_mode='dysk',
                       f_output='PKG_NF_dysk_adaptive_comp_L.png')
plot_bar_adaptive_comp(df_pkg_avg_valid_R, bool_avg_valid_R, df_night_forms, str_side='R', str_mode='dysk',
                       f_output='PKG_NF_dysk_adaptive_comp_R.png')

t1 = 1