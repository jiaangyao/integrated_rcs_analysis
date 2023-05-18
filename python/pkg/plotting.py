import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# from statannot import add_stat_annotation
from scipy import stats

def plot_PKG_NF_time_series(df_pkg_valid, df_pkg_avg_valid, df_night_forms_interp_avg_valid, str_side,
                            p_output='/home/jyao/Downloads/PKG_NF/PKG_NF_time_series/',
                            f_output='PKG_NF_time_series_L.png'):
    # make some quick plots
    fig = plt.figure(figsize=(2560 / 300, 1440 / 300), dpi=300)
    plt.subplot(211)
    ln1 = plt.plot(df_pkg_valid['Date_Time'], df_pkg_valid['BKS'], color='#1f77b4', alpha=0.2,
                   linewidth=0.8, label='_nolegend_')
    ln2 = plt.plot(df_pkg_avg_valid['Date_Time'], df_pkg_avg_valid['BKS'], color='#1f77b4', alpha=1,
                   linewidth=1.5, label='PKG')
    plt.ylabel('PKG BKS (A.U.)')

    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', colors='#1f77b4')
    ax.yaxis.label.set_color('#1f77b4')

    plt.twinx()
    ln3 = plt.plot(df_night_forms_interp_avg_valid.loc[:, ('general patient info', 'day', 'day')],
                   df_night_forms_interp_avg_valid.loc[:, ('My symptoms', 'Bradykinesia (0-24h)', 'total')],
                   color='#ff7f0e', alpha=1, linewidth=1.5, label='Night Form')
    plt.ylabel('Night Form Brady Duration (Hr)', fontsize=8)
    plt.title('PKG Brady Score vs Night Form Brady Duration: RCS02 Body Side {}'.format(str_side))

    ax = fig.gca()
    ax.spines['left'].set_edgecolor('#1f77b4')
    ax.spines['right'].set_edgecolor('#ff7f0e')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', colors='#ff7f0e', labelsize=6)
    ax.yaxis.label.set_color('#ff7f0e')

    # added these three lines
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, frameon=False, loc='upper right', fontsize=8)

    ########################################################
    # now plot the dyskinesia
    plt.subplot(212)
    ln1 = plt.plot(df_pkg_valid['Date_Time'], df_pkg_valid['DKS'], color='#1f77b4', alpha=0.2,
                   linewidth=0.8, label='_nolegend_')
    ln2 = plt.plot(df_pkg_avg_valid['Date_Time'], df_pkg_avg_valid['DKS'], color='#1f77b4', alpha=1,
                   linewidth=1.5, label='PKG')
    plt.ylabel('PKG DKS (A.U.)')

    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', colors='#1f77b4')
    ax.yaxis.label.set_color('#1f77b4')

    plt.twinx()
    ln3 = plt.plot(df_night_forms_interp_avg_valid.loc[:, ('general patient info', 'day', 'day')],
                   df_night_forms_interp_avg_valid.loc[:, ('My symptoms', 'Dyskinesia (0-24h)', 'total')],
                   color='#ff7f0e', alpha=1, linewidth=1.5, label='Night Form')
    plt.ylabel('Night Form Dysk Duration (Hr)', fontsize=8)
    plt.title('PKG Dysk Score vs Night Form Dysk Duration: RCS02 Body Side {}'.format(str_side))

    ax = fig.gca()
    ax.spines['left'].set_edgecolor('#1f77b4')
    ax.spines['right'].set_edgecolor('#ff7f0e')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', colors='#ff7f0e', labelsize=6)
    ax.yaxis.label.set_color('#ff7f0e')

    # added these three lines
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, frameon=False, loc='upper right', fontsize=8)

    plt.tight_layout()

    # save output figure
    fig.savefig(str(pathlib.Path(p_output) / f_output), dpi=300)
    plt.close(fig)


def plot_bar_comp(df_pkg_valid, bool_valid, df_night_form, str_side, str_mode='brady', bool_smooth=False,
                  bool_thresh=False, thresh=26, p_output='/home/jyao/Downloads/PKG_NF/PKG_NF_comp_bar/',
                  f_output='PKG_NF_bar_comp_L.png'):

    if str_side == 'L':
        str_side_cortex = 'RIGHT BRAIN'
    else:
        str_side_cortex = 'LEFT BRAIN'

    if str_mode == 'brady':
        str_pkg_query = 'BKS'
        str_nf_query = ('My symptoms', 'Bradykinesia (0-24h)', 'total')
    else:
        str_pkg_query = 'DKS'
        str_nf_query = ('My symptoms', 'Dyskinesia (0-24h)', 'total')

    # make local copy and change time
    df_pkg_valid_curr = df_pkg_valid.copy()
    df_pkg_valid_curr['Date_Time'] = df_pkg_valid_curr['Date_Time'].dt.date
    df_pkg_valid_curr = df_pkg_valid_curr.loc[bool_valid, :]

    # also optionally threshold data
    if bool_thresh:
        idx_valid_thresh = np.logical_and((df_pkg_valid_curr.loc[:, str_pkg_query] > thresh).array,
                                          (df_pkg_valid_curr.loc[:, str_pkg_query] < 80).array)
        # idx_valid_thresh = df_pkg_valid_curr.loc[:, str_pkg_query] > thresh
        df_pkg_valid_curr = df_pkg_valid_curr.loc[idx_valid_thresh, :]

    df_night_form_curr = df_night_form.copy()
    df_night_form_curr.loc[:, ('general patient info', 'day', 'day')] = df_night_form_curr.loc[:, ('general patient info', 'day', 'day')].dt.date

    # next also create entries for cDBS and aDBS
    vec_str_adaptive = []
    vec_total_dur = []
    for i in range(df_night_form_curr.shape[0]):
        curr_date = df_night_form_curr.loc[i, ('general patient info', 'day', 'day')]
        curr_str_adaptive = df_night_form_curr.loc[i, ('aDBS algorithm - {}'.format(str_side_cortex), 'title', 'title')]

        n_curr_date = np.sum(df_pkg_valid_curr['Date_Time'] == curr_date)
        vec_str_adaptive_curr = [curr_str_adaptive] * n_curr_date
        vec_str_adaptive = vec_str_adaptive + vec_str_adaptive_curr

        if bool_thresh:
            if bool_smooth:
                normal_factor = 20
            else:
                normal_factor = 2
            vec_total_dur_curr = n_curr_date * normal_factor/60
            vec_total_dur.append(vec_total_dur_curr)

    # sanity check and add back to original dataframe
    assert len(vec_str_adaptive) == df_pkg_valid_curr.shape[0]
    df_pkg_valid_curr = df_pkg_valid_curr.assign(adaptive=vec_str_adaptive)
    if bool_thresh:
        df_night_form_curr = df_night_form_curr.assign(total_dur=vec_total_dur)

    # now generate the plot
    fig = plt.figure(figsize=(2560/300, 1440/300), dpi=300)
    plt.subplot(211)
    if not bool_thresh:
        # sns.violinplot(data=df_pkg_valid_curr, x='Date_Time', y='BKS', hue='adaptive', split=False, inner='quartile', scale='count')
        sns.boxplot(data=df_pkg_valid_curr, x='Date_Time', y=str_pkg_query, hue='adaptive')
        plt.legend(loc='upper right', frameon=False, fontsize=8)
        if not bool_smooth:
            plt.title('PKG {} Score: RCS02 Body Side {}'.format(str_mode.capitalize(), str_side), fontsize=8)
        else:
            plt.title('Smoothed PKG {} Score: RCS02 Body Side {}'.format(str_mode.capitalize(), str_side), fontsize=8)
    else:
        sns.barplot(data=df_night_form_curr, x=('general patient info', 'day', 'day'),
                    y='total_dur', hue=('aDBS algorithm - {}'.format(str_side_cortex), 'title', 'title'))
        plt.legend(loc='upper right', frameon=False, fontsize=8)
        plt.title('PKG {} Duration: RCS02 Body Side {} - Threshold {}'.format(str_mode.capitalize(), str_side, thresh),
                  fontsize=8)
        plt.ylabel('Total {} Duration (Hr)'.format(str_mode.capitalize()), fontsize=8)

    # change tick parameter sizes
    ax = fig.gca()
    ax.set(xlabel=None)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)

    # also plot the night form
    plt.subplot(212)
    sns.barplot(data=df_night_form_curr, x=('general patient info', 'day', 'day'), y=str_nf_query,
                hue=('aDBS algorithm - {}'.format(str_side_cortex), 'title', 'title'))
    plt.title('Night Form {} Duration'.format(str_mode.capitalize()), fontsize=8)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Night Form {} Duration (Hr)'.format(str_mode.capitalize()), fontsize=6)

    ax = fig.gca()
    ax.get_legend().remove()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=6)

    plt.tight_layout()

    # save the output figure
    fig.savefig(str(pathlib.Path(p_output) / f_output), dpi=300)
    plt.close(fig)


def plot_bar_adaptive_comp(df_pkg_valid, bool_valid, df_night_form, str_side, str_mode='brady', p_output='/home/jyao/Downloads/PKG_NF/PKG_NF_comp_adaptive_bar/',
                  f_output='PKG_NF_brady_adaptive_comp_L.png'):


    if str_side == 'L':
        str_side_cortex = 'RIGHT BRAIN'
    else:
        str_side_cortex = 'LEFT BRAIN'

    if str_mode == 'brady':
        str_pkg_query = 'BKS'
    else:
        str_pkg_query = 'DKS'

    # make local copy and change time
    df_pkg_valid_curr = df_pkg_valid.copy()
    df_pkg_valid_curr['Date_Time'] = df_pkg_valid_curr['Date_Time'].dt.date
    df_pkg_valid_curr = df_pkg_valid_curr.loc[bool_valid, :]

    df_night_form_curr = df_night_form.copy()
    df_night_form_curr.loc[:, ('general patient info', 'day', 'day')] = df_night_form_curr.loc[:, ('general patient info', 'day', 'day')].dt.date

    # next also create entries for cDBS and aDBS
    vec_str_adaptive = []
    vec_idx_adaptive = []
    for i in range(df_night_form_curr.shape[0]):
        curr_date = df_night_form_curr.loc[i, ('general patient info', 'day', 'day')]
        curr_str_adaptive = df_night_form_curr.loc[i, ('aDBS algorithm - {}'.format(str_side_cortex), 'title', 'title')]

        n_curr_date = np.sum(df_pkg_valid_curr['Date_Time'] == curr_date)
        vec_str_adaptive_curr = [curr_str_adaptive] * n_curr_date
        vec_str_adaptive = vec_str_adaptive + vec_str_adaptive_curr

        if curr_str_adaptive == 'aDBS':
            vec_idx_adaptive_curr = [True] * n_curr_date
        else:
            vec_idx_adaptive_curr = [False] * n_curr_date
        vec_idx_adaptive = vec_idx_adaptive + vec_idx_adaptive_curr

    # sanity check and add back to original dataframe
    assert len(vec_str_adaptive) == df_pkg_valid_curr.shape[0]
    df_pkg_valid_curr = df_pkg_valid_curr.assign(adaptive=vec_str_adaptive)
    df_pkg_valid_curr[str_pkg_query] = df_pkg_valid_curr[str_pkg_query].astype(int)

    # now generate the plot
    box_pairs = [("aDBS", "cDBS")]
    pvalues = [stats.mannwhitneyu(df_pkg_valid_curr.loc[vec_idx_adaptive, str_pkg_query],
                                  df_pkg_valid_curr.loc[np.logical_not(vec_idx_adaptive), str_pkg_query])[1]]

    fig = plt.figure(figsize=(800/300, 1080/300), dpi=300)
    ax = sns.violinplot(data=df_pkg_valid_curr, x='adaptive', y=str_pkg_query, inner='quartile')
    # sns.boxplot(data=df_pkg_valid_curr, x='adaptive', y=str_pkg_query)
    plt.legend(loc='upper right', frameon=False, fontsize=8)
    transform = plt.gca().transAxes
    plt.text(0.03, 0.99, 'Mann-Whitney U\np={:.3f}'.format(pvalues[0]), fontsize=4,
             ha='left', va='top', transform=ax.transAxes)
    plt.ylabel('PKG {} Score (A.U.)'.format(str_mode.capitalize()), fontsize=8)

    ax.set(xlabel=None)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    if str_mode == 'dysk':
        plt.ylim([0, 40])

    test_short_name = "MannWhitneyU"
    test_results = add_stat_annotation(ax, data=df_pkg_valid_curr, x='adaptive', y=str_pkg_query,
                                       box_pairs=box_pairs, loc='outside', perform_stat_test=False,
                                       pvalues=pvalues, test_short_name=test_short_name,
                                       text_format='star', verbose=2)

    plt.tight_layout()

    fig.savefig(str(pathlib.Path(p_output) / f_output), dpi=300)
    plt.close(fig)
