import pickle
import pathlib

import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.stats as stats


def load_model_id_results(output):
    acc_python_full = np.stack([output['sinPB'][i]['vec_acc'][:5] for i in range(len(output['sinPB']))], axis=1).T
    auc_python_full = np.stack([output['sinPB'][i]['vec_auc'][:5] for i in range(len(output['sinPB']))], axis=1).T
    acc_python_top = [output['sinPB'][i]['vec_acc'][0] for i in range(len(output['sinPB']))]
    auc_python_top = [output['sinPB'][i]['vec_auc'][0] for i in range(len(output['sinPB']))]

    acc_python_full_sfs = np.stack([output['sfsPB'][i]['vec_acc'][:5] for i in range(len(output['sfsPB']))], axis=1).T
    auc_python_full_sfs = np.stack([output['sfsPB'][i]['vec_auc'][:5] for i in range(len(output['sfsPB']))], axis=1).T
    acc_python_top_sfs = [output['sfsPB'][i]['vec_acc'][-1] for i in range(len(output['sfsPB']))]
    auc_python_top_sfs = [output['sfsPB'][i]['vec_auc'][-1] for i in range(len(output['sfsPB']))]

    return acc_python_full, auc_python_full, acc_python_top, auc_python_top, acc_python_full_sfs, auc_python_full_sfs, acc_python_top_sfs, auc_python_top_sfs


def plot_lda_results():
    # hard code all paths
    p_output = pathlib.Path('/home/jyao/Downloads/biomarker_id/model_id/RCS02')
    f_output_python = 'RCS02_R_med_avg_auc_LDA.pkl'
    f_output_matlab = 'RCS02_R_med_level_stats.mat'

    p_figure_output = pathlib.Path('/home/jyao/Downloads/biomarker_id/figures/model_id/RCS02')

    # load in the python values
    output = pickle.load(open(str(p_output / f_output_python), 'rb'))
    acc_python_full, auc_python_full, acc_python_top, auc_python_top, \
        acc_python_full_sfs, auc_python_full_sfs, acc_python_top_sfs, auc_python_top_sfs = \
            load_model_id_results(output)



    # # plot the figure for auc
    # fig = plt.figure()
    # sns.boxplot(data=results_auc, x='Program', y='AUC', orient='v')
    # ax = plt.gca()

    # # set the labels
    # ax.set(xlabel=None)
    # ax.set(ylabel='AUC with LDA')
    # ax.set(title='AUC with LDA for Top 1 Power Band')

    # # set the ranges
    # ax.set_ylim([0.62, 0.80])

    # # remove the top and right spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # fig.savefig(str(p_figure_output / 'RCS02_R_med_level_auc_LDA.png'), dpi=300)
    # plt.close(fig)

    # # plot the figure for accuracy
    # fig = plt.figure()
    # sns.boxplot(data=results_auc, x='Program', y='ACC', orient='v')
    # ax = plt.gca()

    # # set the labels
    # ax.set(xlabel=None)
    # ax.set(ylabel='ACC with LDA')
    # ax.set(title='ACC with LDA for Top 1 Power Band')

    # # set the ranges
    # ax.set_ylim([0.60, 0.75])

    # # remove the top and right spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # fig.savefig(str(p_figure_output / 'RCS02_R_med_level_acc_LDA.png'), dpi=300)
    # plt.close(fig)

    # # now plot the line plots for the top5 auc
    # fig = plt.figure()
    # idx = np.arange(1, 6, 1)
    # x_ticks = [str(x) for x in idx]
    # plt.plot(idx, np.mean(auc_python_full, axis=0), 'r', label='Python')
    # plt.fill_between(idx, np.mean(auc_python_full, axis=0) - np.std(auc_python_full, axis=0),
    #                  np.mean(auc_python_full, axis=0) + np.std(auc_python_full, axis=0), color='r', alpha=0.2)


    # plt.plot(idx, np.mean(auc_matlab_full, axis=0), 'b', label='MATLAB')
    # plt.fill_between(idx, np.mean(auc_matlab_full, axis=0) - np.std(auc_matlab_full, axis=0),
    #                     np.mean(auc_matlab_full, axis=0) + np.std(auc_matlab_full, axis=0), color='b', alpha=0.2)
    # plt.xlabel('Index of Power Bands')
    # plt.ylabel('AUC with LDA')
    # plt.title('Comparison of AUC with LDA for Top 5 Power Bands')
    # plt.legend(frameon=False)

    # ax = plt.gca()
    # ax.set_xticks(idx)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # fig.savefig(str(p_figure_output / 'RCS02_R_med_level_auc_LDA_top5.png'), dpi=300)
    # plt.close(fig)

    # '''
    # SFS plots
    # '''
    # # also plot the SFS combined AUC
    # fig = plt.figure()
    # idx = np.arange(1, 6, 1)
    # x_ticks = [str(x) for x in idx]
    # plt.plot(idx, np.mean(auc_combined_python, axis=0), 'r', label='Python')
    # plt.fill_between(idx, np.mean(auc_combined_python, axis=0) - np.std(auc_combined_python, axis=0),
    #                  np.mean(auc_combined_python, axis=0) + np.std(auc_combined_python, axis=0), color='r', alpha=0.2)


    # plt.plot(idx, np.mean(auc_combined_matlab, axis=0), 'b', label='MATLAB')
    # plt.fill_between(idx, np.mean(auc_combined_matlab, axis=0) - np.std(auc_combined_matlab, axis=0),
    #                     np.mean(auc_combined_matlab, axis=0) + np.std(auc_combined_matlab, axis=0), color='b', alpha=0.2)
    # plt.xlabel('Index of Power Bands')
    # plt.ylabel('AUC with LDA')
    # plt.title('Comparison of AUC with LDA for Top 5 Power Bands')
    # plt.legend(frameon=False)

    # ax = plt.gca()
    # ax.set_xticks(idx)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # fig.savefig(str(p_figure_output / 'RCS02_R_med_level_auc_LDA_top5_combo.png'), dpi=300)
    # plt.close(fig)


    # also load the SVM and QDA results
    f_output_python_QDA = 'RCS02_R_med_level_auc_QDA.pkl'
    f_output_python_SVM = 'RCS02_R_med_level_auc_SVM.pkl'
    f_output_python_RF = 'RCS02_R_med_level_acc_RF.pkl'
    f_output_python_MLP = 'RCS02_R_med_avg_auc_MLP.pkl'
    f_output_python_RNN = 'RCS02_R_med_avg_auc_RNN.pkl'
    
    output_QDA = pickle.load(open(str(p_output / f_output_python_QDA), 'rb'))
    _, auc_python_full_QDA, _, _, _, auc_python_full_sfs_QDA, _, _ = \
            load_model_id_results(output_QDA)
            
    output_SVM = pickle.load(open(str(p_output / f_output_python_SVM), 'rb'))
    _, auc_python_full_SVM, _, _, _, auc_python_full_sfs_SVM, _, _ = \
            load_model_id_results(output_SVM)

    output_RF = pickle.load(open(str(p_output / f_output_python_RF), 'rb'))
    _, auc_python_full_RF, _, _, _, auc_python_full_sfs_RF, _, _ = \
            load_model_id_results(output_RF)
    
    output_MLP = pickle.load(open(str(p_output / f_output_python_MLP), 'rb'))
    _, auc_python_full_MLP, _, _, _, auc_python_full_sfs_MLP, _, _ = \
            load_model_id_results(output_MLP)

    output_RNN = pickle.load(open(str(p_output / f_output_python_RNN), 'rb'))
    _, auc_python_full_RNN, _, _, _, auc_python_full_sfs_RNN, _, _ = \
            load_model_id_results(output_RNN)

    fig = plt.figure()
    idx = np.arange(1, 6, 1)
    x_ticks = [str(x) for x in idx]
    plt.plot(idx, np.mean(auc_python_full_sfs, axis=0), 'r', label='LDA')
    plt.fill_between(idx, np.mean(auc_python_full_sfs, axis=0) - np.std(auc_python_full_sfs, axis=0),
                     np.mean(auc_python_full_sfs, axis=0) + np.std(auc_python_full_sfs, axis=0), color='r', alpha=0.2)

    plt.plot(idx, np.mean(auc_python_full_sfs_QDA, axis=0), 'k', label='QDA')
    plt.fill_between(idx, np.mean(auc_python_full_sfs_QDA, axis=0) - np.std(auc_python_full_sfs_QDA, axis=0),
                     np.mean(auc_python_full_sfs_QDA, axis=0) + np.std(auc_python_full_sfs_QDA, axis=0),
                     color='k', alpha=0.2)

    plt.plot(idx, np.mean(auc_python_full_sfs_SVM, axis=0), 'g', label='SVM')
    plt.fill_between(idx, np.mean(auc_python_full_sfs_SVM, axis=0) - np.std(auc_python_full_sfs_SVM, axis=0),
                     np.mean(auc_python_full_sfs_SVM, axis=0) + np.std(auc_python_full_sfs_SVM, axis=0),
                     color='g', alpha=0.2)

    plt.plot(idx, np.mean(auc_python_full_sfs_RF, axis=0), 'm', label='RF')
    plt.fill_between(idx, np.mean(auc_python_full_sfs_RF, axis=0) - np.std(auc_python_full_sfs_RF, axis=0),
                        np.mean(auc_python_full_sfs_RF, axis=0) + np.std(auc_python_full_sfs_RF, axis=0),
                     color='m', alpha=0.2)

    plt.plot(idx, np.mean(auc_python_full_sfs_MLP, axis=0), 'b', label='MLP')
    plt.fill_between(idx, np.mean(auc_python_full_sfs_MLP, axis=0) - np.std(auc_python_full_sfs_MLP, axis=0),
                     np.mean(auc_python_full_sfs_MLP, axis=0) + np.std(auc_python_full_sfs_MLP, axis=0),
                     color='b', alpha=0.2)

    plt.plot(idx, np.mean(auc_python_full_sfs_RNN, axis=0), 'c', label='RNN')
    plt.fill_between(idx, np.mean(auc_python_full_sfs_RNN, axis=0) - np.std(auc_python_full_sfs_RNN, axis=0),
                        np.mean(auc_python_full_sfs_RNN, axis=0) + np.std(auc_python_full_sfs_RNN, axis=0),
                        color='c', alpha=0.2)

    plt.xlabel('Number of Power Bands Included')
    plt.ylabel('AUC with Algorithms')
    plt.title('Comparison of AUC for Top 5 Power Bands')
    plt.legend(frameon=False)
    plt.xticks(idx, x_ticks)

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(str(p_figure_output / 'RCS02_R_med_level_auc_comp_top5_combo.png'), dpi=300)
    plt.close(fig)

    t1 = 1


if __name__ == '__main__':
    plot_lda_results()
