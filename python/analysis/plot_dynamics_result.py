import pickle
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.stats as stats


def plot_dynamics_results():
    # hard code all paths
    p_output = pathlib.Path('/home/jyao/Downloads/biomarker_id/dynamics/')
    f_output_python = 'RCS02_R_med_level_auc_LDA_dynamics.pkl'

    # load in the python values
    output = pickle.load(open(str(p_output / f_output_python), 'rb'))

    # obtain the full acc
    acc_output_full = []
    for i in range(1, len(output['sinPB']) + 1):
        str_n_dyna = 'n_dynmamics_{}'.format(i)
        acc_output_n_dyna_curr = np.stack([output['sinPB'][str_n_dyna][0][j]['vec_acc'][:5] for j in range(len(output['sinPB'][str_n_dyna][0]))])
        acc_output_full.append(acc_output_n_dyna_curr)
    acc_output_full = np.stack(acc_output_full, axis=0)

    # obtain the full auc
    auc_output_full = []
    for i in range(1, len(output['sinPB']) + 1):
        str_n_dyna = 'n_dynmamics_{}'.format(i)
        auc_output_n_dyna_curr = np.stack([output['sinPB'][str_n_dyna][0][j]['vec_auc'][:5] for j in range(len(output['sinPB'][str_n_dyna][0]))])
        auc_output_full.append(auc_output_n_dyna_curr)
    auc_output_full = np.stack(auc_output_full, axis=0)

    # obtain the full acc from SFS
    acc_output_full_sfs = []
    for i in range(1, len(output['sfsPB']) + 1):
        str_n_dyna = 'n_dynmamics_{}'.format(i)
        auc_output_n_dyna_curr = np.stack([output['sfsPB'][str_n_dyna][0][j]['vec_acc'][:5] for j in range(len(output['sfsPB'][str_n_dyna][0]))])
        acc_output_full_sfs.append(auc_output_n_dyna_curr)
    acc_output_full_sfs = np.stack(acc_output_full_sfs, axis=0)

    # obtain the full auc from SFS
    auc_output_full_sfs = []
    for i in range(1, len(output['sfsPB']) + 1):
        str_n_dyna = 'n_dynmamics_{}'.format(i)
        auc_output_n_dyna_curr = np.stack([output['sfsPB'][str_n_dyna][0][j]['vec_auc'][:5] for j in range(len(output['sfsPB'][str_n_dyna][0]))])
        auc_output_full_sfs.append(auc_output_n_dyna_curr)
    auc_output_full_sfs = np.stack(auc_output_full_sfs, axis=0)

    # now plot the line plots for the top5 auc
    fig = plt.figure(figsize=(12, 6))
    idx = np.arange(1, 7, 1)
    x_ticks = [str(x) for x in idx]
    plt.subplot(1, 2, 1)
    plt.plot(idx, np.mean(acc_output_full[..., 0], axis=-1), 'r')
    plt.fill_between(idx, np.mean(acc_output_full[..., 0], axis=-1) - np.std(acc_output_full[..., 0], axis=-1),
                     np.mean(acc_output_full[..., 0], axis=-1) + np.std(acc_output_full[..., 0], axis=-1), color='r', alpha=0.2)
    plt.xlabel('# of windows provided to the model')
    plt.ylabel('Accuracy')

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    plt.subplot(1, 2, 2)
    plt.plot(idx, np.mean(auc_output_full[..., 0], axis=-1), 'r')
    plt.fill_between(idx, np.mean(auc_output_full[..., 0], axis=-1) - np.std(auc_output_full[..., 0], axis=-1),
                        np.mean(auc_output_full[..., 0], axis=-1) + np.std(auc_output_full[..., 0], axis=-1), color='r', alpha=0.2)
    plt.xlabel('# of windows provided to the model')
    plt.ylabel('AUC')

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('sinPB: LDA Classification ACC and AUC for different dynamics')
    plt.savefig('/home/jyao/Downloads/biomarker_id/figures/dynamics/sinPB_acc_auc.png', dpi=300)
    plt.close(fig)

    # now plot the line plots for sfsPB
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(idx, np.mean(acc_output_full_sfs[..., -1], axis=-1), 'r')
    plt.fill_between(idx, np.mean(acc_output_full_sfs[..., -1], axis=-1) - np.std(acc_output_full_sfs[..., -1], axis=-1),
                        np.mean(acc_output_full_sfs[..., -1], axis=-1) + np.std(acc_output_full_sfs[..., -1], axis=-1), color='r', alpha=0.2)
    plt.xlabel('# of windows provided to the model')
    plt.ylabel('Accuracy')

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplot(1, 2, 2)
    plt.plot(idx, np.mean(auc_output_full_sfs[..., -1], axis=-1), 'r')
    plt.fill_between(idx, np.mean(auc_output_full_sfs[..., -1], axis=-1) - np.std(auc_output_full_sfs[..., -1], axis=-1),
                    np.mean(auc_output_full_sfs[..., -1], axis=-1) + np.std(auc_output_full_sfs[..., -1], axis=-1), color='r', alpha=0.2)
    plt.xlabel('# of windows provided to the model')
    plt.ylabel('AUC')

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('sfsPB: LDA Classification ACC and AUC for different dynamics')
    plt.savefig('/home/jyao/Downloads/biomarker_id/figures/dynamics/sfsPB_acc_auc.png', dpi=300)
    plt.close(fig)


    print('debug')


if __name__ == '__main__':
    plot_dynamics_results()