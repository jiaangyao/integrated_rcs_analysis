import pickle
import pathlib

import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.stats as stats


def load_dyna_summary(output):
    # obtain the full acc
    acc_output_full = []
    for i in range(1, len(output["sinPB"]) + 1):
        str_n_dyna = "n_dynamics_{}".format(i)
        acc_output_n_dyna_curr = np.stack(
            [
                output["sinPB"][str_n_dyna][0][j]["vec_acc"][:5]
                for j in range(len(output["sinPB"][str_n_dyna][0]))
            ]
        )
        acc_output_full.append(acc_output_n_dyna_curr)
    acc_output_full = np.stack(acc_output_full, axis=0)

    # obtain the full auc
    auc_output_full = []
    for i in range(1, len(output["sinPB"]) + 1):
        str_n_dyna = "n_dynamics_{}".format(i)
        auc_output_n_dyna_curr = np.stack(
            [
                output["sinPB"][str_n_dyna][0][j]["vec_auc"][:5]
                for j in range(len(output["sinPB"][str_n_dyna][0]))
            ]
        )
        auc_output_full.append(auc_output_n_dyna_curr)
    auc_output_full = np.stack(auc_output_full, axis=0)

    # obtain the full acc from SFS
    acc_output_full_sfs = []
    for i in range(1, len(output["sfsPB"]) + 1):
        str_n_dyna = "n_dynamics_{}".format(i)
        auc_output_n_dyna_curr = np.stack(
            [
                output["sfsPB"][str_n_dyna][0][j]["vec_acc"][:5]
                for j in range(len(output["sfsPB"][str_n_dyna][0]))
            ]
        )
        acc_output_full_sfs.append(auc_output_n_dyna_curr)
    acc_output_full_sfs = np.stack(acc_output_full_sfs, axis=0)

    # obtain the full auc from SFS
    auc_output_full_sfs = []
    for i in range(1, len(output["sfsPB"]) + 1):
        str_n_dyna = "n_dynamics_{}".format(i)
        auc_output_n_dyna_curr = np.stack(
            [
                output["sfsPB"][str_n_dyna][0][j]["vec_auc"][:5]
                for j in range(len(output["sfsPB"][str_n_dyna][0]))
            ]
        )
        auc_output_full_sfs.append(auc_output_n_dyna_curr)
    auc_output_full_sfs = np.stack(auc_output_full_sfs, axis=0)

    return acc_output_full, auc_output_full, acc_output_full_sfs, auc_output_full_sfs

def load_model_id_results(output):
    acc_python_full = np.stack([output['sinPB'][i]['vec_acc'][:5] for i in range(len(output['sinPB']))], axis=1).T
    auc_python_full = np.stack([output['sinPB'][i]['vec_auc'][:5] for i in range(len(output['sinPB']))], axis=1).T
    acc_python_top = [output['sinPB'][i]['vec_acc'][0] for i in range(len(output['sinPB']))]
    auc_python_top = [output['sinPB'][i]['vec_auc'][0] for i in range(len(output['sinPB']))]

    acc_python_full_sfs = np.stack([output['sfsPB'][i]['vec_acc'][:5] for i in range(len(output['sfsPB']))], axis=1).T
    auc_python_full_sfs = np.stack([output['sfsPB'][i]['vec_auc'][:5] for i in range(len(output['sfsPB']))], axis=1).T
    acc_python_top_sfs = [output['sfsPB'][i]['vec_acc'][-1] for i in range(len(output['sfsPB']))]
    auc_python_top_sfs = [output['sfsPB'][i]['vec_auc'][-1] for i in range(len(output['sfsPB']))]

    return acc_python_top, auc_python_top, acc_python_top_sfs, auc_python_top_sfs


def plot_dynamics_results(str_subject="RCS02", str_side="R", str_dynamics_id='dynamics', str_model_id = 'model_id'):
    # hard code all paths
    p_output = pathlib.Path("/home/jyao/Downloads/biomarker_id/{}/{}".format(str_dynamics_id, str_subject))
    f_output_python = f"{str_subject}_{str_side}_med_avg_auc_LDA_dynamics.pkl"
    f_output_qda = f"{str_subject}_{str_side}_med_avg_auc_QDA_dynamics.pkl"
    f_output_svm = f"{str_subject}_{str_side}_med_avg_auc_SVM_dynamics.pkl"
    f_output_mlp = f"{str_subject}_{str_side}_med_avg_auc_MLP_dynamics.pkl"
    f_output_rnn = f"{str_subject}_{str_side}_med_avg_auc_RNN_dynamics.pkl"

    # load in the python values
    output = pickle.load(open(str(p_output / f_output_python), "rb"))
    (
        acc_output_full,
        auc_output_full,
        acc_output_full_sfs,
        auc_output_full_sfs,
    ) = load_dyna_summary(output)
    idx = np.arange(1, 6, 1)

    # also load in the various other models
    output_qda = pickle.load(open(str(p_output / f_output_qda), "rb"))
    (
        acc_output_full_qda,
        auc_output_full_qda,
        acc_output_full_sfs_qda,
        auc_output_full_sfs_qda,
    ) = load_dyna_summary(output_qda)

    output_svm = pickle.load(open(str(p_output / f_output_svm), "rb"))
    (
        acc_output_full_svm,
        auc_output_full_svm,
        acc_output_full_sfs_svm,
        auc_output_full_sfs_svm,
    ) = load_dyna_summary(output_svm)
    
    output_mlp = pickle.load(open(str(p_output / f_output_mlp), "rb"))
    (
        acc_output_full_mlp,
        auc_output_full_mlp,
        acc_output_full_sfs_mlp,
        auc_output_full_sfs_mlp,
    ) = load_dyna_summary(output_mlp)
    
    output_rnn = pickle.load(open(str(p_output / f_output_rnn), "rb"))
    (
        acc_output_full_rnn,
        auc_output_full_rnn,
        acc_output_full_sfs_rnn,
        auc_output_full_sfs_rnn,
    ) = load_dyna_summary(output_rnn)
    

    # now plot the line plots for the top5 auc
    fig = plt.figure(figsize=(12, 6))
    idx = np.arange(1, 6, 1)
    x_ticks = [str(x) for x in idx]
    plt.subplot(1, 2, 1)
    plt.plot(idx, np.mean(acc_output_full[..., 0], axis=-1), "r", label="LDA")
    plt.fill_between(
        idx,
        np.mean(acc_output_full[..., 0], axis=-1)
        - np.std(acc_output_full[..., 0], axis=-1),
        np.mean(acc_output_full[..., 0], axis=-1)
        + np.std(acc_output_full[..., 0], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(acc_output_full_qda[..., 0], axis=-1), "k", label="QDA")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_qda[..., 0], axis=-1)
        - np.std(acc_output_full_qda[..., 0], axis=-1),
        np.mean(acc_output_full_qda[..., 0], axis=-1)
        + np.std(acc_output_full_qda[..., 0], axis=-1),
        color="k",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(acc_output_full_svm[..., 0], axis=-1), "g", label="SVM")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_svm[..., 0], axis=-1)
        - np.std(acc_output_full_svm[..., 0], axis=-1),
        np.mean(acc_output_full_svm[..., 0], axis=-1)
        + np.std(acc_output_full_svm[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    plt.plot(idx, np.mean(acc_output_full_mlp[..., 0], axis=-1), "b", label="MLP")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_mlp[..., 0], axis=-1)
        - np.std(acc_output_full_mlp[..., 0], axis=-1),
        np.mean(acc_output_full_mlp[..., 0], axis=-1)
        + np.std(acc_output_full_mlp[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    plt.plot(idx, np.mean(acc_output_full_rnn[..., 0], axis=-1), "c", label="RNN")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_rnn[..., 0], axis=-1)
        - np.std(acc_output_full_rnn[..., 0], axis=-1),
        np.mean(acc_output_full_rnn[..., 0], axis=-1)
        + np.std(acc_output_full_rnn[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )

    plt.xlabel("# of windows provided to the model")
    plt.ylabel("Accuracy")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    plt.plot(idx, np.mean(auc_output_full[..., 0], axis=-1), "r", label='LDA')
    plt.fill_between(
        idx,
        np.mean(auc_output_full[..., 0], axis=-1)
        - np.std(auc_output_full[..., 0], axis=-1),
        np.mean(auc_output_full[..., 0], axis=-1)
        + np.std(auc_output_full[..., 0], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_output_full_qda[..., 0], axis=-1), "k", label="QDA")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_qda[..., 0], axis=-1)
        - np.std(auc_output_full_qda[..., 0], axis=-1),
        np.mean(auc_output_full_qda[..., 0], axis=-1)
        + np.std(auc_output_full_qda[..., 0], axis=-1),
        color="k",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_output_full_svm[..., 0], axis=-1), "g", label="SVM")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_svm[..., 0], axis=-1)
        - np.std(auc_output_full_svm[..., 0], axis=-1),
        np.mean(auc_output_full_svm[..., 0], axis=-1)
        + np.std(auc_output_full_svm[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    plt.plot(idx, np.mean(auc_output_full_mlp[..., 0], axis=-1), "b", label="MLP")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_mlp[..., 0], axis=-1)
        - np.std(auc_output_full_mlp[..., 0], axis=-1),
        np.mean(auc_output_full_mlp[..., 0], axis=-1)
        + np.std(auc_output_full_mlp[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    plt.plot(idx, np.mean(auc_output_full_rnn[..., 0], axis=-1), "c", label="RNN")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_rnn[..., 0], axis=-1)
        - np.std(auc_output_full_rnn[..., 0], axis=-1),
        np.mean(auc_output_full_rnn[..., 0], axis=-1)
        + np.std(auc_output_full_rnn[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )


    plt.xlabel("# of windows provided to the model")
    plt.ylabel("AUC")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.suptitle(f"sinPB: LDA Classification ACC and AUC for different dynamics - {str_subject}{str_side}")
    plt.savefig(
        f"/home/jyao/Downloads/biomarker_id/figures/{str_dynamics_id}/{str_subject}/{str_subject}_{str_side}_sinPB_acc_auc.png", dpi=300
    )
    plt.close(fig)

    # now plot the line plots for sfsPB
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(idx, np.mean(acc_output_full_sfs[..., -1], axis=-1), "r", label='LDA')
    plt.fill_between(
        idx,
        np.mean(acc_output_full_sfs[..., -1], axis=-1)
        - np.std(acc_output_full_sfs[..., -1], axis=-1),
        np.mean(acc_output_full_sfs[..., -1], axis=-1)
        + np.std(acc_output_full_sfs[..., -1], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(acc_output_full_sfs_qda[..., 0], axis=-1), "k", label="QDA")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_sfs_qda[..., 0], axis=-1)
        - np.std(acc_output_full_sfs_qda[..., 0], axis=-1),
        np.mean(acc_output_full_sfs_qda[..., 0], axis=-1)
        + np.std(acc_output_full_sfs_qda[..., 0], axis=-1),
        color="k",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(acc_output_full_sfs_svm[..., 0], axis=-1), "g", label="SVM")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_sfs_svm[..., 0], axis=-1)
        - np.std(acc_output_full_sfs_svm[..., 0], axis=-1),
        np.mean(acc_output_full_sfs_svm[..., 0], axis=-1)
        + np.std(acc_output_full_sfs_svm[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    plt.plot(idx, np.mean(acc_output_full_sfs_mlp[..., 0], axis=-1), "b", label="MLP")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_sfs_mlp[..., 0], axis=-1)
        - np.std(acc_output_full_sfs_mlp[..., 0], axis=-1),
        np.mean(acc_output_full_sfs_mlp[..., 0], axis=-1)
        + np.std(acc_output_full_sfs_mlp[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    
    plt.plot(idx, np.mean(acc_output_full_sfs_rnn[..., 0], axis=-1), "c", label="RNN")
    plt.fill_between(
        idx,
        np.mean(acc_output_full_sfs_rnn[..., 0], axis=-1)
        - np.std(acc_output_full_sfs_rnn[..., 0], axis=-1),
        np.mean(acc_output_full_sfs_rnn[..., 0], axis=-1)
        + np.std(acc_output_full_sfs_rnn[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    

    plt.xlabel("# of windows provided to the model")
    plt.ylabel("Accuracy")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    plt.plot(idx, np.mean(auc_output_full_sfs[..., -1], axis=-1), "r", label='LDA')
    plt.fill_between(
        idx,
        np.mean(auc_output_full_sfs[..., -1], axis=-1)
        - np.std(auc_output_full_sfs[..., -1], axis=-1),
        np.mean(auc_output_full_sfs[..., -1], axis=-1)
        + np.std(auc_output_full_sfs[..., -1], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_output_full_sfs_qda[..., 0], axis=-1), "k", label="QDA")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_sfs_qda[..., 0], axis=-1)
        - np.std(auc_output_full_sfs_qda[..., 0], axis=-1),
        np.mean(auc_output_full_sfs_qda[..., 0], axis=-1)
        + np.std(auc_output_full_sfs_qda[..., 0], axis=-1),
        color="k",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_output_full_sfs_svm[..., 0], axis=-1), "g", label="SVM")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_sfs_svm[..., 0], axis=-1)
        - np.std(auc_output_full_sfs_svm[..., 0], axis=-1),
        np.mean(auc_output_full_sfs_svm[..., 0], axis=-1)
        + np.std(auc_output_full_sfs_svm[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    plt.plot(idx, np.mean(auc_output_full_sfs_mlp[..., 0], axis=-1), "b", label="MLP")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_sfs_mlp[..., 0], axis=-1)
        - np.std(auc_output_full_sfs_mlp[..., 0], axis=-1),
        np.mean(auc_output_full_sfs_mlp[..., 0], axis=-1)
        + np.std(auc_output_full_sfs_mlp[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )
    
    plt.plot(idx, np.mean(auc_output_full_sfs_rnn[..., 0], axis=-1), "c", label="RNN")
    plt.fill_between(
        idx,
        np.mean(auc_output_full_sfs_rnn[..., 0], axis=-1)
        - np.std(auc_output_full_sfs_rnn[..., 0], axis=-1),
        np.mean(auc_output_full_sfs_rnn[..., 0], axis=-1)
        + np.std(auc_output_full_sfs_rnn[..., 0], axis=-1),
        color="g",
        alpha=0.2,
    )


    plt.xlabel("# of windows provided to the model")
    plt.ylabel("AUC")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.suptitle(f"sfsPB: LDA Classification ACC and AUC for different dynamics - {str_subject}{str_side}")
    plt.savefig(
        f"/home/jyao/Downloads/biomarker_id/figures/{str_dynamics_id}/{str_subject}/{str_subject}_{str_side}_sfsPB_acc_auc.png", dpi=300
    )
    plt.close(fig)

    # load the various model id results
    p_output = pathlib.Path(f'/home/jyao/Downloads/biomarker_id/{str_model_id}/{str_subject}')
    f_output_python = f'{str_subject}_{str_side}_med_avg_auc_LDA.pkl'
    f_output_python_UR60 = 'RCS02_R_med_avg_auc_LDA_UR60.pkl'
    f_output_python_UR90 = 'RCS02_R_med_avg_auc_LDA_UR90.pkl'
    f_output_python_UR120 = 'RCS02_R_med_avg_auc_LDA_UR120.pkl'
    f_output_python_UR150 = 'RCS02_R_med_avg_auc_LDA_UR150.pkl'
    f_output_python_UR180 = 'RCS02_R_med_avg_auc_LDA_UR180.pkl'
    vec_f_output = [f_output_python, f_output_python_UR60, f_output_python_UR90, 
                    f_output_python_UR120, f_output_python_UR150, f_output_python_UR180]

    # load the various model id results
    vec_acc_python_top = []
    vec_auc_python_top = []
    vec_acc_python_top_sfs = []
    vec_auc_python_top_sfs = []
    for i in range(len(vec_f_output)):
        output = pickle.load(open(str(p_output / vec_f_output[i]), 'rb'))
        acc_python_top, auc_python_top, acc_python_top_sfs, auc_python_top_sfs = \
            load_model_id_results(output)  
        
        vec_acc_python_top.append(acc_python_top)
        vec_auc_python_top.append(auc_python_top)
        vec_acc_python_top_sfs.append(acc_python_top_sfs)
        vec_auc_python_top_sfs.append(auc_python_top_sfs)
    vec_acc_python_top = np.stack(vec_acc_python_top, axis=-1).T
    vec_auc_python_top = np.stack(vec_auc_python_top, axis=-1).T
    vec_acc_python_top_sfs = np.stack(vec_acc_python_top_sfs, axis=-1).T
    vec_auc_python_top_sfs = np.stack(vec_auc_python_top_sfs, axis=-1).T

    # now plot the line plots for sinPB
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(idx, np.mean(acc_output_full[..., 0], axis=-1), "r", label='LDA DYNA')
    plt.fill_between(
        idx,
        np.mean(acc_output_full[..., 0], axis=-1)
        - np.std(acc_output_full[..., 0], axis=-1),
        np.mean(acc_output_full[..., 0], axis=-1)
        + np.std(acc_output_full[..., 0], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(vec_acc_python_top, axis=-1), "b", label='LDA AVG')
    plt.fill_between(
        idx,
        np.mean(vec_acc_python_top, axis=-1)
        - np.std(vec_acc_python_top, axis=-1),
        np.mean(vec_acc_python_top, axis=-1)
        + np.std(vec_acc_python_top, axis=-1),
        color="b",
        alpha=0.2,
    )

    plt.xlabel("# of windows provided to the model")
    plt.ylabel("Accuracy")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    plt.plot(idx, np.mean(auc_output_full[..., 0], axis=-1), "r", label='LDA DYNA')
    plt.fill_between(
        idx,
        np.mean(auc_output_full[..., 0], axis=-1)
        - np.std(auc_output_full[..., 0], axis=-1),
        np.mean(auc_output_full[..., 0], axis=-1)
        + np.std(auc_output_full[..., 0], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(vec_auc_python_top, axis=-1), "b", label='LDA AVG')
    plt.fill_between(
        idx,
        np.mean(vec_auc_python_top, axis=-1)
        - np.std(vec_auc_python_top, axis=-1),
        np.mean(vec_auc_python_top, axis=-1)
        + np.std(vec_auc_python_top, axis=-1),
        color="b",
        alpha=0.2,
    )

    plt.xlabel("# of windows provided to the model")
    plt.ylabel("AUC")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.suptitle(f"sinPB: LDA Classification ACC and AUC for Dynamics vs Avg - {str_subject}{str_side}")
    plt.savefig(
        f"/home/jyao/Downloads/biomarker_id/figures/{str_dynamics_id}/{str_subject}/{str_subject}_{str_side}_sfsPB_acc_auc_dyna_avg_sinPB.png", dpi=300
    )
    plt.close(fig)
    
    
    # now plot the line plots for sfsPB
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(idx, np.mean(acc_output_full_sfs[..., -1], axis=-1), "r", label='LDA DYNA')
    plt.fill_between(
        idx,
        np.mean(acc_output_full_sfs[..., -1], axis=-1)
        - np.std(acc_output_full_sfs[..., -1], axis=-1),
        np.mean(acc_output_full_sfs[..., -1], axis=-1)
        + np.std(acc_output_full_sfs[..., -1], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(vec_acc_python_top_sfs, axis=-1), "b", label='LDA AVG')
    plt.fill_between(
        idx,
        np.mean(vec_acc_python_top_sfs, axis=-1)
        - np.std(vec_acc_python_top_sfs, axis=-1),
        np.mean(vec_acc_python_top_sfs, axis=-1)
        + np.std(vec_acc_python_top_sfs, axis=-1),
        color="b",
        alpha=0.2,
    )

    plt.xlabel("# of windows provided to the model")
    plt.ylabel("Accuracy")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)
    plt.plot(idx, np.mean(auc_output_full_sfs[..., -1], axis=-1), "r", label='LDA DYNA')
    plt.fill_between(
        idx,
        np.mean(auc_output_full_sfs[...,-1], axis=-1)
        - np.std(auc_output_full_sfs[...,-1], axis=-1),
        np.mean(auc_output_full_sfs[...,-1], axis=-1)
        + np.std(auc_output_full_sfs[...,-1], axis=-1),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(vec_auc_python_top_sfs, axis=-1), "b", label='LDA AVG')
    plt.fill_between(
        idx,
        np.mean(vec_auc_python_top_sfs, axis=-1)
        - np.std(vec_auc_python_top_sfs, axis=-1),
        np.mean(vec_auc_python_top_sfs, axis=-1)
        + np.std(vec_auc_python_top_sfs, axis=-1),
        color="b",
        alpha=0.2,
    )

    plt.xlabel("# of windows provided to the model")
    plt.ylabel("AUC")

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)

    plt.suptitle(f"sfsPB: LDA Classification ACC and AUC for Dynamics vs Avg - {str_subject}{str_side}")
    plt.savefig(
        f"/home/jyao/Downloads/biomarker_id/figures/dynamics/{str_subject}/{str_subject}_{str_side}_sfsPB_acc_auc_dyna_avg_sfsPB.png", dpi=300
    )
    plt.close(fig)
    print("debug")


if __name__ == "__main__":
    # plot_dynamics_results()
    plot_dynamics_results(str_subject="RCS02", str_side="R")
    plot_dynamics_results(str_subject="RCS08", str_side="R")
    plot_dynamics_results(str_subject="RCS11", str_side="L")
    plot_dynamics_results(str_subject="RCS12", str_side="L")
    plot_dynamics_results(str_subject="RCS18", str_side="L")