import pickle
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

# import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.stats as stats


def load_model_id_results(output, n_pb=4):
    # load the sinPB results
    acc_python_full = np.stack(
        [output["sinPB"][i]["vec_acc"][:n_pb] for i in range(len(output["sinPB"]))],
        axis=1,
    ).T
    auc_python_full = np.stack(
        [output["sinPB"][i]["vec_auc"][:n_pb] for i in range(len(output["sinPB"]))],
        axis=1,
    ).T
    acc_python_top = [
        output["sinPB"][i]["vec_acc"][0] for i in range(len(output["sinPB"]))
    ]
    auc_python_top = [
        output["sinPB"][i]["vec_auc"][0] for i in range(len(output["sinPB"]))
    ]

    # load the sfsPB results
    acc_python_full_sfs = np.stack(
        [output["sfsPB"][i]["vec_acc"][:n_pb] for i in range(len(output["sfsPB"]))],
        axis=1,
    ).T
    auc_python_full_sfs = np.stack(
        [output["sfsPB"][i]["vec_auc"][:n_pb] for i in range(len(output["sfsPB"]))],
        axis=1,
    ).T
    acc_python_top_sfs = [
        output["sfsPB"][i]["vec_acc"][-1] for i in range(len(output["sfsPB"]))
    ]
    auc_python_top_sfs = [
        output["sfsPB"][i]["vec_auc"][-1] for i in range(len(output["sfsPB"]))
    ]

    return (
        acc_python_full,
        auc_python_full,
        acc_python_top,
        auc_python_top,
        acc_python_full_sfs,
        auc_python_full_sfs,
        acc_python_top_sfs,
        auc_python_top_sfs,
    )


def plot_lda_matlab_results(vec_str_subject, vec_str_side, str_model_id="model_id", n_pb=4, n_rep=5):
    # quick function for comparing matlab with python results

    vec_auc_python_full = []
    vec_auc_python_full_sfs = []
    
    vec_auc_matlab = []
    vec_auc_matlab_sfs = []
    for i in range(len(vec_str_subject)):
        str_subject = vec_str_subject[i]
        str_side = vec_str_side[i]

        # load the results
        p_output = pathlib.Path(
            f"/home/jyao/Downloads/biomarker_id/{str_model_id}/{str_subject}"
        )
        f_output_python = f"{str_subject}_{str_side}_med_avg_auc_LDA.pkl"
        f_output_matlab = f"{str_subject}_{str_side}_med_level_stats.mat"

        # create output figure folder
        p_figure_output = pathlib.Path(
            f"/home/jyao/Downloads/biomarker_id/figures/{str_model_id}/{str_subject}"
        )
        p_figure_output.mkdir(parents=True, exist_ok=True)

        # load in the python values
        output = pickle.load(open(str(p_output / f_output_python), "rb"))
        (
            _,
            auc_python_full,
            _,
            _,
            _,
            auc_python_full_sfs,
            _,
            _,
        ) = load_model_id_results(output)
        
        # append to the outer list
        vec_auc_python_full.append(auc_python_full)
        vec_auc_python_full_sfs.append(auc_python_full_sfs)
        
        # laod the MATLAB results
        mat = sio.loadmat(str(p_output / f_output_matlab))
        auc_matlab = mat["auc_sin"]
        auc_matlab_sfs = mat["auc_sfs"]
        
        vec_auc_matlab.append(auc_matlab)
        vec_auc_matlab_sfs.append(auc_matlab_sfs)
        
        if str_subject == "RCS18":
            fig = plt.figure()
            
            for i in range(n_pb):
                plt.subplot(1, 4, i + 1)
                dict_results_auc = {'Program': [], "AUC": []}
                for j in range(n_rep):
                    dict_results_auc["Program"].append("Python")
                    dict_results_auc["AUC"].append(auc_python_full[j, i])
                
                dict_results_auc["Program"].append("MATLAB")
                dict_results_auc["AUC"].append(auc_matlab[i, 0])
                
                # plot
                results_auc = pd.DataFrame(dict_results_auc)
                sns.boxplot(data=results_auc, x='Program', y='AUC', orient='v')
                ax = plt.gca()

                # set the labels
                if i == 0:
                    ax.set(ylabel='AUC')
                ax.set(xlabel=None)
                ax.set(title=f"PB{i + 1}")
                ax.set(ylim=[0.5, 1])

                # # remove the top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            fig.savefig(f"{p_figure_output}/auc_comparison.png", dpi=300)
            plt.close(fig)
            t1 = 1
             
    
    # # reshape and form pandas dataframe
    # vec_auc_python_full = np.stack(vec_auc_python_full, axis=0)
    # vec_auc_python_full_sfs = np.stack(vec_auc_python_full_sfs, axis=0)
    # vec_auc_matlab = np.stack(vec_auc_matlab, axis=0)
    # vec_auc_matlab_sfs = np.stack(vec_auc_matlab_sfs, axis=0)

    # # now loop
    # for i in range(n_pb):
    # results_dict = {"str_subject": [], "AUC": [], "PB": [], "Method": []}
    # results_dict_sfs = {"str_subject": [], "AUC": [], "PB": [], "Method": []}
    # for i in range(len(vec_str_subject)): 
    #     # load python   
    #     for j in range(n_pb):
    #         for k in range(n_rep):
    #             # load sin PB
    #             results_dict["str_subject"].append(vec_str_subject[i])
    #             results_dict["PB"].append(j + 1)
    #             results_dict["AUC"].append(vec_auc_python_full[i, k, j])
    #             results_dict["Method"].append("Python")
                
    #             # load sfs PB
    #             results_dict_sfs["str_subject"].append(vec_str_subject[i])
    #             results_dict_sfs["PB"].append(j + 1)
    #             results_dict_sfs["AUC"].append(vec_auc_python_full_sfs[i, k, j])
    #             results_dict_sfs["Method"].append("Python")
                
    #     # load MATLAB
    #     for j in range(n_pb):
    # t1 = 1
        
        
        
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


def plot_lda_results(str_model_id="model_id", str_subject="RCS02", str_side="R"):
    
    if str_model_id == "model_id_woart":
        # hard code all paths
        p_output = pathlib.Path(
            "/home/jyao/Downloads/biomarker_id/dynamics_woart/{}".format(str_subject)
        )
        f_output_python = f"{str_subject}_{str_side}_med_avg_auc_LDA_dynamics_woart.pkl"
        
        output_full = pickle.load(open(str(p_output / f_output_python), "rb"))
        
        output = dict()
        output['sinPB'] = output_full['sinPB']["n_dynamics_1"]
        output['sfsPB'] = output_full['sfsPB']["n_dynamics_1"]
        (
            acc_python_full,
            auc_python_full,
            acc_python_top,
            auc_python_top,
            acc_python_full_sfs,
            auc_python_full_sfs,
            acc_python_top_sfs,
            auc_python_top_sfs,
        ) = load_model_id_results(output)
        
    else:
        # hard code all paths
        p_output = pathlib.Path(
            "/home/jyao/Downloads/biomarker_id/{}/{}".format(str_model_id, str_subject)
        )
        f_output_python = f"{str_subject}_{str_side}_med_avg_auc_LDA.pkl"
        
        output = pickle.load(open(str(p_output / f_output_python), "rb"))
        (
            acc_python_full,
            auc_python_full,
            acc_python_top,
            auc_python_top,
            acc_python_full_sfs,
            auc_python_full_sfs,
            acc_python_top_sfs,
            auc_python_top_sfs,
        ) = load_model_id_results(output)
        
        

    p_figure_output = pathlib.Path(
        "/home/jyao/Downloads/biomarker_id/figures/{}/{}".format(str_model_id, str_subject)
    )
    p_figure_output.mkdir(parents=True, exist_ok=True)

    # load in the python values


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
    f_output_python_QDA = f"{str_subject}_{str_side}_med_avg_auc_QDA.pkl"
    f_output_python_SVM = f"{str_subject}_{str_side}_med_avg_auc_SVM.pkl"
    f_output_python_RF = f"{str_subject}_{str_side}_med_avg_auc_RF.pkl"
    f_output_python_MLP = f"{str_subject}_{str_side}_med_avg_auc_MLP.pkl"
    f_output_python_RNN = f"{str_subject}_{str_side}_med_avg_auc_RNN.pkl"

    output_QDA = pickle.load(open(str(p_output / f_output_python_QDA), "rb"))
    (
        _,
        auc_python_full_QDA,
        _,
        _,
        _,
        auc_python_full_sfs_QDA,
        _,
        _,
    ) = load_model_id_results(output_QDA)

    output_SVM = pickle.load(open(str(p_output / f_output_python_SVM), "rb"))
    (
        _,
        auc_python_full_SVM,
        _,
        _,
        _,
        auc_python_full_sfs_SVM,
        _,
        _,
    ) = load_model_id_results(output_SVM)

    output_RF = pickle.load(open(str(p_output / f_output_python_RF), "rb"))
    (
        _,
        auc_python_full_RF,
        _,
        _,
        _,
        auc_python_full_sfs_RF,
        _,
        _,
    ) = load_model_id_results(output_RF)

    output_MLP = pickle.load(open(str(p_output / f_output_python_MLP), "rb"))
    (
        _,
        auc_python_full_MLP,
        _,
        _,
        _,
        auc_python_full_sfs_MLP,
        _,
        _,
    ) = load_model_id_results(output_MLP)

    output_RNN = pickle.load(open(str(p_output / f_output_python_RNN), "rb"))
    (
        _,
        auc_python_full_RNN,
        _,
        _,
        _,
        auc_python_full_sfs_RNN,
        _,
        _,
    ) = load_model_id_results(output_RNN)

    fig = plt.figure()
    idx = np.arange(1, 5, 1)
    x_ticks = [str(x) for x in idx]
    plt.plot(idx, np.mean(auc_python_full_sfs, axis=0), "r", label="LDA")
    plt.fill_between(
        idx,
        np.mean(auc_python_full_sfs, axis=0) - np.std(auc_python_full_sfs, axis=0),
        np.mean(auc_python_full_sfs, axis=0) + np.std(auc_python_full_sfs, axis=0),
        color="r",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_python_full_sfs_QDA, axis=0), "k", label="QDA")
    plt.fill_between(
        idx,
        np.mean(auc_python_full_sfs_QDA, axis=0)
        - np.std(auc_python_full_sfs_QDA, axis=0),
        np.mean(auc_python_full_sfs_QDA, axis=0)
        + np.std(auc_python_full_sfs_QDA, axis=0),
        color="k",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_python_full_sfs_SVM, axis=0), "g", label="SVM")
    plt.fill_between(
        idx,
        np.mean(auc_python_full_sfs_SVM, axis=0)
        - np.std(auc_python_full_sfs_SVM, axis=0),
        np.mean(auc_python_full_sfs_SVM, axis=0)
        + np.std(auc_python_full_sfs_SVM, axis=0),
        color="g",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_python_full_sfs_RF, axis=0), "m", label="RF")
    plt.fill_between(
        idx,
        np.mean(auc_python_full_sfs_RF, axis=0)
        - np.std(auc_python_full_sfs_RF, axis=0),
        np.mean(auc_python_full_sfs_RF, axis=0)
        + np.std(auc_python_full_sfs_RF, axis=0),
        color="m",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_python_full_sfs_MLP, axis=0), "b", label="MLP")
    plt.fill_between(
        idx,
        np.mean(auc_python_full_sfs_MLP, axis=0)
        - np.std(auc_python_full_sfs_MLP, axis=0),
        np.mean(auc_python_full_sfs_MLP, axis=0)
        + np.std(auc_python_full_sfs_MLP, axis=0),
        color="b",
        alpha=0.2,
    )

    plt.plot(idx, np.mean(auc_python_full_sfs_RNN, axis=0), "c", label="RNN")
    plt.fill_between(
        idx,
        np.mean(auc_python_full_sfs_RNN, axis=0)
        - np.std(auc_python_full_sfs_RNN, axis=0),
        np.mean(auc_python_full_sfs_RNN, axis=0)
        + np.std(auc_python_full_sfs_RNN, axis=0),
        color="c",
        alpha=0.2,
    )

    plt.xlabel("Number of Power Bands Included")
    plt.ylabel("AUC with Algorithms")
    plt.title(f"Comparison of AUC for Top 4 Power Bands: {str_subject}_{str_side}")
    plt.legend(frameon=False)
    plt.xticks(idx, x_ticks)

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(
        str(
            p_figure_output
            / f"{str_subject}_{str_side}_med_level_auc_comp_top5_combo.png"
        ),
        dpi=300,
    )
    plt.close(fig)

    t1 = 1


if __name__ == "__main__":
    # plot_lda_matlab_results(["RCS02", "RCS11", "RCS12", "RCS18"], ["R", "L", "L", "L"])
    # plot_lda_results(str_subject="RCS02", str_side="R")
    plot_lda_results(str_subject="RCS08", str_side="R")
    # plot_lda_results(str_subject="RCS11", str_side="L")
    # plot_lda_results(str_subject="RCS12", str_side="L")
    # plot_lda_results(str_subject="RCS18", str_side="L")
    # plot_lda_results(str_subject="RCS17", str_side="L")
    
    
    # plot_lda_results(str_subject="RCS02", str_side="R", str_model_id='model_id_woart')
#