import pickle
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

# import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels
import scipy.io as sio
import scipy.stats as stats
from statannotations.Annotator import Annotator

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
    # acc_python_top = [
    #     output["sinPB"][i]["vec_acc"][0] for i in range(len(output["sinPB"]))
    # ]
    # auc_python_top = [
    #     output["sinPB"][i]["vec_auc"][0] for i in range(len(output["sinPB"]))
    # ]

    # load the sfsPB results
    acc_python_full_sfs = np.stack(
        [output["sfsPB"][i]["vec_acc"][:n_pb] for i in range(len(output["sfsPB"]))],
        axis=1,
    ).T
    auc_python_full_sfs = np.stack(
        [output["sfsPB"][i]["vec_auc"][:n_pb] for i in range(len(output["sfsPB"]))],
        axis=1,
    ).T
    # acc_python_top_sfs = [
    #     output["sfsPB"][i]["vec_acc"][-1] for i in range(len(output["sfsPB"]))
    # ]
    # auc_python_top_sfs = [
    #     output["sfsPB"][i]["vec_auc"][-1] for i in range(len(output["sfsPB"]))
    # ]

    # load the PB results
    sinPB = [output["sinPB"][i]["sinPB"][:n_pb] for i in range(len(output["sinPB"]))]
    sfsPB = [output["sfsPB"][i]["sfsPB"][:n_pb] for i in range(len(output["sfsPB"]))]

    return (
        acc_python_full,
        auc_python_full,
        sinPB,
        acc_python_full_sfs,
        auc_python_full_sfs,
        sfsPB,
    )


def plot_python_matlab_diff_sub(
    auc_python,
    auc_matlab,
    n_pb,
    n_rep,
    p_figure_output,
    f_figure_output,
    figsize=(10, 6),
):
    fig = plt.figure(figsize=figsize)
    for i in range(n_pb):
        plt.subplot(1, 4, i + 1)

        # append the python results
        dict_results_auc = {"Program": [], "AUC": []}
        for j in range(n_rep):
            dict_results_auc["Program"].append("Python")
            dict_results_auc["AUC"].append(auc_python[j, i])

        # append the MATLAB results for both
        dict_results_auc["Program"].append("MATLAB")
        dict_results_auc["AUC"].append(auc_matlab[i, 0])

        # compute the mean difference
        mean_diff = np.abs(np.mean(auc_python[:, i]) - auc_matlab[i, 0])

        # plot
        results_auc = pd.DataFrame(dict_results_auc)
        sns.boxplot(data=results_auc, x="Program", y="AUC", orient="v")
        ax = plt.gca()

        # set the labels
        if i == 0:
            ax.set(ylabel="AUC")
        else:
            ax.set(ylabel=None)
        ax.set(xlabel=None)
        ax.set(title=f"PB{i + 1}, mean diff: {mean_diff:.2E}")
        ax.set(ylim=[0.5, 1])

        # remove the top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # get tight layout
    plt.tight_layout()

    # save output figure
    fig.savefig(f"{p_figure_output}/{f_figure_output}.png", dpi=300)
    plt.close(fig)


def plot_model_comp(
    vec_metric_python_full,
    vec_str_model,
    vec_colormap,
    str_subject,
    str_side,
    str_metric,
    p_figure_output,
    f_figure_output,
    figsize=(8, 6),
):
    # create the index for plotting
    fig = plt.figure(figsize=figsize)
    idx = np.arange(1, vec_metric_python_full[0].shape[1] + 1, 1)
    x_ticks = [str(x) for x in idx]

    # now loop through the different models
    for i in range(len(vec_metric_python_full)):
        vec_metric_python_curr = vec_metric_python_full[i]

        plt.plot(
            idx,
            np.mean(vec_metric_python_curr, axis=0),
            vec_colormap[i],
            label=vec_str_model[i],
        )
        plt.fill_between(
            idx,
            np.mean(vec_metric_python_curr, axis=0)
            - np.std(vec_metric_python_curr, axis=0),
            np.mean(vec_metric_python_curr, axis=0)
            + np.std(vec_metric_python_curr, axis=0),
            color=vec_colormap[i],
            alpha=0.2,
        )

    # get the labels right
    plt.xlabel("Number of Power Bands Included")
    plt.ylabel(f"{str_metric} with Algorithms")
    plt.title(
        f"Comparison of {str_metric} for Top 4 Power Bands: {str_subject}_{str_side}"
    )
    plt.legend(frameon=False)
    plt.xticks(idx, x_ticks)

    ax = plt.gca()
    ax.set_xticks(idx)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # save the figure
    fig.savefig(str(p_figure_output / f_figure_output), dpi=300)
    plt.close(fig)


def plot_lda_matlab_results(
    vec_str_subject, vec_str_side, str_model_id="model_id", n_pb=4, n_rep=5
):
    # quick function for comparing matlab with python results

    vec_auc_python_full = []
    vec_sinPB_python = []
    vec_auc_python_full_sfs = []
    vec_sfsPB_python = []

    vec_auc_matlab = []
    vec_sinPB_matlab = []
    vec_auc_matlab_sfs = []
    vec_sfsPB_matlab = []
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
            sinPB,
            _,
            auc_python_full_sfs,
            sfsPB,
        ) = load_model_id_results(output)

        # append to the outer list
        vec_auc_python_full.append(auc_python_full)
        vec_sinPB_python.append(sinPB)
        vec_auc_python_full_sfs.append(auc_python_full_sfs)
        vec_sfsPB_python.append(sfsPB)

        # laod the MATLAB results
        mat = sio.loadmat(str(p_output / f_output_matlab))
        auc_matlab = mat["auc_sin"]
        sinPB_matlab = mat["pb_sin"]
        auc_matlab_sfs = mat["auc_sfs"]
        sfsPB_matlab = mat["pb_sfs"]

        vec_auc_matlab.append(auc_matlab)
        vec_sinPB_matlab.append(sinPB_matlab)
        vec_auc_matlab_sfs.append(auc_matlab_sfs)
        vec_sfsPB_matlab.append(sfsPB_matlab)

        # plot the single PB results
        f_figure_output_sinPB = f"matlab_python_auc_comp_{str_subject}_sinPB"
        plot_python_matlab_diff_sub(
            auc_python_full,
            auc_matlab,
            n_pb,
            n_rep,
            p_figure_output,
            f_figure_output_sinPB,
            figsize=(10, 6),
        )

        # plot the SFS PB results
        f_figure_output_sfsPB = f"matlab_python_auc_comp_{str_subject}_sfsPB"
        plot_python_matlab_diff_sub(
            auc_python_full_sfs,
            auc_matlab_sfs,
            n_pb,
            n_rep,
            p_figure_output,
            f_figure_output_sfsPB,
            figsize=(10, 6),
        )

        # get the best sinPB based on avg AUC from the rep
        sinPB_python_best = sinPB[np.argmax(auc_python_full[:, 0])]
        sfsPB_python_best = sfsPB[np.argmax(auc_python_full_sfs[:, -1])]

        # now print out the power bands
        print(f"Subject: {str_subject}, Side: {str_side}")
        print(f"Python sinPB: {sinPB_python_best}")
        print(f"MATLAB sinPB: {sinPB_matlab}")
        print(f"Python sfsPB: {sfsPB_python_best}")
        print(f"MATLAB sfsPB: {sfsPB_matlab}")

        print("")

    # reshape and form pandas dataframe
    vec_auc_python_full = np.stack(vec_auc_python_full, axis=0)
    vec_auc_python_full_sfs = np.stack(vec_auc_python_full_sfs, axis=0)
    vec_auc_matlab = np.stack(vec_auc_matlab, axis=0)
    vec_auc_matlab_sfs = np.stack(vec_auc_matlab_sfs, axis=0)

    t1 = 1

    # # now loop
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
    
    
def plot_model_comp(vec_str_sub, vec_str_side, vec_str_model):
    # set font size for everything
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # create the output variable
    vec_auc_python_sin_full = []
    vec_auc_python_sfs_full = []
    for i in range(len(vec_str_model)):
        auc_model_sin_full = []
        auc_model_sfs_full = []
        for j in range(len(vec_str_sub)):
            # obtain current subject and side
            str_subject = vec_str_sub[j]
            str_side = vec_str_side[j]
            str_model = vec_str_model[i]
            
            # hard code all paths
            p_output = pathlib.Path(
                "/home/jyao/Downloads/biomarker_id/model_id/{}".format(str_subject)
            )
            f_output_python = f"{str_subject}_{str_side}_med_avg_auc_{str_model}.pkl"
            
            # load the current results
            output = pickle.load(open(str(p_output / f_output_python), "rb"))
            (
                acc_python_full,
                auc_python_full,
                sinPB,
                acc_python_full_sfs,
                auc_python_full_sfs,
                sfsPB,
            ) = load_model_id_results(output)

            # append to the output
            auc_model_sin_full.append(auc_python_full[:, 0])
            auc_model_sfs_full.append(auc_python_full_sfs[:, -1])
            
        # append to the output
        vec_auc_python_sin_full.append(np.stack(auc_model_sin_full, axis=0))
        vec_auc_python_sfs_full.append(np.stack(auc_model_sfs_full, axis=0))
    
    # next form the dictionaries
    n_rep = 5 
    
    dict_results_auc_sin = {"STR_MODEL": [], "AUC": [], "SUB": []}
    dict_avg_results_auc_sin = {"STR_MODEL": [], "AUC": [], "SUB": []}
    
    dict_results_auc_sfs = {"STR_MODEL": [], "AUC": [], "SUB": []}
    dict_avg_results_auc_sfs = {"STR_MODEL": [], "AUC": [], "SUB": []}
    
    for i in range(len(vec_str_model)):
        for j in range(len(vec_str_sub)):
            auc_python_full_sin = vec_auc_python_sin_full[i][j]
            auc_python_full_sfs = vec_auc_python_sfs_full[i][j]
            
            # append the python results
            for k in range(n_rep):
                # append the sin results
                dict_results_auc_sin["STR_MODEL"].append(vec_str_model[i])
                dict_results_auc_sin["AUC"].append(auc_python_full_sin[k])
                dict_results_auc_sin["SUB"].append(vec_str_sub[j] + vec_str_side[j])
                
                # append the sfs result
                dict_results_auc_sfs["STR_MODEL"].append(vec_str_model[i])
                dict_results_auc_sfs["AUC"].append(auc_python_full_sfs[k])
                dict_results_auc_sfs["SUB"].append(vec_str_sub[j] + vec_str_side[j])

            # append the average results for SIN
            dict_avg_results_auc_sin["STR_MODEL"].append(vec_str_model[i])
            dict_avg_results_auc_sin["AUC"].append(np.mean(auc_python_full_sin))
            dict_avg_results_auc_sin["SUB"].append(vec_str_sub[j] + vec_str_side[j])
            
            # append the average results for SFS
            dict_avg_results_auc_sfs["STR_MODEL"].append(vec_str_model[i])
            dict_avg_results_auc_sfs["AUC"].append(np.mean(auc_python_full_sfs))
            dict_avg_results_auc_sfs["SUB"].append(vec_str_sub[j] + vec_str_side[j])
        
    # plot
    results_auc_sin = pd.DataFrame(dict_results_auc_sin)
    avg_results_auc_sin = pd.DataFrame(dict_avg_results_auc_sin)
    sns.boxplot(data=results_auc_sin, x="STR_MODEL", y="AUC", orient="v", color="gray")
    sns.pointplot(data=avg_results_auc_sin, x="STR_MODEL", y="AUC", hue="SUB", orient="v")
    ax = plt.gca()
    
    p_lda_rnn = stats.wilcoxon(
        avg_results_auc_sin[avg_results_auc_sin["STR_MODEL"] == "LDA"]["AUC"].to_numpy(),
        avg_results_auc_sin[avg_results_auc_sin["STR_MODEL"] == "RNN"]["AUC"].to_numpy(),
    ).pvalue
    
    p_lda_mlp = stats.wilcoxon(
        avg_results_auc_sin[avg_results_auc_sin["STR_MODEL"] == "LDA"]["AUC"].to_numpy(),
        avg_results_auc_sin[avg_results_auc_sin["STR_MODEL"] == "MLP"]["AUC"].to_numpy(),
    ).pvalue
    
    pairs = [("LDA", "RNN"), ("LDA", "MLP")]
    annotator = Annotator(ax, pairs, data=results_auc_sin, x="STR_MODEL", y="AUC")
    annotator.configure(test=None, test_short_name='Wilcoxon', loc='inside').set_pvalues([p_lda_rnn, p_lda_mlp]).annotate()
    
    # set the labels to right ones
    ax.set(xlabel="Model Used for Classification")
    plt.legend(frameon=False, loc="lower right")
    # plt.title("Comparison of AUC for Single Power Band")

    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # plot for SFS
    results_auc_sfs = pd.DataFrame(dict_results_auc_sfs)
    avg_results_auc_sfs = pd.DataFrame(dict_avg_results_auc_sfs)
    sns.boxplot(data=results_auc_sfs, x="STR_MODEL", y="AUC", orient="v", color="gray")
    sns.pointplot(data=avg_results_auc_sfs, x="STR_MODEL", y="AUC", hue="SUB", orient="v")
    ax = plt.gca()
    
    p_lda_rnn_sfs = stats.wilcoxon(
        avg_results_auc_sfs[avg_results_auc_sfs["STR_MODEL"] == "LDA"]["AUC"].to_numpy(),
        avg_results_auc_sfs[avg_results_auc_sfs["STR_MODEL"] == "RNN"]["AUC"].to_numpy(),
    ).pvalue
    
    p_lda_mlp_sfs = stats.wilcoxon(
        avg_results_auc_sfs[avg_results_auc_sfs["STR_MODEL"] == "LDA"]["AUC"].to_numpy(),
        avg_results_auc_sfs[avg_results_auc_sfs["STR_MODEL"] == "MLP"]["AUC"].to_numpy(),
    ).pvalue
    
    pairs = [("LDA", "RNN"), ("LDA", "MLP")]
    annotator = Annotator(ax, pairs, data=results_auc_sfs, x="STR_MODEL", y="AUC")
    annotator.configure(test=None, test_short_name='Wilcoxon', loc='outside').set_pvalues([p_lda_rnn_sfs, p_lda_mlp_sfs]).annotate()
    
    # set the labels to right ones
    ax.set(xlabel="Model Used for Classification")
    plt.legend(frameon=False, loc="lower right")
    # plt.title("Comparison of AUC for Single Power Band")

    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    t1 = 1

def plot_lda_comp_bands(vec_str_sub, vec_str_side):
    
    # set font size for everything
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # create the output variables
    vec_auc_python_sfs_full = []

    for i in range(len(vec_str_sub)):
        # obtain current subject and side
        str_subject = vec_str_sub[i]
        str_side = vec_str_side[i]

        # hard code all paths
        p_output = pathlib.Path(
            "/home/jyao/Downloads/biomarker_id/model_id/{}".format(str_subject)
        )
        f_output_python = f"{str_subject}_{str_side}_med_avg_auc_LDA.pkl"

        # load the current results
        output = pickle.load(open(str(p_output / f_output_python), "rb"))
        (
            acc_python_full,
            auc_python_full,
            sinPB,
            acc_python_full_sfs,
            auc_python_full_sfs,
            sfsPB,
        ) = load_model_id_results(output)

        # append to the output
        vec_auc_python_sfs_full.append(auc_python_full_sfs)

    # form the dataframe
    n_pb = 4
    n_rep = 5

    # append the python results
    dict_results_auc = {"NUM_PB": [], "AUC": [], "SUB": []}
    dict_avg_results_auc = {"NUM_PB": [], "AUC": [], "SUB": []}
    for i in range(n_pb):
        for j in range(len(vec_str_sub)):
            auc_python_full_sfs = vec_auc_python_sfs_full[j]

            # append the python results
            for k in range(n_rep):
                dict_results_auc["NUM_PB"].append(i + 1)
                dict_results_auc["AUC"].append(auc_python_full_sfs[k, i])
                dict_results_auc["SUB"].append(vec_str_sub[j] + vec_str_side[j])

            # append the average results
            dict_avg_results_auc["NUM_PB"].append(i + 1)
            dict_avg_results_auc["AUC"].append(np.mean(auc_python_full_sfs[:, i]))
            dict_avg_results_auc["SUB"].append(vec_str_sub[j] + vec_str_side[j])

    # plot
    results_auc = pd.DataFrame(dict_results_auc)
    avg_results_auc = pd.DataFrame(dict_avg_results_auc)
    sns.boxplot(data=results_auc, x="NUM_PB", y="AUC", orient="v", color="gray")
    sns.pointplot(data=dict_avg_results_auc, x="NUM_PB", y="AUC", hue="SUB", orient="v")
    ax = plt.gca()

    # obtain the stats
    # p_one_two = stats.wilcoxon(
    #     avg_results_auc[avg_results_auc["NUM_PB"] == 1]["AUC"].to_numpy(),
    #     avg_results_auc[avg_results_auc["NUM_PB"] == 2]["AUC"].to_numpy(),
    # ).pvalue
    
    # p_one_three = stats.wilcoxon(
    #     avg_results_auc[avg_results_auc["NUM_PB"] == 1]["AUC"].to_numpy(),
    #     avg_results_auc[avg_results_auc["NUM_PB"] == 3]["AUC"].to_numpy(),
    # ).pvalue
    
    p_one_four = stats.wilcoxon(
        avg_results_auc[avg_results_auc["NUM_PB"] == 1]["AUC"].to_numpy(),
        avg_results_auc[avg_results_auc["NUM_PB"] == 4]["AUC"].to_numpy(),
    ).pvalue

    pairs = [(1, 4)]
    annotator = Annotator(ax, pairs, data=results_auc, x="NUM_PB", y="AUC")
    annotator.configure(test=None, test_short_name='Wilcoxon', loc='outside').set_pvalues([p_one_four]).annotate()
    
    # set the labels to right ones
    ax.set(xlabel="Number of Power Bands Included")
    plt.legend(frameon=False, loc="lower right")

    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    t1 = 1


def plot_lda_results(str_model_id="model_id", str_subject="RCS02", str_side="R"):
    if str_model_id == "model_id_woart":
        # hard code all paths
        p_output = pathlib.Path(
            "/home/jyao/Downloads/biomarker_id/dynamics_woart/{}".format(str_subject)
        )
        f_output_python = f"{str_subject}_{str_side}_med_avg_auc_LDA_dynamics_woart.pkl"

        output_full = pickle.load(open(str(p_output / f_output_python), "rb"))

        output = dict()
        output["sinPB"] = output_full["sinPB"]["n_dynamics_1"]
        output["sfsPB"] = output_full["sfsPB"]["n_dynamics_1"]
        (
            acc_python_full,
            auc_python_full,
            sinPB,
            acc_python_full_sfs,
            auc_python_full_sfs,
            sfsPB,
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
            sinPB,
            acc_python_full_sfs,
            auc_python_full_sfs,
            sfsPB,
        ) = load_model_id_results(output)

    p_figure_output = pathlib.Path(
        "/home/jyao/Downloads/biomarker_id/figures/{}/{}".format(
            str_model_id, str_subject
        )
    )
    p_figure_output.mkdir(parents=True, exist_ok=True)

    # also load the SVM and QDA results
    f_output_python_QDA = f"{str_subject}_{str_side}_med_avg_auc_QDA.pkl"
    f_output_python_SVM = f"{str_subject}_{str_side}_med_avg_auc_SVM.pkl"
    f_output_python_RF = f"{str_subject}_{str_side}_med_avg_auc_RF.pkl"
    f_output_python_MLP = f"{str_subject}_{str_side}_med_avg_auc_MLP.pkl"
    f_output_python_RNN = f"{str_subject}_{str_side}_med_avg_auc_RNN.pkl"

    # form the output vectors
    vec_acc_python_full = [acc_python_full]
    vec_auc_python_full = [auc_python_full]
    vec_sinPB = [sinPB]

    vec_acc_python_full_sfs = [acc_python_full_sfs]
    vec_auc_python_full_sfs = [auc_python_full_sfs]
    vec_sfsPB = [sfsPB]
    # vec_str_model = ["LDA", "QDA", "SVM", "RF", "MLP", "RNN"]
    # vec_colormap = ["r", "k", "g", "m", "b", "c"]

    vec_str_model = ["LDA", "QDA", "SVM", "MLP", "RNN"]
    vec_colormap = ["r", "k", "g", "b", "c"]

    vec_f_output_python = [
        f_output_python_QDA,
        f_output_python_SVM,
        # f_output_python_RF,
        f_output_python_MLP,
        f_output_python_RNN,
    ]
    for i in range(len(vec_f_output_python)):
        # load and unpack output struct
        output_curr = pickle.load(open(str(p_output / vec_f_output_python[i]), "rb"))
        (
            acc_python_full_curr,
            auc_python_full_curr,
            sinPB_curr,
            acc_python_full_sfs_curr,
            auc_python_full_sfs_curr,
            sfsPB_curr,
        ) = load_model_id_results(output_curr)

        # append to the output vectors
        vec_acc_python_full.append(acc_python_full_curr)
        vec_auc_python_full.append(auc_python_full_curr)
        vec_sinPB.append(sinPB_curr)

        vec_acc_python_full_sfs.append(acc_python_full_sfs_curr)
        vec_auc_python_full_sfs.append(auc_python_full_sfs_curr)
        vec_sfsPB.append(sfsPB_curr)

    # now plot the model comp figures
    # start with AUC comparison for sinPB
    str_metric = "AUC"
    f_figure_output_sinPB = (
        f"{str_subject}_{str_side}_med_level_{str_metric.lower()}_comp_sinPB.png"
    )
    plot_model_comp(
        vec_auc_python_full,
        vec_str_model,
        vec_colormap,
        str_subject,
        str_side,
        str_metric=str_metric,
        p_figure_output=p_figure_output,
        f_figure_output=f_figure_output_sinPB,
    )

    # next plot the AUC for sfsPB
    f_figure_output_sfsPB = (
        f"{str_subject}_{str_side}_med_level_{str_metric.lower()}_comp_sfsPB.png"
    )
    plot_model_comp(
        vec_auc_python_full_sfs,
        vec_str_model,
        vec_colormap,
        str_subject,
        str_side,
        str_metric=str_metric,
        p_figure_output=p_figure_output,
        f_figure_output=f_figure_output_sfsPB,
    )

    t1 = 1


if __name__ == "__main__":
    # plot_lda_matlab_results(["RCS02", "RCS11", "RCS12", "RCS18"], ["R", "L", "L", "L"])
    # plot_lda_matlab_results(["RCS17"], ["L"])
    # plot_lda_matlab_results(["RCS17"], ["R"])
    # plot_lda_results(str_subject="RCS02", str_side="R")
    # plot_lda_results(str_subject="RCS08", str_side="R")
    # plot_lda_results(str_subject="RCS11", str_side="L")
    # plot_lda_results(str_subject="RCS12", str_side="L")
    # plot_lda_results(str_subject="RCS18", str_side="L")

    vec_str_sub = ["RCS02", "RCS08", "RCS11", "RCS12", "RCS18"]
    vec_str_side = ["R", "R", "L", "L", "L"]
    vec_str_model = ["LDA", "QDA", "SVM", "RF", "MLP", "RNN"]
    
    vec_str_sub_v1 = ["RCS02", "RCS08", "RCS11", "RCS12", "RCS17", "RCS18"]
    vec_str_side_v1 = ["R", "R", "L", "L", "L", "L"]
    
    # plot_lda_comp_bands(vec_str_sub_v1, vec_str_side_v1)
    plot_model_comp(vec_str_sub, vec_str_side, vec_str_model)
    