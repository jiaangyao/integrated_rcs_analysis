import numpy as np


from dataset.struct_dataset import combine_struct_by_field


# TODO: tease some of these out into a separate function


def sfs_forward_pass(
    output_fin,
    output_init,
    vec_output_pb_grow,
    vec_pb_full,
    vec_idx_peak_pb,
    idx_all,
    str_metric,
    n_fin_pb,
    n_candidate_pb,
):
    # loop through the candidate combinations for each peak and pick the best one
    vec_output_sfs = []
    for i in range(len(np.unique(vec_idx_peak_pb))):
        # loop through the peaks and for each peak create output structure
        idx_pb_curr = np.where(vec_idx_peak_pb == i)[0]
        vec_output_sfs_pb_curr = [vec_output_pb_grow[j] for j in idx_pb_curr]
        vec_pb_curr = [vec_pb_full[j] for j in idx_pb_curr]

        # now obtain the metric and sort
        vec_metric_pb_curr = combine_struct_by_field(vec_output_sfs_pb_curr, str_metric)
        idx_max_metric = np.argsort(vec_metric_pb_curr)[::-1][:n_candidate_pb]

        # now form the best power band from current peak
        # _, output_sfs['pb_best'] = combine_hist(vec_metric_pb_curr)
        output_sfs = vec_output_sfs_pb_curr[idx_max_metric[0]]
        output_sfs["pb_best"] = vec_pb_curr[idx_max_metric[0]]

        vec_output_sfs.append(output_sfs)

    # organize the different power bands by metric
    vec_metric_sfs = combine_struct_by_field(vec_output_sfs, str_metric)
    idx_sort = np.argsort(vec_metric_sfs)[::-1][:n_fin_pb]
    vec_output_sfs = [vec_output_sfs[i] for i in idx_sort]

    # initialize the initial output structure for the single power bands
    n_pb_init = min([n_fin_pb, len(vec_output_sfs)])
    if len(output_fin["vec_pb_ord"]) == 0:
        output_init["vec_pb_ord"] = [
            idx_all[vec_output_sfs[i]["pb_best"]] for i in range(n_pb_init)
        ]
        output_init["vec_acc"] = [
            vec_output_sfs[i]["avg_acc"] for i in range(n_pb_init)
        ]
        output_init["vec_f1"] = [vec_output_sfs[i]["avg_f1"] for i in range(n_pb_init)]
        output_init["vec_conf_mat"] = [
            vec_output_sfs[i]["avg_conf_mat"] for i in range(n_pb_init)
        ]
        output_init["vec_auc"] = [
            vec_output_sfs[i]["avg_auc"] for i in range(n_pb_init)
        ]

    # for the SFS output, pick the top power band and proceed
    output_fin["vec_pb_ord"].append(idx_all[vec_output_sfs[0]["pb_best"]])
    output_fin["vec_acc"].append(vec_output_sfs[0]["avg_acc"])
    output_fin["vec_f1"].append(vec_output_sfs[0]["avg_f1"])
    output_fin["vec_conf_mat"].append(vec_output_sfs[0]["avg_conf_mat"])
    output_fin["vec_auc"].append(vec_output_sfs[0]["avg_auc"])

    return output_fin, output_init, vec_output_sfs
