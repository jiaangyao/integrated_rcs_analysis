from itertools import chain

import wandb
import numpy as np
import pandas as pd


_MAX_NUMEBER_OF_FINAL_PB = 5


def create_dyna_dataframe(output_dyna, n_rep, n_pb, n_dynamics, str_PB="sfsPB"):
    vec_n_dyna = np.ones((n_rep * n_pb,)) * n_dynamics
    vec_avg_acc = []
    vec_avg_f1 = []
    vec_avg_auc = []
    vec_num_pb = []
    for i_rep in range(n_rep):
        vec_avg_acc.append(output_dyna[str_PB][i_rep]["vec_acc"])
        vec_avg_f1.append(output_dyna[str_PB][i_rep]["vec_f1"])
        vec_avg_auc.append(output_dyna[str_PB][i_rep]["vec_auc"])
        vec_num_pb.append(np.arange(n_pb) + 1)
    vec_avg_acc = np.concatenate(vec_avg_acc, axis=0)
    vec_avg_f1 = np.concatenate(vec_avg_f1, axis=0)
    vec_avg_auc = np.concatenate(vec_avg_auc, axis=0)
    vec_num_pb = np.concatenate(vec_num_pb, axis=0)

    # create the dataframe
    dyna_df = pd.DataFrame(
        {
            "SFS_n_dyna": vec_n_dyna,
            "avg_acc": vec_avg_acc,
            "avg_f1": vec_avg_f1,
            "avg_auc": vec_avg_auc,
            "num_pb": vec_num_pb,
        }
    )

    return dyna_df


def calc_dyna_summary_stats(
    wandb_dyna_df,
    n_dynamics,
    n_pb,
):
    vec_dyna = np.arange(1, n_dynamics + 1)
    avg_acc_ys = [
        [
            np.mean(
                wandb_dyna_df[
                    (wandb_dyna_df["SFS_n_dyna"] == n_dyna_curr)
                    & (wandb_dyna_df["num_pb"] == i)
                ]["avg_acc"].to_numpy()
            )
            for n_dyna_curr in vec_dyna
        ]
        for i in range(1, n_pb + 1)
    ]
    avg_f1_ys = [
        [
            np.mean(
                wandb_dyna_df[
                    (wandb_dyna_df["SFS_n_dyna"] == n_dyna_curr)
                    & (wandb_dyna_df["num_pb"] == i)
                ]["avg_f1"].to_numpy()
            )
            for n_dyna_curr in vec_dyna
        ]
        for i in range(1, n_pb + 1)
    ]
    avg_auc_ys = [
        [
            np.mean(
                wandb_dyna_df[
                    (wandb_dyna_df["SFS_n_dyna"] == n_dyna_curr)
                    & (wandb_dyna_df["num_pb"] == i)
                ]["avg_auc"].to_numpy()
            )
            for n_dyna_curr in vec_dyna
        ]
        for i in range(1, n_pb + 1)
    ]

    return avg_acc_ys, avg_f1_ys, avg_auc_ys


def wandb_logging_dyna(
    output_dyna: dict,
    n_dynamics: int,
    bool_use_wandb: bool,
    wandb_sfsPB_dyna: wandb.Table | None,
    wandb_sinPB_dyna: wandb.Table | None,
):
    # perform logging using wandb
    if bool_use_wandb:
        n_rep = len(output_dyna["sfsPB"])
        n_pb = len(output_dyna["sfsPB"][0]["vec_auc"])

        # now process the SFS table
        if wandb_sfsPB_dyna is None:
            # create the dataframe and the wandb table
            wandb_sfsPB_dyna_df = create_dyna_dataframe(
                output_dyna,
                n_rep,
                n_pb,
                n_dynamics,
                str_PB="sfsPB",
            )
            wandb_sfsPB_dyna = wandb.Table(data=wandb_sfsPB_dyna_df)
        else:
            # obtain current dataframe and append to existing dataframe
            df_existing = wandb_sfsPB_dyna.get_dataframe()
            new_dyna_df = create_dyna_dataframe(
                output_dyna,
                n_rep,
                n_pb,
                n_dynamics,
                str_PB="sfsPB",
            )
            wandb_sfsPB_dyna_df = pd.concat([df_existing, new_dyna_df])
            wandb_sfsPB_dyna = wandb.Table(data=wandb_sfsPB_dyna_df)

        # next process the sinPB table
        if wandb_sinPB_dyna is None:
            # create the dataframe and the wandb table
            wandb_sinPB_dyna_df = create_dyna_dataframe(
                output_dyna,
                n_rep,
                n_pb,
                n_dynamics,
                str_PB="sinPB",
            )
            wandb_sinPB_dyna = wandb.Table(data=wandb_sinPB_dyna_df)
        else:
            # obtain current dataframe and append to existing dataframe
            df_existing = wandb_sinPB_dyna.get_dataframe()
            new_dyna_df = create_dyna_dataframe(
                output_dyna,
                n_rep,
                n_pb,
                n_dynamics,
                str_PB="sinPB",
            )
            wandb_sinPB_dyna_df = pd.concat([df_existing, new_dyna_df])
            wandb_sinPB_dyna = wandb.Table(data=wandb_sinPB_dyna_df)

        # now log the table SFS
        wandb.log({"SFS_DYNA/sfsPB_dyna_table": wandb_sfsPB_dyna})
        wandb_sfsPB_dyna_df = wandb_sfsPB_dyna.get_dataframe()
        vec_dyna = np.arange(1, n_dynamics + 1)
        avg_sfsPB_acc_ys, avg_sfsPB_f1_ys, avg_sfsPB_auc_ys = calc_dyna_summary_stats(
            wandb_sfsPB_dyna_df,
            n_dynamics,
            n_pb,
        )
        n_pb_keys = [f"N_PB_{i}" for i in range(1, n_pb + 1)]

        # log the plots
        wandb.log(
            {
                f"SFS_DYNA/sfsPB_avg_acc_plot": wandb.plot.line_series(
                    xs=vec_dyna,
                    ys=avg_sfsPB_acc_ys,
                    keys=n_pb_keys,
                    title="sfsPB_avg_acc",
                    xname="SFS_n_dyna",
                ),
                f"SFS_DYNA/sfsPB_avg_f1_plot": wandb.plot.line_series(
                    xs=vec_dyna,
                    ys=avg_sfsPB_f1_ys,
                    keys=n_pb_keys,
                    title="sfsPB_avg_f1",
                    xname="SFS_n_dyna",
                ),
                f"SFS_DYNA/sfsPB_avg_auc_plot": wandb.plot.line_series(
                    xs=vec_dyna,
                    ys=avg_sfsPB_auc_ys,
                    keys=n_pb_keys,
                    title="sfsPB_avg_auc",
                    xname="SFS_n_dyna",
                ),
            }
        )

        # next log the table for sinPB
        wandb.log({"SFS_DYNA/sinPB_dyna_table": wandb_sinPB_dyna})
        wandb_sinPB_dyna_df = wandb_sinPB_dyna.get_dataframe()
        avg_sinPB_acc_ys, avg_sinPB_f1_ys, avg_sinPB_auc_ys = calc_dyna_summary_stats(
            wandb_sinPB_dyna_df,
            n_dynamics,
            n_pb,
        )

        # log the plots
        wandb.log(
            {
                f"SFS_DYNA/sinPB_avg_acc_plot": wandb.plot.line_series(
                    xs=vec_dyna,
                    ys=avg_sinPB_acc_ys,
                    keys=n_pb_keys,
                    title="sinPB_avg_acc",
                    xname="SFS_n_dyna",
                ),
                f"SFS_DYNA/sinPB_avg_f1_plot": wandb.plot.line_series(
                    xs=vec_dyna,
                    ys=avg_sinPB_f1_ys,
                    keys=n_pb_keys,
                    title="sinPB_avg_f1",
                    xname="SFS_n_dyna",
                ),
                f"SFS_DYNA/sinPB_avg_auc_plot": wandb.plot.line_series(
                    xs=vec_dyna,
                    ys=avg_sinPB_auc_ys,
                    keys=n_pb_keys,
                    title="sinPB_avg_auc",
                    xname="SFS_n_dyna",
                ),
            }
        )

        # log the simple outputs
        log_dict = {
            "SFS_DYNA_TOP/n_dyna": n_dynamics,
            "SFS_DYNA_TOP/best_avg_sinPB_auc": avg_sinPB_auc_ys[0][-1],
            "SFS_DYNA_TOP/best_avg_sfsPB_auc": avg_sfsPB_auc_ys[-1][-1]
        }
        wandb.log(log_dict)


    return wandb_sfsPB_dyna, wandb_sinPB_dyna


def wandb_logging_sfs_outer(
    output_fin: dict,
    output_init: dict,
    idx_rep: int,
    vec_wandb_sfsPB: list[wandb.Table] | None,
    vec_wandb_sinPB: list[wandb.Table] | None,
    bool_use_wandb: bool,
    n_fin_pb: int,
    n_dynamics: int,
):
    # use wandb for logging
    if bool_use_wandb:
        # compute the number of PBs possible
        n_pb = min(n_fin_pb, _MAX_NUMEBER_OF_FINAL_PB)

        # form the rows in the output table
        # start with the SFS output
        vec_sfsPB_values = [
            [
                n_dynamics,
                idx_rep + 1,
                output_fin["sfsPB"][idx_pb][0],
                output_fin["sfsPB"][idx_pb][1],
                output_fin["sfsPB"][idx_pb][2],
                output_fin["vec_acc"][idx_pb],
                output_fin["vec_f1"][idx_pb],
                output_fin["vec_auc"][idx_pb],
            ]
            for idx_pb in range(0, n_pb)
        ]

        # create the column headers
        vec_sfsPB_columns = [
            [
                "SFS_n_dyna",
                "SFS_rep",
                f"PB{idx_pb}_ch",
                f"PB{idx_pb}_freq_low",
                f"PB{idx_pb}_freq_high",
                f"PB{idx_pb}_acc",
                f"PB{idx_pb}_f1",
                f"PB{idx_pb}_auc",
            ]
            for idx_pb in range(1, n_pb + 1)
        ]

        # next form the sinPB output
        vec_sinPB_values = [
            [
                n_dynamics,
                idx_rep + 1,
                output_init["sinPB"][idx_pb][0],
                output_init["sinPB"][idx_pb][1],
                output_init["sinPB"][idx_pb][2],
                output_init["vec_acc"][idx_pb],
                output_init["vec_f1"][idx_pb],
                output_init["vec_auc"][idx_pb],
            ]
            for idx_pb in range(0, n_pb)
        ]

        # create the column headers
        vec_sinPB_columns = [
            [
                "SFS_n_dyna",
                "SFS_rep",
                f"PB{idx_pb}_ch",
                f"PB{idx_pb}_freq_low",
                f"PB{idx_pb}_freq_high",
                f"PB{idx_pb}_acc",
                f"PB{idx_pb}_f1",
                f"PB{idx_pb}_auc",
            ]
            for idx_pb in range(1, n_pb + 1)
        ]

        # next proceed with updating sfsPB table
        # if table doesn't exist then create it
        if vec_wandb_sfsPB is None:
            # create the respective tables
            vec_wandb_sfsPB = []
            for i in range(n_pb):
                # for each power band create the dataframe and the wandB table
                wandb_table_curr = wandb.Table(
                    data=pd.DataFrame(
                        data={
                            vec_sfsPB_columns[i][j]: vec_sfsPB_values[i][j]
                            for j in range(len(vec_sfsPB_columns[i]))
                        },
                        index=[idx_rep],
                    )
                )
                vec_wandb_sfsPB.append(wandb_table_curr)

        # otherwise append to existing table
        else:
            for i in range(n_pb):
                # concatenate the dataframe
                df_existing = vec_wandb_sfsPB[i].get_dataframe()
                df_curr = pd.DataFrame(
                    data={
                        vec_sfsPB_columns[i][j]: vec_sfsPB_values[i][j]
                        for j in range(len(vec_sfsPB_columns[i]))
                    },
                    index=[idx_rep],
                )
                df_full = pd.concat([df_existing, df_curr])

                # mutate the table
                vec_wandb_sfsPB[i] = wandb.Table(data=df_full)

        # next proceed with updating sinPB table
        if vec_wandb_sinPB is None:
            # create the respective tables
            vec_wandb_sinPB = []
            for i in range(n_pb):
                # for each power band create the dataframe and the wandB table
                wandb_table_curr = wandb.Table(
                    data=pd.DataFrame(
                        data={
                            vec_sinPB_columns[i][j]: vec_sinPB_values[i][j]
                            for j in range(len(vec_sinPB_columns[i]))
                        },
                        index=[idx_rep],
                    )
                )
                vec_wandb_sinPB.append(wandb_table_curr)

        # otherwise append to existing table
        else:
            for i in range(n_pb):
                vec_wandb_sinPB[i].add_data(*vec_sinPB_values[i])

                # concatenate the dataframe
                df_existing = vec_wandb_sinPB[i].get_dataframe()
                df_curr = pd.DataFrame(
                    data={
                        vec_sinPB_columns[i][j]: vec_sinPB_values[i][j]
                        for j in range(len(vec_sinPB_columns[i]))
                    },
                    index=[idx_rep],
                )
                df_full = pd.concat([df_existing, df_curr])

                # mutate the table
                vec_wandb_sinPB[i] = wandb.Table(data=df_full)

        # log sfsPB tables to wandb
        log_dict_sfsPB_table = dict()
        for i in range(n_pb):
            log_dict_sfsPB_table[f"sfsPB/sfsPB_PB{i + 1}"] = vec_wandb_sfsPB[i]
        wandb.log(log_dict_sfsPB_table)

        # log sinPB tables to wandb
        log_dict_sinPB_table = dict()
        for i in range(n_pb):
            log_dict_sinPB_table[f"sinPB/sinPB_PB{i + 1}"] = vec_wandb_sinPB[i]
        wandb.log(log_dict_sinPB_table)

        # first create the simple plots
        log_dict = {
            "SFS_ITER/rep": idx_rep + 1,
            "SFS_ITER/best_sinPB_auc": output_init["vec_auc"][0],
            "SFS_ITER/best_sfsPB_auc": output_fin["vec_auc"][-1],
        }
        wandb.log(log_dict)

    return vec_wandb_sfsPB, vec_wandb_sinPB


def wandb_logging_sfs_inner(
    vec_output: list[dict],
    wandb_table: wandb.Table | None,
    n_iter: int,
    str_sfs: str,
    bool_use_wandb: bool,
    bool_use_lightweight_wandb: bool,
    bool_use_ray: bool,
):
    # use wandb for logging
    if bool_use_wandb:
        # if use the full wandb features
        if not bool_use_lightweight_wandb:
            # now check if input table exists already and if not exist then create
            if wandb_table is None:
                # if none then create a new table
                # obtain the various variables to log
                center_freq = np.arange(1, len(vec_output) + 1)
                vec_avg_acc = []
                vec_avg_f1 = []
                vec_avg_auc = []
                vec_n_iter = np.ones_like(center_freq) * n_iter

                # loop through the output
                for idx_feature in range(len(vec_output)):
                    vec_avg_acc.append(vec_output[idx_feature]["avg_acc"])
                    vec_avg_f1.append(vec_output[idx_feature]["avg_f1"])
                    vec_avg_auc.append(vec_output[idx_feature]["avg_auc"])
                vec_avg_acc = np.stack(vec_avg_acc, axis=0)
                vec_avg_f1 = np.stack(vec_avg_f1, axis=0)
                vec_avg_auc = np.stack(vec_avg_auc, axis=0)

                wandb_table = wandb.Table(
                    data=pd.DataFrame(
                        {
                            f"{str_sfs}_center_freq": center_freq,
                            f"{str_sfs}_avg_acc": vec_avg_acc,
                            f"{str_sfs}_avg_f1": vec_avg_f1,
                            f"{str_sfs}_avg_auc": vec_avg_auc,
                            f"{str_sfs}_n_iter": vec_n_iter,
                        }
                    )
                )

            elif bool_use_ray and wandb_table is not None:
                # otherwise parse the output structure from ray and append to existing table
                for idx_feature in range(len(vec_output)):
                    wandb_table.add_data(
                        idx_feature + 1,
                        vec_output[idx_feature]["avg_acc"],
                        vec_output[idx_feature]["avg_f1"],
                        vec_output[idx_feature]["avg_auc"],
                        n_iter,
                    )

            # log the output
            wandb.log({f"{str_sfs}/{str_sfs}_table": wandb_table})
            wandb_table_df = wandb_table.get_dataframe()
            center_freq_xs = wandb_table_df[f"{str_sfs}_center_freq"].to_numpy()
            vec_unique_iter = pd.unique(wandb_table_df[f"{str_sfs}_n_iter"])
            avg_acc_ys = [
                wandb_table_df[wandb_table_df[f"{str_sfs}_n_iter"] == unique_iter][
                    f"{str_sfs}_avg_acc"
                ].to_numpy()
                for unique_iter in vec_unique_iter
            ]
            avg_f1_ys = [
                wandb_table_df[wandb_table_df[f"{str_sfs}_n_iter"] == unique_iter][
                    f"{str_sfs}_avg_f1"
                ].to_numpy()
                for unique_iter in vec_unique_iter
            ]
            avg_auc_ys = [
                wandb_table_df[wandb_table_df[f"{str_sfs}_n_iter"] == unique_iter][
                    f"{str_sfs}_avg_auc"
                ].to_numpy()
                for unique_iter in vec_unique_iter
            ]
            n_iter_keys = [f"iter_{n_iter}" for n_iter in vec_unique_iter]

            # log the plots
            wandb.log(
                {
                    f"{str_sfs}/avg_acc_plot": wandb.plot.line_series(
                        xs=center_freq_xs,
                        ys=avg_acc_ys,
                        keys=n_iter_keys,
                        title=f"{str_sfs}_avg_acc",
                        xname=f"{str_sfs}_center_freq",
                    ),
                    f"{str_sfs}/avg_f1_plot": wandb.plot.line_series(
                        xs=center_freq_xs,
                        ys=avg_f1_ys,
                        keys=n_iter_keys,
                        title=f"{str_sfs}_avg_f1",
                        xname=f"{str_sfs}_center_freq",
                    ),
                    f"{str_sfs}/avg_auc_plot": wandb.plot.line_series(
                        xs=center_freq_xs,
                        ys=avg_auc_ys,
                        keys=n_iter_keys,
                        title=f"{str_sfs}_avg_auc",
                        xname=f"{str_sfs}_center_freq",
                    ),
                }
            )
            
    return wandb_table
