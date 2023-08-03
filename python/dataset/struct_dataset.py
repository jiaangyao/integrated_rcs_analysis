import copy
import typing as tp

import numpy as np
import numpy.typing as npt


def create_output_struct(
        output_fin: dict,
        output_curr: dict,
):
    # TODO: change into a OOP structure
    for key in output_curr.keys():
        # sanity check
        if not isinstance(key, str):
            raise ValueError("Key must be a string")

        # now create empty fields in output array
        str_key_fin = "vec_{}".format(key)
        if str_key_fin not in output_fin.keys():
            output_fin[str_key_fin] = []

    return output_fin


def append_output_struct(
        output_fin: dict,
        output_curr: dict,
        str_except: list[str] | None = None,
):
    for key in output_curr.keys():
        # sanity check
        if not isinstance(key, str):
            raise ValueError("Key must be a string")

        # skip if key is in except list
        if str_except is not None and key in str_except:
            continue

        # now create empty fields in output array
        str_key_fin = "vec_{}".format(key)
        if str_key_fin not in output_fin.keys():
            raise ValueError("Field not found in struct")

        # now append the first element
        output_fin[str_key_fin].append(output_curr[key])

    return output_fin


def arrayize_output_struct(
        output_fin,
):
    for key in output_fin.keys():
        if isinstance(output_fin[key], list):
            output_fin[key] = np.stack(output_fin[key], axis=0)

    return output_fin


def append_pred_output_struct(
        output_fin,
):
    # create gigantic list in final structure and append everything to it
    output_fin['pred'] = dict()
    for dict_curr in output_fin['vec_pred']:
        for key, val in dict_curr.items():
            if key not in output_fin['pred'].keys():
                output_fin['pred'][key] = []
            output_fin['pred'][key].append(val)

    n_folds = len(output_fin['vec_pred'])
    for key, val in output_fin['pred'].items():
        # sanity check
        assert len(val) == n_folds, "Number of folds mismatch"

        # next turn into numpy array
        val = np.concatenate(val, axis=0)
        output_fin['pred'][key] = val

    return output_fin


def comp_summary_output_struct(
        output_fin,
):
    # TODO: tidy this up and make less hard-coded
    output_fin["avg_acc"] = np.mean(output_fin["vec_acc"])
    output_fin["avg_conf_mat"] = np.sum(output_fin["vec_conf_mat"], axis=0) / np.sum(
        output_fin["vec_conf_mat"]
    )
    output_fin["avg_f1"] = float(np.mean(output_fin["vec_f1"]))
    output_fin["avg_auc"] = float(np.mean(output_fin["vec_auc"]))
    output_fin["std_f1"] = float(np.std(output_fin["vec_f1"]))
    output_fin["std_auc"] = float(np.std(output_fin["vec_auc"]))

    return output_fin


def combine_struct_by_field_list(
        vec_struct: list | tuple,
        str_field: str,
) -> list:
    output_array = []
    for i in range(len(vec_struct)):
        struct_curr = vec_struct[i]
        assert str_field in struct_curr.keys(), "Field not found in struct"
        output_array.append(struct_curr[str_field])

    return output_array


def combine_struct_by_field(
        vec_struct: list | tuple,
        str_field: str,
) -> npt.NDArray[np.float_]:
    return np.stack(combine_struct_by_field_list(vec_struct, str_field), axis=0)
