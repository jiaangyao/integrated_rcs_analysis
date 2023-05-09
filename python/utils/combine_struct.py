import numpy as np


def combine_struct_by_field(vec_struct, str_field):

    output_array = []
    for i in range(len(vec_struct)):
        struct_curr = vec_struct[i]
        assert str_field in struct_curr.keys(), 'Field not found in struct'
        output_array.append(struct_curr[str_field])

    return np.stack(output_array, axis=0)