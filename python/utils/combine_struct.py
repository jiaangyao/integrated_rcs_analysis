import typing as tp

import numpy as np
import numpy.typing as npt


def combine_struct_by_field_list(vec_struct: tp.Union[list, tuple], str_field: str) -> list:

    output_array = []
    for i in range(len(vec_struct)):
        struct_curr = vec_struct[i]
        assert str_field in struct_curr.keys(), 'Field not found in struct'
        output_array.append(struct_curr[str_field])

    return output_array

def combine_struct_by_field(vec_struct: tp.Union[list, tuple], str_field: str) -> npt.NDArray[np.float64]:
    return np.stack(combine_struct_by_field_list(vec_struct, str_field), axis=0)
