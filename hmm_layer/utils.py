import os

import numpy as np
import torch


def show_value(value, name="value"):
    # if isinstance(value, torch.Tensor):
    #     value_arr = value.detach().cpu().numpy()
    # else:
    #     value_arr = value
    # npy_path = "../outputs"
    # file_name = os.path.join(npy_path, f"{name}.npy")
    # if os.path.exists(file_name):
    #     ref_value = np.load(file_name)
    #     ref_comp = ref_value - value_arr
    #     print(f"{name}_comp: {ref_comp.mean():.4f}, {ref_comp.std():.4f}, {ref_comp.min():.4f}, {ref_comp.max():.4f}")
    #
    # print(f"{name}: {value_arr.mean():.4f}, {value_arr.std():.4f}, {value_arr.min():.4f}, {value_arr.max():.4f}")
    pass
