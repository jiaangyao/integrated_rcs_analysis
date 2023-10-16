import re
import subprocess

import torch
import torch.nn as nn
import numpy.typing as npt


# TODO: transfer constants to constants directory
# define global variables
_STR_TO_ACTIVATION = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


device = None


def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]  # type: ignore
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldn't parse " + line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map() -> dict:
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(
        r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB"
    )
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory() -> int:
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def init_gpu(
        use_gpu: bool = True,
        gpu_id: int = 0,
        bool_use_best_gpu: bool = True,
        bool_limit_gpu_mem: bool = False,
        gpu_memory_fraction: float = 0.5,
        verbose: bool = False,
):
    global device

    # pick best gpu if flag is set
    if bool_use_best_gpu:
        gpu_id = pick_gpu_lowest_memory()

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))

        force_cudnn_initialization()
        if verbose:
            print("Using GPU id {}".format(gpu_id))

        # optionally limit gpu memory
        if bool_limit_gpu_mem:
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=device)
            if verbose:
                print("GPU memory limited to {}%".format(gpu_memory_fraction * 101))
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU not detected. Defaulting to CPU.")


def set_device(
        gpu_id: int,
):
    torch.cuda.set_device(gpu_id)


def from_numpy(
        *args,
        **kwargs,
) -> torch.Tensor:
    return torch.from_numpy(*args, **kwargs).float().to(device)


def from_numpy_same_device(
        *args,
        **kwargs,
) -> torch.Tensor:
    return torch.from_numpy(*args, **kwargs).float()


def to_numpy(
        tensor: torch.Tensor,
) -> npt.NDArray:
    return tensor.to("cpu").detach().numpy()


def get_act_func() -> dict:
    return _STR_TO_ACTIVATION


def force_cudnn_initialization():
    s = 32
    dev = torch.device("cuda")
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
    )
