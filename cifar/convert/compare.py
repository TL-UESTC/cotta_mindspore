from compare_torch import WideResNet as torch_res
from compare_minds import WideResNet as minds_res

from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore import Tensor
import os
import torch


def show_params(ckpt_file_path, frame="torch", key=True, value=False):
    """ Show contents of a checkpoint file.
    """
    if frame == "torch":
        params = torch.load(ckpt_file_path, map_location=torch.device('cpu'))['state_dict']
    elif frame == "mindspore":
        params = load_checkpoint(ckpt_file_path)
    else:
        raise ValueError("Attribute `params` must be in [`torch`, `mindspore`]! ")
    if key and value:
        for k, v in params.items():
            # if "running_mean" in k:
                print(k, v)
    elif key and not value:
        for k in params.keys():
            print(k)
    elif value and not key:
        for v in params.values():
            print(v)


def compare_model_names(torch_params_dict, mindspore_params_dict):
    """ Compare the params' names between torch and mindspore nets.
    """
    t_params_dict = torch_params_dict.copy()
    m_params_dict = mindspore_params_dict.copy()
    for key in torch_params_dict.keys():
        if "num_batches_tracked" in key:
            t_params_dict.pop(key)

    for t, m in zip(t_params_dict.keys(), m_params_dict.keys()):
        print(t)
        print(m)
        print("=============================")


def from_torch_to_mindspore(net, ckpt_file_path, save_path):
    """ Transform a torch checkpoint file into mindspore checkpoint.
        Modify the param's name first, then change tensor type.
    """
    if not os.path.isfile(ckpt_file_path):
        raise FileExistsError("The file `{}` is not exist! ".format(ckpt_file_path))
    if ".ckpt" not in save_path:
        raise ValueError("Attribute `save_path` should be a checkpoint file with the end of `.ckpt`!")

    params = torch.load(ckpt_file_path, map_location=torch.device('cpu'))

    torch_params = list(params.items())
    num_params = len(torch_params)
    params_list = []
    for i in range(num_params):
        key, value = torch_params[i]
        if "weight" in key and i + 2 < num_params:
            if "running_mean" in torch_params[i + 2][0]:
                key = key.replace("weight", "gamma")
        if "bias" in key and i + 1 < num_params:
            if "running_mean" in torch_params[i + 1][0]:
                key = key.replace("bias", "beta")
        if "running_var" in key:
            key = key.replace("running_var", "moving_variance")
        if "running_mean" in key:
            key = key.replace("running_mean", "moving_mean")
        if "num_batches_tracked" in key:
            continue
        # if "incre" in key:      # `incre` is a name of params in hrnet for classification.
        #     break
        params_list.append({"name": key, "data": Tensor(value.numpy())})
    save_checkpoint(params_list, save_path)


if __name__ == "__main__":
    net = minds_res()
    m_params = net.parameters_dict()
    t_params = torch.load("../ckpt/cifar10/corruptions/Standard.pt", map_location=torch.device('cpu'))['state_dict']
    show_params("../ckpt/cifar10/corruptions/Standard.pt")
    compare_model_names(t_params, m_params)
