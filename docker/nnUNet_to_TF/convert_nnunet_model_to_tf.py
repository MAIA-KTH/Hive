#!/usr/bin/env python

import datetime
import pickle
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import numpy as np
import onnx
import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from onnx_tf.backend import prepare
from torch import nn

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to convert a nnUNet trained model into a TensorFlow bundle.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --plans-pickle-file /PATH/TO/PLANS_PICKLE_FILE --model-checkpoint /PATH/TO/CHECKPOINT_FILE --output-folder /PATH/TO/TF_BUNDLE
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--plans-pickle-file",
        type=str,
        required=True,
        help="File path to the nnUnet plans.pkl file, used to load U-Net features.",
    )

    pars.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="File path to the trained model nnUNet checkpoint. ",
    )

    pars.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output path where to save the TensorFlow bundle.",
    )

    return pars


def network(plans, checkpoint_file):

    stage_plans = plans["plans_per_stage"][1]
    net_numpool = len(stage_plans["pool_op_kernel_sizes"])
    patch_size = np.array(stage_plans["patch_size"]).astype(int)

    if len(patch_size) == 3:
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
    else:
        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d

    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    dropout_op_kwargs = {"p": 0, "inplace": True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
    num_input_channels = plans["num_modalities"]
    base_num_features = plans["base_num_features"]
    num_classes = plans["num_classes"] + 1
    conv_per_stage = plans["conv_per_stage"]
    net_conv_kernel_sizes = stage_plans["conv_kernel_sizes"]
    net_num_pool_op_kernel_sizes = stage_plans["pool_op_kernel_sizes"]

    network = Generic_UNet(
        num_input_channels,
        base_num_features,
        num_classes,
        net_numpool,
        conv_per_stage,
        2,
        conv_op,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        net_nonlin,
        net_nonlin_kwargs,
        False,
        False,
        lambda x: x,
        InitWeights_He(1e-2),
        net_num_pool_op_kernel_sizes,
        net_conv_kernel_sizes,
        False,
        True,
        True,
    )
    state_dict = torch.load(checkpoint_file, map_location="cuda")["state_dict"]
    network.load_state_dict(state_dict)
    # network.cuda()
    network.eval()

    return network


def main():
    parser = get_arg_parser()
    arguments = vars(parser.parse_args())

    with open(arguments["plans_pickle_file"], "rb") as file:
        plans = pickle.load(file)

    checkpoint_file = arguments["model_checkpoint"]

    net = network(plans, checkpoint_file)
    net = torch.nn.Sequential(net, nn.Softmax(dim=1))

    batch_size = 1
    x = torch.randn(batch_size, 1, 128, 128, 128, requires_grad=True)

    # Export the model
    torch.onnx.export(
        net,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "nnunet.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
    )

    # Load ONNX model and convert to TensorFlow format
    model_onnx = onnx.load("nnunet.onnx")

    tf_rep = prepare(model_onnx)

    tf_rep.export_graph(arguments["output_folder"])


if __name__ == "__main__":
    main()
