import sys
from argparse import ArgumentParser

from ext.lab2im.utils import infer


def parse_cmdline_arguments():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(help="sub-command help")

    # create the parser for the "resume-train" command
    parser_resume = subparsers.add_parser(
        "resume-train", help="Use this sub-command for resuming training"
    )
    parser_resume.add_argument(
        "logdir",
        type=str,
        help="""Folder containing previous checkpoints""",
    )
    parser_resume.add_argument("--debug", action="store_true", dest="debug")

    # create the parser for the "train" command
    parser_train = subparsers.add_parser(
        "train", help="Use this sub-command for training"
    )

    parser_train.add_argument("--debug", action="store_true", dest="debug")
    parser_train.add_argument("--model_idx", type=int, dest="model_idx", default=1)
    parser_train.add_argument("--epochs", type=int, dest="epochs", default=500)
    parser_train.add_argument("--batch_size", type=int, dest="batch_size", default=32)
    parser_train.add_argument("--time_steps", type=int, dest="time_steps", default=500)
    parser_train.add_argument(
        "--beta_schedule", type=str, dest="beta_schedule", default="linear"
    )
    parser_train.add_argument("--logdir", type=str, dest="logdir", default="test")
    parser_train.add_argument("--jei_flag", type=int, dest="jei_flag", default=0)
    parser_train.add_argument(
        "--group_labels", type=int, dest="group_labels", default=0
    )
    parser_train.add_argument("--lr", type=float, dest="lr", default=5e-5)
    parser_train.add_argument(
        "--im_size", nargs="?", type=infer, dest="im_size", default=None
    )
    parser_train.add_argument("--im_channels", type=int, dest="im_channels", default=1)
    parser_train.add_argument(
        "--loss_type", type=str, dest="loss_type", default="huber"
    )
    parser_train.add_argument("--downsample", action="store_true", dest="downsample")

    # If running the code in debug mode (vscode)
    gettrace = getattr(sys, "gettrace", None)

    if gettrace():
        sys.argv = [
            "main.py",
            "train",
            "--lr",
            "1e-4",
            "--time_steps",
            "800",
            "--jei_flag",
            "1",
            "--group_labels",
            "1",
            "--logdir",
            "test2",
            "--im_size",
            "(192, 224)",
            "--epochs",
            "10",
            "--beta_schedule",
            "linear",
            "--model_idx",
            "1",
        ]

        # sys.argv = [
        #     "main.py",
        #     "resume-train",
        #     "/space/calico/1/users/Harsha/ddpm-labels/logs/20230313-test",
        #     "--debug",
        # ]

        # sys.argv = [
        #     "main.py",
        #     "train",
        #     "--model_idx",
        #     "1",
        #     "--time_steps",
        #     "300",
        #     "--beta_schedule",
        #     "linear",
        #     "--logdir",
        #     "mnist",
        #     "--epochs",
        #     "10",
        #     "--im_size",
        #     "(28, 28)",
        #     "--debug"
        # ]

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_cmdline_arguments()
