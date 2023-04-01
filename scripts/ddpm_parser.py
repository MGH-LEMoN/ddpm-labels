import sys
from argparse import ArgumentParser

from ext.lab2im.utils import infer


def parse_cmdline_arguments():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(help="sub-command help")

    # create subparser for "resume-train" command
    resume = subparsers.add_parser(
        "resume-train", help="Use this sub-command for resuming training"
    )
    resume.add_argument(
        "logdir",
        type=str,
        help="Folder containing previous checkpoints",
    )
    resume.add_argument("--debug", action="store_true", dest="debug")

    # create subparser for "train" command
    train = subparsers.add_parser("train", help="Use this sub-command for training")

    train.add_argument("--debug", action="store_true", dest="debug")
    train.add_argument("--logdir", type=str, dest="logdir", default="test")

    train.add_argument("--model_idx", type=int, dest="model_idx", default=1)
    train.add_argument("--loss_type", type=str, dest="loss_type", default="l2")
    train.add_argument("--time_steps", type=int, dest="time_steps", default=500)
    train.add_argument("--epochs", type=int, dest="epochs", default=500)
    train.add_argument("--batch_size", type=int, dest="batch_size", default=32)
    train.add_argument("--lr", type=float, dest="lr", default=5e-5)

    train.add_argument("--jei_flag", type=int, dest="jei_flag", default=1)
    train.add_argument("--im_channels", type=int, dest="im_channels", default=1)
    train.add_argument("--downsample", action="store_true", dest="downsample")
    train.add_argument("--group_labels", type=int, dest="group_labels", default=0)
    train.add_argument("--im_size", nargs="?", type=infer, dest="im_size", default=None)
    train.add_argument(
        "--beta_schedule", type=str, dest="beta_schedule", default="linear"
    )

    # If running the code in debug mode (vscode)
    gettrace = getattr(sys, "gettrace", None)

    if gettrace():
        sys.argv = [
            "main.py",
            "train",
            "--time_steps",
            "800",
            "--epochs",
            "10",
            "--group_labels",
            "1",
            "--im_size",
            "(192, 224)",
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
        #     "--time_steps",
        #     "300",
        #     "--epochs",
        #     "10",
        #     "--im_size",
        #     "(28, 28)",
        #     "--debug",
        # ]

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmdline_arguments()
