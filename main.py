"""
Nazwa: main.py
Opis: Punkt wejścia do programu. Obsługa argumentów wejściowych.
Autor: Bartłomiej Moroz, Jan Walczak
"""
import argparse
from src.main import main


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv3")
    parser.add_argument(
        "mode", type=str, choices=["train", "test", "predict"], help="Mode of action"
    )
    parser.add_argument(
        "-b", "--batch_count", type=int, default=1, help="Number of batches to work on"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Load model checkpoint from file",
    )
    parser.add_argument(
        "-e",
        "--earlystop",
        type=bool,
        default=True,
        help="Should use early stop callback?",
    )
    parser.add_argument(
        "-g",
        "--checkpoint_gid",
        type=str,
        default=None,
        help="Load model checkpoint from Google Drive",
    )
    parser.add_argument(
        "-i", "--input", type=str, default="inputs", help="Prediction inputs directory"
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        choices=["csv", "tensorboard", "wandb"],
        default="csv",
        help="Logger to use",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions",
        help="Prediction outputs directory",
    )
    parser.add_argument(
        "-s", "--batch_size", type=int, default=8, help="Number of images in batch"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
