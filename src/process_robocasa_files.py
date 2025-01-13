"""
IMPORTANT:
This script prepares data downloaded from the OneDrive link provided on the repo that introduced the dataset: https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
=> Any issue related to the download of this data should be reported on the dataset repo linked above (NOT on DIAMOND's repo)

This script should be called with exactly 2 positional arguments:

- <tar_dir>: folder containing the .tar files from `dataset_dm_scraped_dust2_tars` folder on the OneDrive
- <out_dir>: a new dir (should not exist already), the script will untar and process data there
"""

import argparse
from functools import partial
from pathlib import Path
from multiprocessing import Pool
import shutil
import subprocess

import torch
import torchvision.transforms.functional as T
from tqdm import tqdm

from data.dataset import Dataset, RobocasaHdf5Dataset
from data.episode import Episode
from data.segment import SegmentId


PREFIX = "hdf5_dm_july2021_"

# low res
# IMAGE_HEIGHT = 30
# IMAGE_WIDTH = 56

# IMAGE_HEIGHT = 512
# IMAGE_WIDTH = 512

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tar_dir",
        type=Path,
        help="folder containing the .tar files from `dataset_dm_scraped_dust2_tars` folder on the OneDrive",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="a new directory (should not exist already), the script will untar and process data there",
    )
    return parser.parse_args()


def process_tar(path_tar: Path, out_dir: Path, remove_tar: bool) -> None:
    d = path_tar.stem
    assert path_tar.stem.startswith(PREFIX)
    d = out_dir / "-".join(path_tar.stem[len(PREFIX) :].split("_to_"))
    d.mkdir(exist_ok=False, parents=True)
    shutil.move(path_tar, d)
    subprocess.run(f"cd {d} && tar -xvf {path_tar.name}", shell=True)
    new_path_tar = d / path_tar.name
    if remove_tar:
        new_path_tar.unlink()
    else:
        shutil.move(new_path_tar, path_tar.parent)


def main():
    args = parse_args()

    tar_dir = args.tar_dir.absolute()
    out_dir = args.out_dir.absolute()

    if not tar_dir.exists():
        print(
            "Wrong usage: the tar directory should exist (and contain the downloaded .tar files)"
        )
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"Removed existing directory {out_dir}")

    # with Path("test_split.txt").open("r") as f:
    #     test_files = f.read().split("\n")

    test_files = []

    # full_res_dir = out_dir / "full_res"
    low_res_dir = out_dir / "low_res"

    #
    # Create low-res data
    #

    robocasa_dataset = RobocasaHdf5Dataset(tar_dir)

    train_dataset = Dataset(low_res_dir / "train", None)
    test_dataset = Dataset(low_res_dir / "test", None)

    length_one_episode = 100

    for i in tqdm(robocasa_dataset._filenames, desc="Creating low_res"):
        episode = Episode(
            **{
                k: v
                for k, v in robocasa_dataset[SegmentId(i, 0, length_one_episode)].__dict__.items()
                if k not in ("mask_padding", "id")
            }
        )
        episode.obs = T.resize(
            episode.obs, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.BICUBIC
        )
        filename = robocasa_dataset._filenames[i]
        # file_id = f"{filename.parent.stem}/{filename.name}"
        # episode.info = {"original_file_id": file_id}
        episode.info = {"original_file_id": i}
        dataset = test_dataset if filename in test_files else train_dataset
        dataset.add_episode(episode)

    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()

    print(
        f"Split train/test data ({train_dataset.num_episodes}/{test_dataset.num_episodes} episodes)\n"
    )

    print("You can now edit `config/env/robocasa.yaml` and set:")
    print(f"path_data_low_res: {low_res_dir}")
    print(f"path_data_full_res: {tar_dir}")


if __name__ == "__main__":
    main()
