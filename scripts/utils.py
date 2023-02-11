import glob
import os
import pathlib
import shutil
import sys

sys.path.append("/space/calico/1/users/Harsha/ddpm-labels")

from datetime import datetime

import numpy as np
from tqdm import tqdm

from ext.lab2im import edit_volumes
from ext.lab2im.utils import (
    find_closest_number_divisible_by_m,
    load_volume,
    save_volume,
)

PRJCT_DIR = "/space/calico/1/users/Harsha/ddpm-labels"
DATA_DIR = os.path.join(PRJCT_DIR, "data")
RESULTS_DIR = os.path.join(PRJCT_DIR, "results")

LABEL_MAPS = "/cluster/vxmdata1/FS_Slim/proc/cleaned"
LABEL_MAPS_COPY = os.path.join(DATA_DIR, "label-maps")
LABEL_MAPS_COMPACT = os.path.join(DATA_DIR, "label-maps-compact")
LABEL_MAPS_PADDED = os.path.join(DATA_DIR, "label-maps-padded")

SLICE_SHAPE = (256, 256)


def main_timer(func):
    """Decorator to time any function"""

    def function_wrapper():
        start_time = datetime.now()
        # print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        # print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(
            f"Function: {func.__name__} Total runtime: {end_time - start_time} (HH:MM:SS)"
        )

    return function_wrapper


def process(filename):
    """Removes empty lines and lines that contain only whitespace, and
    lines with comments"""

    with open(filename) as in_file, open(filename, "r+") as out_file:
        for line in in_file:
            if not line.strip().startswith("#") and not line.isspace():
                out_file.writelines(line)


def write_labelmap_names(folder=LABEL_MAPS, file_name=None):
    """Save label files to a text file"""
    if not folder:
        folder = "/cluster/vxmdata1/FS_Slim/proc/cleaned"
        files = pathlib.Path(folder).glob("*/aseg_23*.mgz")
    else:
        files = pathlib.Path(folder).glob("*.mgz")

    if not file_name:
        file_name = os.path.join(DATA_DIR, "ddpm_pathlib.txt")

    if not os.path.isabs(file_name):
        file_name = os.path.join(DATA_DIR, file_name)

    with open(file_name, "w+") as f:
        for file in sorted(list(files)):
            # curr_name = '/'.join(file.parts[-2:])
            f.write(str(file) + "\n")


# @main_timer
# def list_unique_sites():
#     p = pathlib.Path(LABEL_MAPS_PATH)

#     site_list = set()
#     for x in p.iterdir():
#         if x.is_dir():
#             site_list.add(x.name.split("_")[0])

#     return site_list

# # Paths to all labels maps (which will be loaded in the Dataset object)
# filename = list(pathlib.Path(DATA_PATH).glob("*/aseg_23*.mgz"))


def load_labelmap_names(filename=None):
    if filename is None:
        filename = os.path.join(DATA_DIR, "ddpm_file_list2.txt")

    if not os.path.isabs(filename):
        filename = os.path.join(DATA_DIR, filename)

    with open(filename, "r") as f:
        file_list = f.read().splitlines()
    return file_list


def list_unique_sites():
    """Print unique sites from the label maps name"""
    file_list = load_labelmap_names()

    site_list = set()
    for name in file_list:
        subject_id = os.path.split(os.path.dirname(name))[-1]
        site_list.add(subject_id.split("_")[0])

    return site_list


def count_subjects_per_site(site_list=None):
    if site_list is None:
        site_list = list_unique_sites()

    for site in site_list:
        subjects = glob.glob(os.path.join(LABEL_MAPS, f"{site}*"))
        print(f"{site}: {len(subjects)}")


def get_slice_shapes(file_name=None):
    label_maps = load_labelmap_names(file_name)
    shape_set = set()
    for label_map in label_maps:
        vol_shape = load_volume(label_map).shape
        shape_set.add(vol_shape)

    print(shape_set)


def get_site_slice_shapes(site_list=None):
    if site_list is None:
        site_list = list_unique_sites()

    if isinstance(site_list, str):
        site_list = [site_list]

    for site in site_list:
        print(f"Verifying Site: {site}")
        subjects = glob.glob(os.path.join(LABEL_MAPS, f"{site}*"))
        for subject in subjects:
            slice_file = os.path.join(subject, "aseg_23_talairach_slice.mgz")
            if not os.path.isfile(slice_file):
                print(f"DNE: {slice_file}")
                continue

            vol_shape = load_volume(slice_file).shape
            if vol_shape != SLICE_SHAPE:
                print(subject)


def copy_label_maps():
    """Copy label maps from source folder to destination"""
    dst_dir = "/space/calico/1/users/Harsha/ddpm-labels/data/label-maps"
    os.makedirs(dst_dir, exist_ok=True)

    filename = load_labelmap_names()

    for file in tqdm(filename):
        dst_file = pathlib.Path(file).parent.parts[-1] + ".mgz"
        shutil.copyfile(file, os.path.join(dst_dir, dst_file))


def create_compact_label_maps(labels_dir=None, result_dir=None):
    if not labels_dir:
        labels_dir = LABEL_MAPS_COPY

    if not os.path.isabs(labels_dir):
        labels_dir = os.path.join(DATA_DIR, labels_dir)

    if not result_dir:
        result_dir = os.path.join(DATA_DIR, "label-maps-compact")

    if not os.path.isabs(result_dir):
        result_dir = os.path.join(DATA_DIR, result_dir)

    if not os.path.isdir(labels_dir):
        print("Input directory with Label maps does not exist")

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    maximum_size = edit_volumes.crop_dataset_to_minimum_size(
        labels_dir, result_dir, image_dir=None, image_result_dir=None, margin=0
    )

    return maximum_size


def pad_compact_label_maps(maximum_size, labels_dir=None, result_dir=None):

    if isinstance(maximum_size, np.ndarray):
        pad_to_shape = []
        for val in maximum_size:
            pad_to_shape.append(
                find_closest_number_divisible_by_m(
                    val, 32, answer_type="higher"
                )
            )
            print(val, pad_to_shape)

    if isinstance(maximum_size, (int, float)):
        pad_to_shape = find_closest_number_divisible_by_m(
            val, 32, answer_type="higher"
        )
        print(val, pad_to_shape)

    if not labels_dir:
        labels_dir = os.path.join(DATA_DIR, LABEL_MAPS_COMPACT)

    if not os.path.isabs(labels_dir):
        labels_dir = os.path.join(DATA_DIR, labels_dir)

    if not result_dir:
        result_dir = os.path.join(DATA_DIR, LABEL_MAPS_PADDED)

    if not os.path.isabs(result_dir):
        result_dir = os.path.join(DATA_DIR, result_dir)

    file_list = pathlib.Path(labels_dir).glob("*.mgz")

    for file in file_list:
        volume, aff, header = load_volume(str(file), im_only=False)
        padded_volume = edit_volumes.pad_volume(volume, pad_to_shape)
        save_volume(
            padded_volume, aff, header, os.path.join(result_dir, file.name)
        )


if __name__ == "__main__":
    # site_list = list_unique_sites()
    # print(site_list)
    # # copy_label_maps()
    # # count_subjects_per_site(site_list)
    # # get_slice_shapes()
    # # write_labelmap_names()

    # max_size = create_compact_label_maps(
    #     labels_dir="label-maps", result_dir="label-maps-compact"
    # )
    # # get_slice_shapes("ddpm_files_compact.txt")
    # pad_compact_label_maps(
    #     max_size, labels_dir="label-maps-compact", result_dir="label-maps-padded"
    # )
    # write_labelmap_names(LABEL_MAPS_COMPACT, "ddpm_files_compact.txt")
    # get_slice_shapes("ddpm_files_compact.txt")
    write_labelmap_names(LABEL_MAPS_PADDED, "ddpm_files_padded.txt")
    get_slice_shapes("ddpm_files_padded.txt")
