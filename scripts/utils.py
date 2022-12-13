import glob
import os
import pathlib
from datetime import datetime

from ext.lab2im.utils import load_volume

LABEL_MAPS_PATH = "/cluster/vxmdata1/FS_Slim/proc/cleaned"
SLICE_SHAPE = (256, 256)


def main_timer(func):
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


def save_label_file_names(file_name=None):
    """Save label files to a text file"""
    files = pathlib.Path("/cluster/vxmdata1/FS_Slim/proc/cleaned").glob(
        "*/aseg_23*.mgz"
    )
    with open(
        "/space/calico/1/users/Harsha/ddpm-labels/ddpm_pathlib.txt", "w+"
    ) as f:
        for idx, file in enumerate(files):
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


def load_file_list(filename=None):
    if filename is None:
        filename = (
            "/space/calico/1/users/Harsha/ddpm-labels/data/ddpm_file_list.txt"
        )

    with open(filename, "r") as f:
        file_list = f.read().splitlines()
    return file_list


def list_unique_sites():
    file_list = load_file_list()

    site_list = set()
    for name in file_list:
        subject_id = os.path.split(os.path.dirname(name))[-1]
        site_list.add(subject_id.split("_")[0])

    return site_list


# @main_timer
def count_subjects_per_site(site_list=None):
    if site_list is None:
        site_list = list_unique_sites()

    for site in site_list:
        subjects = glob.glob(os.path.join(LABEL_MAPS_PATH, f"{site}*"))
        print(f"{site}: {len(subjects)}")

    return


def get_slice_shapes(site_list=None):
    if site_list is None:
        site_list = list_unique_sites()

    if isinstance(site_list, str):
        site_list = [site_list]

    for site in site_list:
        print(f"Verifying Site: {site}")
        subjects = glob.glob(os.path.join(LABEL_MAPS_PATH, f"{site}*"))
        for subject in subjects:
            slice_file = os.path.join(subject, "aseg_23_talairach_slice.mgz")
            if not os.path.isfile(slice_file):
                print(f"DNE: {slice_file}")
                continue

            vol_shape = load_volume(slice_file).shape
            if vol_shape != SLICE_SHAPE:
                print(subject)


if __name__ == "__main__":
    site_list = list_unique_sites()
    # count_subjects_per_site(site_list)
    # get_slice_shapes()
