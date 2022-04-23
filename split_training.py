import argparse
import os
from pathlib import Path

import logging
import shutil

def move_files(abs_dirname, N):
    """Move files into subdirectories."""

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]
    N = int(N)
    i = 0
    curr_subdir = None
    files.sort()

    for f in files:
        # create new subdir if necessary
        if i % N == 0:
            subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i // N + 1))
            os.mkdir(subdir_name)
            curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)
        shutil.move(f, os.path.join(subdir_name, f_base))
        i += 1


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--dir_path", help="path to the images or labels dir")
    ap.add_argument("-n", "--num_images", help="no_images_in_folder")
    args = ap.parse_args()

    label_path = Path(args.dir_path).absolute()

    assert label_path.is_dir(), "Label directory needs to exist"
    
    N = args.num_images

    move_files(label_path, N)


if __name__ == "__main__":
    main()
