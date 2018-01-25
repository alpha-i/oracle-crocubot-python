import glob
import os
import argparse
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_TEMPLATE = '*_train_crocubot*'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Copy train files and create checkpoint')
    parser.add_argument('train_file_dir', type=str, help='Path for source Train file directory')
    parser.add_argument('output_directory', type=str, help='Path for destination Train file directory')
    args = parser.parse_args()

    source_destination_file_list = []

    for single_file in glob.glob1(args.train_file_dir, TRAIN_FILE_TEMPLATE):
        file_parts = single_file.split('_')
        source_destination_file_list.append(
            (single_file, '19990101000000_' + "_".join(file_parts[1:]))
        )

    if len(source_destination_file_list):
        for source, destination in source_destination_file_list:
            shutil.copy(
                os.path.join(args.train_file_dir, source),
                os.path.join(args.output_directory, destination)
            )
    else:
        print("No checkpoint files to copy. Skip")
