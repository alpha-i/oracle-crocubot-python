import glob
import os
import argparse
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Copy train files and create checkpoint')
parser.add_argument('train_file_dir', type=str, help='Path for source Train file directory')
parser.add_argument('output_directory', type=str, help='Path for destination Train file directory')
args = parser.parse_args()


TRAIN_FILE_TEMPLATE = '*_train_crocubot*'
CHECKPOINT_CONTENT = """
model_checkpoint_path: "{}t"
all_model_checkpoint_paths: "{}"
"""

copy_files = []

for single_file in glob.glob1(args.train_file_dir, TRAIN_FILE_TEMPLATE):
    file_parts = single_file.split('_')
    copy_files.append(
        (single_file, '19990101000000_' + "_".join(file_parts[1:]))
    )

if __name__ == '__main__':
    for src, dest in copy_files:
        shutil.copy(
            os.path.join(args.train_file_dir, src),
            os.path.join(args.output_directory, dest)
        )
