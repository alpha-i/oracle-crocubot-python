#!/usr/bin/env bash

conda install -c conda-forge -y nuitka
pip install yaml

CONFIG_FILE=$1
CHECKPOINT_FILE_DIR=$2

BUILD_FOLDER='build'

declare -a FILES_TO_COPY=(alcova_build/install.sh alcova_build/README.md alcova_build/alcova.yml.dist alcova_build/alcova_cli.py setup.py requirements.txt  alphai_crocubot_oracle)

#copy the files to build
mkdir $BUILD_FOLDER
mkdir $BUILD_FOLDER/runtime
mkdir $BUILD_FOLDER/result

for file_or_dir in ${FILES_TO_COPY[*]}; do
    cp -r $file_or_dir $BUILD_FOLDER/
done;

# BUILD CONFIGURATION
python alcova_build/build_config.py ${CONFIG_FILE} $BUILD_FOLDER

# COPY CHECKPOINT FILES
python alcova_build/build_checkpoint.py ${CHECKPOINT_FILE_DIR} ${BUILD_FOLDER}/runtime

find $BUILD_FOLDER -type d -name "__pycache__" -exec rm -rf {} \;

nuitka --module $BUILD_FOLDER/alcova_init.py --output-dir=$BUILD_FOLDER
rm -rf $BUILD_FOLDER/alcova_init.py $BUILD_FOLDER/alcova_init.build


COMPILE_SUBDIR='alphai_crocubot_oracle/crocubot'
declare -a MODULE_LIST=(evaluate helpers model network train);

#compile all the module
for module in ${MODULE_LIST[*]}; do

    full_module_path=$BUILD_FOLDER/$COMPILE_SUBDIR/$module.py
    echo Building $full_module_path

    nuitka --module $BUILD_FOLDER/$COMPILE_SUBDIR/$module.py --output-dir=$BUILD_FOLDER/$COMPILE_SUBDIR
    rm -rf $BUILD_FOLDER/$COMPILE_SUBDIR/*build

    rm -rf $full_module_path

    echo Removed $full_module_path
done;

#create the tar
CURRENT_TIMESTAMP=`date '+%Y%m%d%H%M%S'`

IFS='=' read -r -a version_string <<< `cat setup.py | grep version`

version_name=${version_string[1]/\',/}
version=${version_name/\'/}
BUILD_NAME=crocubot_${version}_${CURRENT_TIMESTAMP}

mv $BUILD_FOLDER $BUILD_NAME
tar -cvf $BUILD_NAME.tar $BUILD_NAME

#delete the build folder
rm -rf $BUILD_NAME




