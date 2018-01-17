#!/usr/bin/env bash

#Install nuitka (conda install -c conda-forge nuitka)
#Install python2 (which is a Scones dependency);
#MAKE SURE YOU'RE NOT USING PYTHON 3.6, which is unsupported (their website is a lie);
#nuitka --module [--recurse-all] path/to/module/module;
#[maybe there's no need to --recurse-all, especially if the plan is just to compile the module]
#Delete dist and build artifacts
#The resulting .so file is the compiled oracle artifact, which can be imported and run from python modules (as it was cython).

conda install -c conda-forge -y nuitka

BUILD_FOLDER='build'

declare -a FILES_TO_COPY=(alcova_runtime/install.sh alcova_runtime/README.md alcova_runtime/alcova.yml.dist alcova_runtime/alcova_cli.py alcova_runtime/alcova_init.py setup.py requirements.txt  alphai_crocubot_oracle)

#copy the files to build
mkdir $BUILD_FOLDER
mkdir $BUILD_FOLDER/runtime
mkdir $BUILD_FOLDER/result

for file_or_dir in ${FILES_TO_COPY[*]}; do
    cp -r $file_or_dir $BUILD_FOLDER/
done;

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
tar -cvf $BUILD_NAME.tar.gz $BUILD_NAME

#delete the build folder
rm -rf $BUILD_NAME




