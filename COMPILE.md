Compilation how to
==================

The process of compilation creates a tar which can be installed on the alcova machine.

During the compilation time, the `qw configuration` is translate into `scheduling` and `oracle_config` 
for the runner.

If there are some `train` files in the `checkpoint_dir` parameter, those will be copied and renamed
and they will become part of the package.

The core crocubot files are compiled to obfuscate sensitive code

Compiling
---------

The command to compile the model is the following
```bash
$ ./compile.sh <configuration_file> <checkpoint_dir>
```

Package content
---------------

The result of the tar will be a directory structure containing the following

```
alphai_crocubot_oracle/ # crocubot python package
result/ # directory where the result will be saved
runtime/ # directory where the runtime files will be create. any train file will be put here
README.md # installation instruction
alcova_cli.py # main script
alcova_init.so # compiled configuration
alcova.yml.dist # example of alcova config
install.sh # scrip to create the environment and install the software
requirements.txt # pip requirements files
setup.py # setup of the crocubot
```

Once the package is build, you can follow the README.md file to get instruction on
how to install and run the model in the alcova machine

