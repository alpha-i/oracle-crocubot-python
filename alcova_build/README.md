Running the model
=================


Installation
------------

1. Create the conda environment and install packages
```bash
$ ./install.sh environment_name
```


Running a backtest
------------------

1. Create the configuration: copy `alcova.yml.dist` to `alcova.yml` and insert the required values
   
   The file must be located in the same directory of the `alcova_cli.py` and `alcova_init.so`

2. Run the backtest
```bash
$ source activate environment_name
$ python alcova_cli.py
```

3. At the end of the backtest you'll find your result in hdf5 file `result/alcova_oracle_results_mean_vector.hdf5`
   
   The file is indexed by a time-stamp string with the format (YMD-HMS). The time is expressed in UTC zone.
   
   At each index you'll find a pd.Series containing  stock_id / expected return.
