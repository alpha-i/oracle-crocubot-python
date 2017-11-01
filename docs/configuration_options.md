# Configuration Options

The main sections of the configuration file are:
+ [`logging`](#logging)
+ [`quant_workflow`](#quant_workflow)
+ [`zipline`](#zipline)
+ [`live_clock_configuration`](#live_clock_configuration)

The `yaml` will look like:
```yaml
logging:
quant_workflow:
zipline:
live_clock_configuration:
```

Defining the crocubot-oracle is separately discussed in the [crocubot_options](crocubot_options.md)

We will now explain the options to configure a backtest.

## `logging`
This section specifies the logging options for the backtest. For more information about the keys here
please see the official python 
[documentation](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details).

## `quant_workflow`
For example, if we are using the `http` this section will look like the one below:
```yaml
quant_workflow:
  run_mode: 'backtest'
  results_path: 'D:\Zipline\20100101_20150101_10S\results'
  fill_limit: 5
  trade_resample_rule: '15T'
  trade_history_ndays: 31
  trade_frequency: 'weekly'
  trade_days_offset: 1
  trade_minutes_offset: 60
  trade_horizon_ncycles: 1
  train_resample_rule: '15T'
  train_history_ndays: 100
  train_frequency: 'weekly'
  train_days_offset: 0
  train_minutes_offset: 60
  alert_level: 'NONE'
  execution_timeout: 180.
  open_order_timeout: 3600.
  oracle:
    method: 'http'
    protocol: 'http'
    host: '0.0.0.0'
    port: 8080
  portfolio:
    max_abs_individual_weight: 0.2
    max_abs_pos_gross_exposure: 0.75
    max_abs_neg_gross_exposure: 0.75
    margin_ratio: 1.0
    max_annualised_std: 0.3
  universe:
    method: 'fixed'
    symbol_list: ['AAPL', 'GOOGL', 'XOM', 'MSFT', 'JNJ', 'JPM', 'IBM', 'PG', 'BAC', 'T']
```
The main keys under `quant_workflow` are:

| key | description |
| --- | --- |
| `run_mode` | This key specifies the mode in which the `quant_workflow` is run. The possible options are `'backtest'`, `'oracle'` and `'live'`.
| `results_path` | specifies the path in which results files are to be written. e.g `'D:\Zipline\20100101_20150101_10S\results'` |
| `fill_limit` | specifies the maximum number of 1-minute time stamps in which missing data can be filled using previous values. |
| `trade_resample_rule` | the period with which the data is to be re-sampled before passing to the oracle. We use `pandas` notation. e.g `'15T'`|
| `trade_history_ndays` | the number days of historical data provided for inference. |
| `trade_frequency` | how often we trade. Options are `'weekly'` and `'daily'` |
| `trade_days_offset` | Specifies the day we trade. `0` implies Monday and so on. :exclamation: ONLY USED IF `trade_frequency: 'weekly'`. |
| `trade_minutes_offset` | specifies the time at which the trade happens after the market opens. e.g. `60` implies an hour after. |
| `trade_horizon_ncycles` | The number units of `trade_frequency` at which the oracle should do the prediction. |
| `train_resample_rule` | :exclamation: THIS SHOULD BE IDENTICAL TO `trade_resample_rule`. e.g `'15T'` |
| `train_history_ndays` | the number days of historical data provided for training. .e.g `100`. |
| `train_frequency` | specifies how often we train the network. Can be different from the `trade_frequency`. Options are `'weekly'` and `'daily'` |
| `train_days_offset` | Specifies the day we train. `0` implies Monday and so on. :exclamation: ONLY USED IF `train_frequency: 'weekly'`. | 
| `train_minutes_offset` | specifies the time at which the training happens after the market opens. |
| `alert_level` | specifies the level of alert messages. Possible values are `'NONE'` :exclamation: NOT USED FOR NOW|
| `execution_timeout` | specifies the time out in *seconds* after which the execution will be killed. :exclamation: NOT USED FOR NOW|
| `open_order_timeout` | specifies the time out in *seconds* after which the open orders will be killed. :exclamation: NOT USED FOR NOW|
| `oracle` | Configuration parameters for the oracle as sub a dictionary. See the section on `oracle`. |
| `portfolio` | Configuration parameters for the to portfolio creation as sub a dictionary. See the section on `portfolio`. |
| `universe` | Configuration parameters for the to stock universe selection  as sub a dictionary. See the section on `universe`.
 
## `oracle`
The `oracle` section should contain a key called `method` which can either be `'http'` or `'library'`.
If `'http'` is specified as the `method` we need further keys as follows:
```yaml
oracle:
  method: 'http'
  protocol: 'http'
  host: '0.0.0.0'
  port: 8080
```

| key | description |
| --- | --- |
| `protocol` | what protocol to use for communication. e.g. `'http'` |
| `host` | hostname e.g. `'0.0.0.0'` |
| `port` | the port for communication `8080` |


If we are using the `library` option for the `oracle` we need to specify the following keys.

| key | description |
| --- | --- |
| `module_path` | a path to the module that is being used as the oracle. e.g. `alphai_crocubot_oracle.oracle` |
| `oracle_class_name` | name of the oracle python class. e.g. `CrocubotOracle`. |
| `oracle_arguments` | This subsection will specify *all* the arguments required to create a `oracle` class. |

For example, if we are using `alphai_crocubot_oracle.oracle`, the `oracle` section will look like:
```yaml
oracle:
  method: library
  module_path: alphai_crocubot_oracle.oracle
  oracle_class_name: CrocubotOracle
  oracle_arguments: ***
```
For more details on the oracle arguments see the [crocubot options](crocubot_options.md) page.

### `portfolio`
The `portfolio` section can be used to specify the portfolio creation. The following keys are valid.

| key | description |
| --- | --- |
| `max_individual_weight` | scalar - bound on long positions for individual stocks |
| `min_individual_weight` |scalar - bound on short positions for individual stocks |
| `max_aggregate_exposure` | scalar - bound on net long position for portfolio |
| `min_aggregate_exposure` | scalar - bound on net short position for portfolio |
| `margin_ratio` | scalar > 1 - max allowed ratio of invested capital to real value of assets  |
| `max_annualised_std` | scalar - max standard deviation of the portfolio over a year |
| `alpha_plus` | scalar - linear cost for buying a unit of stock |
| `alpha_minus` | scalar - linear cost for shorting a unit of stock |
| `beta_plus` | scalar - fixed cost for buying a nonzero quantity of stock |
| `beta_minus` | scalar - fixed cost for shorting a nonzero quantity of stock  |

### `universe` 
The `universe` section deals with the universe creation. This section requires the key `method` to be specified. It 
can either be `'fixed'`, `'liquidity'` or `'fixed_historical'`. 
If `method: 'fixed'`, the universe is constant over time and equal to the user input:
```yaml
  universe:
    method: 'fixed'
    symbol_list: ['AAPL', 'GOOGL', 'XOM', 'MSFT', 'JNJ', 'JPM', 'IBM', 'PG', 'BAC', 'T']
```
| key | description |
| --- | --- |
| `symbol_list` | List of symbols of the fixed universe. |

If `method: 'liquidity'`, the universe changes over time to reflect the most liquid stocks available:
```yaml
  universe:
    method: 'liquidity'
    nassets: 10
    ndays_window: 30
    update_frequency: 'monthly'
    avg_function: 'median'
    fill_limit: 5
```
| key | description |
| --- | --- |
| `nassets` | Number of assets to select. E.g.`10` means that the 10 most liquid assets are selected. |
| `ndays_window` | Number of days over which the historical liquidity calculation is performed. |
| `update_frequency` | Frequency of update of the assets in the universe. Options are [`'daily'`, `'weekly'`, `'monthly'`, `'yearly'`] |
| `avg_function` | Averaging function to be used in the calculation of liquidity. Options are [`'median'`, `'mean'`] |
| `fill_limit` | Maximum number of 1-minute time stamps in which missing data can be filled using previous values, for the calculation of liquidity. |

If `method: 'fixed_historical'`, then the liquidity calculation in the previous step is done using a file provided. The
only argument required is below.

| key | description |
| --- | --- |
| `historical_universes_filepath` | A file providing historical universe selection. |

*Note that your ingested data bundle should contain these stocks for this specification to work!*

## `zipline`
This section defines the parameters related to `zipline` library. The following keys are required.

| key | description |
| --- | --- |
| `zipline_root` | The path to `zipline` root where the `extension.py` and the `data` folder resides. e.g. `'D:\Zipline\20100101_20150101_10S\zipline_root'`. |
| `start_date` | start date of the run. e.g. `'20110401'`.|
| `end_date` | end date of the run. e.g. `'20110601'`.|
| `capital_base` | The amount of capital available. e.g. `100000` for $100k |
| `data_frequency` | The data frequency of the ingested data. :exclamation: SHOULD ALWAYS BE `'minute'`. |
| `data_bundle` | the name of the data bundle. This should be defined in the `extension.py` in the `zipline_root`.
| `slippage_type` | the type of slippage to be used in the backtest. :exclamation: SHOULD ALWAYS BE `'TradeAtTheOpenSlippageModel'` |
| `spread` | bid/ask spread SHOULD ALWAYS BE `0.` as we evaluate multiple spreads when analysing performance. |
| `open_close_fraction` | between 0 and 1. Distance between open and close to pick the execution price. e.g. `0.1` execution = open + 0.1*(close-open) |
| `volume_limit`| USE `0.` :exclamation: DEPRECATED: TO BE REMOVED |
| `price_impact`| USE `0.` :exclamation: DEPRECATED: TO BE REMOVED |
| `commission_type` | The type of commission model. Options are [`'PerShare'`, `'PerTrade'`, `'PerDollar'`] |
| `cost` | Defines the commission cost according to `commission_type`. |
| `min_trade_cost` | Minumum commission cost in dollars :exclamation: ONLY USED IF `commission_type` is `'PerTrade'`. |

Thus a fully specified `zipline` section will look like
```yaml
zipline:
  zipline_root: 'D:\Zipline\20100101_20150101_10S\zipline_root'
  start_date: '20110401'
  end_date: '20110430'
  capital_base: 1000000.
  data_frequency: 'minute'
  data_bundle: 'test_bundle'
  slippage_type: 'TradeAtTheOpenSlippageModel'
  spread: 0.
  open_close_fraction: 0.
  volume_limit: 0.
  price_impact: 0.
  commission_type: 'PerShare'
  cost: 0.0005
  min_trade_cost: 1.
```

## `live_clock_configuration`
This section is used to define the details about the live clock. This section is only used in the `live` mode. 
No need to change these options in other modes. 

| key | description |
| --- | --- |
|  `host` | defines the host. |
| `port` | port for communication.|
| `queue_name` | name of the queue. |

For example, using the above values, this section can be written as
```yaml
live_clock_configuration:
  host: 'localhost'
  port: 45672
  queue_name: 'clock-pulse'
```
