import os
import argparse
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Build Alcova Init file from Quant Workflow Config')
parser.add_argument('qw_config', type=str, help='Path for Alcova Config File')
parser.add_argument('output_dir', type=str, help='Path of the Output dir')
args = parser.parse_args()

RUNTIME_DIR_TOKEN = '%%RUNTIME_DIR_PATH%%'
ORACLE_CONFIG_TOKEN = '%%ORACLE_CONFIG%%'
SCHEDULE_CONFIG_TOKEN = '%%SCHEDULE_CONFIG%%'

INIT_TPL_FILE = os.path.join(BASE_DIR, 'alcova_init.tpl.py')
INIT_FILE = os.path.join(args.output_dir, 'alcova_init.py')


def _create_oracle_config_string(parsed_config):
    oracle_config = parsed_config['quant_workflow']['oracle']['oracle_arguments']
    oracle_config['universe'] = parsed_config['quant_workflow']['universe']
    oracle_config['universe']['dropna'] = False
    oracle_config['train_path'] = RUNTIME_DIR_TOKEN
    oracle_config['tensorboard_log_path'] = RUNTIME_DIR_TOKEN
    oracle_config['model_save_path'] = RUNTIME_DIR_TOKEN
    oracle_config['data_transformation']['predict_the_market_close'] = True

    return str(oracle_config).replace("'%%RUNTIME_DIR_PATH%%'", 'RUNTIME_DIR_PATH')


def _create_schedule_string(parsed_config):

    qw_config = parsed_config['quant_workflow']
    oracle_config = qw_config['oracle']

    schedule_config = {
        "prediction_horizon": 24 * oracle_config['oracle_arguments']['data_transformation']['target_delta_ndays'],
        "prediction_frequency": {
            "frequency_type": "DAILY",
            "days_offset": 0,
            "minutes_offset": 300
        },
        "prediction_delta": qw_config['trade_history_ndays'],

        "training_frequency": {
            "frequency_type": "WEEKLY",
            "days_offset": 0,
            "minutes_offset": 300
        },
        "training_delta": qw_config['train_history_ndays'],
    }

    return str(schedule_config)


def _write_alcova_init(replacements):

    with open(INIT_TPL_FILE, 'r') as template_file:
        template_content = template_file.read()

        for key, value in replacements.items():
            template_content = template_content.replace(key, value)

        with open(INIT_FILE, 'w') as init_file:
            init_file.write(template_content)


if __name__ == '__main__':

    with open(args.qw_config, 'r') as qw_config_file:
        source_configuration = yaml.load(qw_config_file)

        tokens = {
            ORACLE_CONFIG_TOKEN : _create_oracle_config_string(source_configuration),
            SCHEDULE_CONFIG_TOKEN: _create_schedule_string(source_configuration)
        }
        _write_alcova_init(tokens)