import tempfile
import unittest

from datetime import datetime

import os

from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.helpers import TrainFileManager
from alphai_crocubot_oracle.oracle import TRAIN_FILE_NAME_TEMPLATE
from tests.helpers import create_fixtures, destroy_fixtures, FIXTURE_TRAIN_DESTINATION_DIR


class TestFileManager(unittest.TestCase):

    def setUp(self):
        create_fixtures()

    def tearDown(self):
        destroy_fixtures()

    def test_no_calibration(self):

        fake_dir = tempfile.TemporaryDirectory().name
        file_manager = TrainFileManager(
            fake_dir,
            TRAIN_FILE_NAME_TEMPLATE,
            DATETIME_FORMAT_COMPACT
        )

        file_manager.ensure_path_exists()
        with self.assertRaises(ValueError):
            file_manager.latest_train_filename(datetime(1998, 1, 1))

        os.rmdir(fake_dir)

    def test_load_file(self):

        file_manager = TrainFileManager(
            FIXTURE_TRAIN_DESTINATION_DIR,
            TRAIN_FILE_NAME_TEMPLATE,
            DATETIME_FORMAT_COMPACT
        )

        train_file = file_manager.latest_train_filename(datetime(1998, 1, 1))
        assert train_file == os.path.join(FIXTURE_TRAIN_DESTINATION_DIR, '19000101000000_train_crocubot')

        train_file = file_manager.latest_train_filename(datetime(2001, 1, 1))
        assert train_file == os.path.join(FIXTURE_TRAIN_DESTINATION_DIR, '20000101000000_train_crocubot')

        train_file = file_manager.latest_train_filename(datetime(2000, 1, 1))
        assert train_file == os.path.join(FIXTURE_TRAIN_DESTINATION_DIR, '19000101000000_train_crocubot')

        train_file = file_manager.latest_train_filename(datetime(2000, 1, 1, 1))
        assert train_file == os.path.join(FIXTURE_TRAIN_DESTINATION_DIR, '20000101000000_train_crocubot')
