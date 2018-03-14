import datetime
import unittest

import pytz

from alphai_crocubot_oracle.oracle import CrocubotOracle
from tests.integration.base_integration import BaseIntegration


class TestCrocubotIntegration(BaseIntegration, unittest.TestCase):
    ORACLE_CLASS = CrocubotOracle
    SIMULATION_START = datetime.datetime(2009, 1, 5, tzinfo=pytz.utc)
    SIMULATION_END = datetime.datetime(2009, 1, 31, tzinfo=pytz.utc)
