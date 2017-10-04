from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_crocubot_oracle',
    version='1.0.2',
    description='Alpha-I Crocubot',
    author='Sreekumar Thaithara Balan, Christopher Bonnett, Fergus Simpson',
    author_email='sreekumar.balan@alpha-i.co, christopher.bonnett@alpha-i.co, fergus.simpson@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'alphai-time-series>=0.0.3',
        'alphai-finance<=1.2.1',
        'pandas-market-calendars>=0.6',
        'alphai_covariance>=0.1.2',
        'alphai-data-sources>=1.0.1',
        'tensorflow>=1.3.0',
        'numpy>=1.11.3',
        'pandas>=0.18.1',

    ],
    dependency_links=[
        'git+ssh://git@github.com/alpha-i/library-alphai-covariance.git@5a23a4496a58fb20dfb1781fa7bcd8951cc76786#egg=alphai_covariance-0.1.1',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_finance/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_time_series/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai-data-sources/'
    ]
)
