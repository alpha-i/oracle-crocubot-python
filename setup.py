from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_crocubot_oracle',
    version='3.0.0',
    description='Alpha-I Crocubot',
    author='Sreekumar Thaithara Balan, Christopher Bonnett, Fergus Simpson',
    author_email='sreekumar.balan@alpha-i.co, christopher.bonnett@alpha-i.co, fergus.simpson@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'alphai-time-series>=0.0.3',
        'pandas-market-calendars>=0.6',
        'alphai_covariance>=0.1.3',
        'alphai-data-sources>=1.0.1',
        'tensorflow>=1.4.0',
        'numpy>=1.12.0',
        'pandas==0.18.1',
        'scikit-learn'
    ],
    dependency_links=[
        'git+ssh://git@github.com/alpha-i/library-alphai-covariance.git@0.1.3#egg=alphai_covariance-0.1.3',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_finance/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_time_series/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai-data-sources/'
    ]
)
