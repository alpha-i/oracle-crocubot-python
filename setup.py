from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_crocubot_oracle',
    version='4.0.1',
    description='Alpha-I Crocubot',
    author='Sreekumar Thaithara Balan, Christopher Bonnett, Fergus Simpson',
    author_email='sreekumar.balan@alpha-i.co, christopher.bonnett@alpha-i.co, fergus.simpson@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'alphai_delphi>=2.0.0,<3.0.0',
        'alphai_finance==2.0.0',
        'alphai-time-series>=0.0.4',
        'alphai_covariance==0.1.4',
        'alphai-data-sources>=1.0.1',
        'alphai_feature_generation>=2.0.2,<3.0.0',
        'tensorflow==1.4.0',
        'numpy>=1.12.0',
        'pandas==0.18.1',
        'scikit-learn>=0.19.0',
    ],
    dependency_links=[
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_delphi/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_finance/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_time_series/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai-data-sources/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_feature_generation/'
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_covariance/'
    ]
)
