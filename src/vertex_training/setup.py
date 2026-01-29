from setuptools import setup, find_packages

setup(
    name='trainer',
    version='1.2',
    packages=find_packages(),
    install_requires=[
        'lightgbm>=3.3.0',
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scikit-learn>=1.2.0',
        'google-cloud-storage>=2.0.0',
        'pyarrow>=10.0.0',
    ],
    python_requires='>=3.8',
)
