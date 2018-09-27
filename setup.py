from setuptools import setup

setup(
    name='hyperoptWorker',
    version='0.1',
    packages=["hyperoptWorker"],
    include_package_data=True,
    install_requires=[
        'azure-storage-file',
        'dill',
        'modelgym'
    ],
)
