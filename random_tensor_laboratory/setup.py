# setup.py
from setuptools import setup, find_packages

setup(
    name='random_tensor_laboratory',
    version='0.0',
    description='Random Tensor Laboratory',
    url='http://github.com/tivnanmatt/random_tensor_laboratory',
    author='Matthew Tivnan',
    author_email='tivnanmatt@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'random_tensor_laboratory=random_tensor_laboratory.cli:main',
        ],
    },
)