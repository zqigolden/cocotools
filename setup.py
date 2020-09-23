from os.path import basename
import shutil
import setuptools
from setuptools import setup
from setuptools.command.install_scripts import install_scripts

setup(
    name='cocotools',
    python_requires='>=3.6',
    version='0.1.0',
    description='a tool to manage coco dataset file',
    author='qzhu',
    author_email='qzhu.working@gmail.com',
    packages=['coco'],
    install_requires=[
        'opencv-python>=4.2.0',
        'tqdm>=4.36.1',
        'numpy>=1.18.5',
    ],
    entry_points={'console_scripts': [
        'coco = coco.__main__:main',
    ]},
)
