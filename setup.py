from setuptools import setup

setup(
    name='coco',
    python_requires='>=3.6',
    version='0.2.0.1',
    description='a tool to manage coco dataset file',
    author='qzhu',
    author_email='qzhu.working@gmail.com',
    packages=['coco'],
    install_requires=[
        'numpy',
        'scipy',
        'loguru',
        'opencv-python',
        'tqdm',
        'pycocotools==1.17.5',
        'typer[all]',
        'setuptools',
        'imagesize',
        'easydict',
    ],
    entry_points={'console_scripts': [
        'coco=coco.__main__:main',
    ]},
)
