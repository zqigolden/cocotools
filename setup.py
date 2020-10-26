from setuptools import setup

setup(
    name='coco',
    python_requires='>=3.6',
    version='0.1.1.4',
    description='a tool to manage coco dataset file',
    author='qzhu',
    author_email='qzhu.working@gmail.com',
    packages=['coco'],
    install_requires=[
        'opencv-python>=4.2.0',
        'tqdm>=4.36.1',
        'numpy>=1.18.5',
        'pycocotools>=2.0.2'
    ],
    entry_points={'console_scripts': [
        'coco=coco.__main__:main',
    ]},
)
