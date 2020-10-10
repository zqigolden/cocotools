from setuptools import setup

setup(
    name='cocotools',
    python_requires='>=3.6',
    version='0.1.1',
    description='a tool to manage cocotools dataset file',
    author='qzhu',
    author_email='qzhu.working@gmail.com',
    packages=['cocotools'],
    install_requires=[
        'opencv-python>=4.2.0',
        'tqdm>=4.36.1',
        'numpy>=1.18.5',
        'pycocotools'
    ],
    entry_points={'console_scripts': [
        'cocotools = cocotools.__main__:main',
    ]},
)
