# cocotools
A tool for python/shell to manage json dataset files in coco format

## introduce
A tool for python/shell to manage json dataset files in coco format

## usage
1. `coco "edit cmd"` (in shell)
    1. `coco -p 1.json 2.json ...` for print stats
    1. `coco -e gt dt` for evaluation
    1. `coco -m 1.json 2.json ... -o out.json` for merge coco datasets
1. `python -m coco "edit cmd"`
1. `python -m coco.badcase gt dt -i imgdir` for visualizing badcase
1. `python -m coco.distribution_analyze ref coco` for obtain channels which boxes similar with ref's
1. `from coco import COCO; ...` in python code

## install
- `python3 -m pip install https://github.com/zqigolden/cocotools/archive/master.zip`
- `python3 -m pip install https://github.com/zqigolden/cocotools/archive/master.zip -i https://mirrors.aliyun.com/pypi/simple` (Using aliyun source)

## uninstall
- pip uninstall cocotools

## changelog
- 0.2.0.0
    add badcase module for badcase box visualize
    add distribution_analyze module for analyze box distribution in one dataset
- 0.1.1.5
    add keep_images option for filters
- 0.1.1.4
    add debug argument
- 0.1.1.3
    support detection file with num ids
- 0.1.1.2
    add -e for evaluate
- 0.1.1.1
     fix console entry
- 0.1.1
    add argparse
- 0.1.0
    initialize the repo
