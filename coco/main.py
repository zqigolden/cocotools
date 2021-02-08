import argparse
import sys
import os
import json
from typing import List
import typer
from pathlib import Path

from easydict import EasyDict
from loguru import logger

from .coco import COCO

app = typer.Typer()
logger.remove()
logger.add(sys.stderr, level=os.environ.get('LOGLEVEL', 'INFO').upper())


@app.callback()
def debug_callback(debug: bool = typer.Option(False, '--debug', '-d', help='debug level for logging')):
    if debug:
        logger.remove()
        logger.add(sys.stderr, level='DEBUG')


@app.command()
def cmd(args: List[str]):
    for arg in args:
        exec(arg)


@app.command()
def evaluate(gt_file: Path = typer.Argument(..., exists=True, dir_okay=False),
             dt_file: Path = typer.Argument(..., exists=True, dir_okay=False)):
    """
    evaluate coco result
    """
    COCO(gt_file).evaluate(dt_file)


@app.command()
def merge(inputs_cocos: List[Path] = typer.Argument(..., exists=True, dir_okay=False),
          output: Path = typer.Option(..., '-o', '--output', writable=True, dir_okay=False)):
    """
    merge all coco inputs
    """
    assert len(inputs_cocos) >= 2
    dst = COCO(inputs_cocos[0])
    for i in inputs_cocos[1:]:
        logger.debug(i)
        dst.merge(COCO(i))
    dst.to_json(out_file=output, indent=2)


@app.command()
def print_stat(coco_files: List[Path] = typer.Argument(..., exists=True, dir_okay=False)):
    """
    print coco stats (img length, ann length, cat length)
    """
    for i in coco_files:
        print(f'Stat of {i}:')
        COCO(i).print_stat()


@app.command()
def visualize(coco_file: Path = typer.Argument(..., exists=True, dir_okay=False),
              img_dir: Path = typer.Argument(..., exists=True, file_okay=False),
              ):
    """
    visualize inputs
    """
    try:
        COCO(coco_file).visualize(img_dir=img_dir)
    except json.JSONDecodeError:
        logger.info(f'non-coco input file {coco_file} detect, trying to convert')
        empty_gt = COCO.from_image_dir(img_dir).tmp_file_name()
        COCO.from_detect_file(coco_file, empty_gt).visualize(img_dir=img_dir)


@app.command()
def convert_box_labeling(label_file: Path = typer.Argument(..., exists=True, dir_okay=False),
                         image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
                         cats=typer.Option(None, '-c'),
                         output: Path = typer.Option('/dev/stdout', '-o', help='default output to stdout')):
    if cats is None:
        with open(label_file) as f:
            cats = {item['type'] for v in json.load(f).values() for item in v['data']}
            if cats == {'body', 'head', 'realface'}:
                cats = ['body', 'head', 'realface']
            elif cats == {'head', 'realface'}:
                cats = ['head', 'realface']
            elif len(cats) == 1:
                cats = list(cats)
            else:
                raise Exception(f'Cannot auto predict cats from {cats}, using argument like "-c body,head,face"')
    else:
        cats = cats.split(',')
    COCO.from_label_file(
        labeling_file_name=label_file,
        image_dir=image_dir,
        categories_list=cats).to_json(out_file=output, indent=2)


@app.command()
def from_image_dir(image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
                   with_box: bool = typer.Option(False, '--with-box/--no-box', '-wb/-nb'),
                   output_coco: Path = typer.Option('/dev/stdout', '-o', dir_okay=False)):
    logger.debug('enter: from image dir')
    COCO.from_image_dir(image_dir=image_dir, with_box=with_box).to_json(output_coco)


@app.command(help='split dataset into train (80%) and val (20%), arg1:coco arg2:img_dir')
def split_dataset(coco_file: str = typer.Argument(..., exists=True, dir_okay=False),
                  image_dir: str = typer.Argument(..., exists=True, file_okay=False),):
    split_args = EasyDict()
    if 'all' in coco_file:
        split_args.front = coco_file.replace('all', 'train')
        split_args.tail = coco_file.replace('all', 'val')
    if 'all' in image_dir:
        split_args.front_dir_name = image_dir.replace('all', 'train')
        split_args.tail_dir_name = image_dir.replace('all', 'val')
    COCO(coco_file).split_dataset(image_dir=image_dir, **split_args)

# @app.command()
# def convert_kps_labeling(label_file: Path = typer.Argument(..., exists=True, dir_okay=False),
#                          image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
#                          cats: str = typer.Option('car', '-c'),
#                          output: Path = typer.Option(None, '-o', help='default output to stdout')):
#     COCO.from_kps_label_file(label_file, image_dir, cats)


logger.catch(app)()
exit()

def arg_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('inputs', nargs='*', help='input files')
    # parser.add_argument('-d', '--debug', action='store_true',
    #                     help='turn on debug mode')
    # parser.add_argument('-c', '--command', action='store_true',
    #                     help='commands in python')
    # parser.add_argument('-e', '--evaluate', action='store_true',
    #                     help='evaluate coco result, file1: gt; file2: dt')
    # parser.add_argument('-m', '--merge', action='store_true',
    #                     help='merge all inputs, file1: coco; file2: coco, ...')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize inputs, file1: coco; file2: img dir')
    # parser.add_argument('-p', '--print_stat', action='store_true',
    #                     help='visualize coco stats(img len, ann len, cat len), file1: coco')
    parser.add_argument('--split_dataset', action='store_true',
                        help='split dataset into train (80%%) and val (20%%), arg1:coco arg2:img_dir')
    # parser.add_argument('--help', action='store_true', help='print help')
    parser.add_argument('-o', '--output', default='/dev/stdout')
    return parser.parse_args()


@logger.catch()
def main():
    args = arg_parse()
    if args.debug:
        logger.getLogger().setLevel(logger.DEBUG)
    logger.debug(vars(args))

    if not args.command:
        # if args.help:
        #     print(_intro_str)
        #     return
        if args.evaluate:
            assert len(args.inputs) == 2
            gt_file = args.inputs[0]
            det_file = args.inputs[1]
            COCO(gt_file).evaluate(det_file)
        elif args.merge:
            assert len(args.inputs) >= 2
            dst = COCO(args.inputs[0])
            for i in args.inputs[1:]:
                dst.merge(COCO(i))
            dst.to_json(out_file=args.output, indent=2)
        elif args.visualize:
            assert len(args.inputs) == 2
            try:
                COCO(args.inputs[0]).visualize(img_dir=args.inputs[1])
            except json.JSONDecodeError:
                logger.info(f'non-coco input file {args.inputs[1]} detect, trying to convert')
                empty_gt = COCO.from_image_dir(args.inputs[1]).tmp_file_name()
                COCO.from_detect_file(args.inputs[0], empty_gt).visualize(img_dir=args.inputs[1])

        elif args.split_dataset:
            COCO(args.input[0]).split_dataset(image_dir=args.input[1])

        elif args.print_stat:
            assert len(args.inputs) >= 1
            for i in args.inputs:
                print(f'Stat of {i}:')
                COCO(i).print_stat()
        else:
            for cmd in args.inputs:
                eval(cmd)
    else:
        for cmd in args.inputs:
            eval(cmd)
