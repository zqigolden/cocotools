import sys
import json
from typing import List
import typer
import code
from pathlib import Path

from easydict import EasyDict
from .log_utils import logger

from . import consts
from .coco import COCO

app = typer.Typer()


@app.callback()
def callback(
        debug: bool = typer.Option(False, '-d', '--debug', help='debug level for logging'),
        indent: int = typer.Option(None, '-i', '--indent', help='set the output json indent')):
    if debug:
        consts.DEBUG = True
        logger.remove()
        logger.add(sys.stderr, level='DEBUG')
    if indent:
        consts.INDENT = indent


@app.command()
def cmd(args: List[str]):
    try:
        for arg in args:
            exec(arg)
    except Exception as e:
        logger.info(repr(e))
        from coco import COCO
        code.interact(local=globals().copy().update(locals()))


@app.command()
def evaluate(gt_file: Path = typer.Argument(..., exists=True, dir_okay=False),
             dt_file: Path = typer.Argument(..., exists=True, dir_okay=False)):
    """
    evaluate coco result
    """
    COCO(gt_file).evaluate(str(dt_file))


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
    dst.to_json(out_file=output, indent=consts.INDENT)


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
        logger.debug('visualize {} with images in {}', coco_file, img_dir)
        COCO(coco_file).visualize(img_dir=img_dir)
    except json.JSONDecodeError:
        logger.info(f'non-coco input file {coco_file} detect, trying to convert')
        empty_gt = COCO.from_image_dir(img_dir).tmp_file_name()
        COCO.from_detect_file(str(coco_file), empty_gt).visualize(img_dir=img_dir)


@app.command()
def convert_box_labeling(label_file: Path = typer.Argument(..., exists=True, dir_okay=False),
                         image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
                         cats=typer.Option(None, '-c'),
                         output: Path = typer.Option('/dev/stdout', '-o', help='default output to stdout')):
    """
    convert human labeling result into coco format
    """
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
        categories_list=cats).to_json(out_file=output, indent=consts.INDENT)


@app.command()
def from_image_dir(image_dir: Path = typer.Argument(..., exists=True, file_okay=False),
                   with_box: bool = typer.Option(False, '--with-box/--no-box', '-wb/-nb'),
                   output_coco: Path = typer.Option('/dev/stdout', '-o', dir_okay=False)):
    """
    create empty coco file from image dir
    """
    logger.debug('enter: from image dir')
    COCO.from_image_dir(image_dir=image_dir, with_box=with_box).to_json(output_coco, indent=consts.INDENT)


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
    COCO(coco_file).split_dataset(image_dir=image_dir, indent=consts.INDENT, **split_args)


@app.command(help='convert image id to num')
def to_num_id(coco_file: Path = typer.Argument(..., exists=True, dir_okay=False)):
    output_file = coco_file.with_name(f'{coco_file.stem}.num_id{coco_file.suffix}')
    COCO(coco_file).to_num_id().to_json(output_file, indent=consts.INDENT)
    print(f'output to {output_file}')

@app.command(help='convert image id to str')
def to_str_id(coco_file: Path = typer.Argument(..., exists=True, dir_okay=False)):
    output_file = coco_file.with_name(f'{coco_file.stem}.str_id{coco_file.suffix}')
    COCO(coco_file).to_str_id().to_json(output_file, indent=consts.INDENT)
    print(f'output to {output_file}')


logger.catch(app)()
