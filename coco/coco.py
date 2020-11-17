#!/usr/bin/env python3

import argparse
import copy
import fnmatch
import json
import os
import random
import re
import shutil
import sys
import tempfile
from loguru import logger
from collections import OrderedDict, defaultdict, Counter
from operator import itemgetter
from typing import List, Dict, Set, Iterable, Union, Optional

import cv2
import numpy as np

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger.remove()
logger.add(sys.stderr, level=LOGLEVEL)

try:
    from tqdm import tqdm
except ImportError:
    print('tqdm not found')


    def tqdm(it, *args, **kwargs):
        return it

MAX_INT = sys.maxsize
IMAGES = 'images'
ANNOTATIONS = 'annotations'
CATEGORIES = 'categories'
ANN_IMG_ID = 'image_id'
ANN_CAT_ID = 'category_id'
IMG_FILENAME = 'file_name'


def promise_set(in_val=None) -> set:
    """
    make anything a set
    :param in_val: input data
    :return: a set suitable in_val
    """
    if in_val is None:
        return set()
    if not isinstance(in_val, str) and isinstance(in_val, Iterable):
        return set(in_val)
    return {in_val}


def get_box(bbox):
    return np.s_[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]


def limit_box_in_image(box: List, cols: int = MAX_INT, rows: int = MAX_INT) -> List:
    """
    limit the box in image which has the size **cols** and **rows**
    :param box: [x, y, w, h] box
    :param cols: the image col size
    :param rows: the image row size
    :return: limited box
    """
    box[2] = int(min(cols - box[0], box[2]))
    box[3] = int(min(rows - box[1], box[3]))
    box[0] = int(max(0, box[0]))
    box[1] = int(max(0, box[1]))
    return box


def get_set(data: Iterable[Dict], label: str = 'id') -> Set[str]:
    """
    get the unique set from the attribute **label** in data
    :param data:
    :param label:
    :return: a set with all **label** in **data**
    """
    return set([i[label] for i in data])


def get_list(data: Iterable[Dict], label: str = 'id') -> List[str]:
    """
    get the ordered list from the attribute **label** in data
    :param data:
    :param label:
    :return: a list with all **label** in **data**
    """
    return [i[label] for i in data]


def get_iter(data: Iterable[Dict], label: str = 'id') -> Iterable[str]:
    """
    get the iterator from the attribute **label** in data
    :param data:
    :param label:
    :return: a iterator with all **label** in **data**
    """
    for i in data:
        yield i[label]


def group_by(data: Iterable[Dict], label: str = 'id'):
    r = defaultdict(list)
    for d in data:
        r[d[label]].append(d)
    return r


def filter_with(data: Iterable[Dict], keep_set: Set[str], label: str = 'id') -> Iterable[Dict]:
    return [i for i in data if i[label] in keep_set]


def mapping(data: Iterable[Dict], mapping_dict: Dict, label: str = 'id') -> List[Dict]:
    def map_it(x):
        return mapping_dict[x] if x in mapping_dict else x

    dst = []
    for i in data:
        i[label] = map_it(i[label])
        dst.append(i)
    return dst


def remap_num(data: List[Dict], label='id', start=0, order_by=None) -> List[dict]:
    """
    Change to int sequence, no other operations
    :param order_by:
    :param start:
    :param label:
    :param data:
    :return:
    """
    if order_by:
        data.sort(key=lambda x: x[order_by])
    for loc, i in enumerate(data, start=start):
        i[label] = loc
    return data


def make_dt_by_gt(dt_file, gt_coco):
    with open(dt_file) as f:
        dt_content = f.read()
        try:
            json.loads(dt_content)
            return dt_file
        except:
            try:
                # format: imgname [conf cls x y w h] ...
                dt = []
                gt_coco.imgname_id_dict(cache=False)
                for line in dt_content.strip().split('\n'):
                    items = line.strip().split(' ')
                    assert len(items) % 6 == 1
                    file_name = items[0]
                    file_id = gt_coco.imgname_id_dict()[file_name]
                    del items[0]
                    while items:
                        conf, cls, x, y, w, h = items[:6]
                        dt.append({
                            ANN_IMG_ID: file_id,
                            ANN_CAT_ID: int(cls),
                            'score': float(conf),
                            'bbox': [int(x), int(y), int(w), int(h)]
                        })
                        del items[:6]
            except:
                # format: imgname [conf cls x y w h unknown1 unknown2] ...
                dt = []
                gt_coco.imgname_id_dict(cache=False)
                for line in dt_content.strip().split('\n'):
                    items = line.strip().split(' ')
                    assert len(items) % 8 == 1
                    file_name = items[0]
                    file_id = gt_coco.imgname_id_dict()[file_name]
                    del items[0]
                    while items:
                        conf, cls, x, y, w, h, _, _ = items[:8]
                        dt.append({
                            ANN_IMG_ID: file_id,
                            ANN_CAT_ID: int(cls),
                            'score': float(conf),
                            'bbox': [int(x), int(y), int(w), int(h)]
                        })
                        del items[:8]
            tmp_file_dt = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            json.dump(dt, tmp_file_dt)
            return tmp_file_dt.name


class COCO:
    def __len__(self):
        return len(self.images)

    def __init__(self, filename=None, json_str=None, obj=None):

        # alias
        self.vis = self.visualize
        self.v = self.visualize
        raw: Dict = {}
        if filename:
            with open(filename) as f:
                raw = json.load(f)
        elif json_str is not None:
            raw = json.loads(json_str)
        elif obj is not None:
            raw = obj
        self.images: List[Dict] = raw[IMAGES]
        self.annotations: List[Dict] = raw[ANNOTATIONS]
        self.categories: List[Dict] = raw[CATEGORIES]
        self._id_img_dict = None
        self._img_name_id_dict = None
        self._id_img_name_dict = None
        self._id_anns_dict = None

    @staticmethod
    def create(images, annotations, categories) -> 'COCO':
        return COCO(obj={
            IMAGES: images,
            ANNOTATIONS: annotations,
            CATEGORIES: categories,
        })

    @staticmethod
    def from_image_dir(image_dir, cls_list=('object',), suffix='jpg'):
        assert os.path.isdir(image_dir)
        image_names = sorted(i for i in os.listdir(image_dir) if i.endswith(suffix))
        images = [{IMG_FILENAME: name, 'id': i} for i, name in enumerate(image_names)]
        cats = [{'name': name, 'id': i} for i, name in enumerate(cls_list, start=1)]
        return COCO.create(images, [], cats)

    @staticmethod
    def from_detect_file(detect_file_name: str, gt: str = None, th: float = 0.5) -> 'COCO':
        """
        convert detection result file into coco format
        detection result format:
            [{
             "bbox": [
              1842.02978515625,
              434.2183532714844,
              76.0950927734375,
              148.32260131835938
             ],
             "score": 0.0180501826107502,
             "category_id": 1,
             "image_id": "FormatFactoryPart1.mp4_20200922_210922.636.jpg"
            }, ...]
        :param detect_file_name: the filename of detection result file
        :param gt: gt coco file, required while labeling_file_name is not coco format or
            image names in labeling_file aren't image file name
        :param th: score threshold, boxes which one's score lower than th are discard.
        :return: COCO result
        """
        gt_coco: COCO = None
        if gt:
            gt_coco = COCO(gt)
        try:
            with open(detect_file_name, "r") as f:
                detect_data = json.load(f)
        except json.decoder.JSONDecodeError:
            assert gt, 'gt required'
            new_dt = make_dt_by_gt(detect_file_name, gt_coco)
            with open(new_dt, "r") as f:
                detect_data = json.load(f)
        detect_data = list(filter(lambda i: i['score'] > th, detect_data))
        for i, data in enumerate(detect_data):
            data['id'] = i
        image_ids = get_set(detect_data, ANN_IMG_ID)
        cat_ids = get_set(detect_data, ANN_CAT_ID)
        if isinstance(detect_data[0][ANN_IMG_ID], str):
            images = [{'id': i, IMG_FILENAME: i} for i in image_ids]
        else:
            gt_coco = COCO(gt)
            images = [{'id': i, IMG_FILENAME: gt_coco.id_imgname_dict()[i]} for i in image_ids]
        cat = copy.deepcopy(gt_coco.cats) if gt else [{'id': i, 'name': str(i)} for i in cat_ids]
        return COCO(obj={
            IMAGES: images,
            ANNOTATIONS: detect_data,
            CATEGORIES: cat,
        })

    @staticmethod
    def from_label_file(labeling_file_name, image_dir, categories_list):
        assert image_dir, f"none img_path: {image_dir}"
        with open(labeling_file_name, "r") as f:
            labeling_data = json.load(f)

        labels = {}
        if isinstance(categories_list, list) and isinstance(categories_list[0], dict) \
                and 'id' in categories_list[0] and 'name' in categories_list[0]:
            labels = {i['id']: i['name'] for i in categories_list}
        else:
            for index, c in enumerate(categories_list, start=1):
                labels[index] = c

        image_name_id_dict = {}
        annotations = []
        g_id = 1
        for imgname in labeling_data:
            img_level_data = labeling_data[imgname]
            imgname = os.path.basename(imgname)
            if not img_level_data['data']:
                continue
            if imgname in image_name_id_dict:
                iid = image_name_id_dict[imgname]
            else:
                iid = len(image_name_id_dict) + 1
                image_name_id_dict[imgname] = iid
            for obj in img_level_data["data"]:
                bbox = [int(obj["bbox"][0] + 0.5), int(obj["bbox"][1] + 0.5), int(obj["bbox"][2] + 0.5),
                        int(obj["bbox"][3] + 0.5)]
                c_id = -1
                values = obj.get("values", {})
                for cur_id in labels:
                    if obj["type"] == labels[cur_id]:
                        c_id = cur_id
                        break
                assert c_id != -1, f'c_id error, {obj["type"]} not in {labels}'
                annotations.append({
                    "image_id": iid,
                    "bbox": bbox,
                    "segmentation": [],
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                    "id": g_id,
                    "category_id": c_id,
                    "values": values,
                })
                g_id += 1

        def get_ch(file_name):
            channel = re.search('(ch[0-9]{5})', file_name)
            return channel.group(0) if channel else file_name

        imgfiles = image_name_id_dict.keys()
        img_size_dict = dict()
        for img in imgfiles:
            ch = get_ch(img)
            if ch not in img_size_dict:
                img_size_dict[ch] = cv2.imread(os.path.join(image_dir, img)).shape[:2]

        images = []
        for imgname in image_name_id_dict.keys():
            height, width = img_size_dict[get_ch(imgname)]
            tmp = {IMG_FILENAME: imgname,
                   "id": image_name_id_dict[imgname],
                   "height": height,
                   "width": width}
            images.append(tmp)

        categories = []
        for nu in labels:
            categories.append({"id": nu, "name": labels[nu]})
        gt_data = {IMAGES: images,
                   ANNOTATIONS: annotations,
                   CATEGORIES: categories}
        return COCO(obj=gt_data)

    def get_img_by_id(self, image_id):
        if not getattr(self, '_id_img_dict', None):
            self._id_img_dict = {i['id']: i for i in self.images}
        return self._id_img_dict[image_id]

    def imgname_id_dict(self, cache=True):
        if not getattr(self, '_img_name_id_dict', None) or not cache:
            self._img_name_id_dict = OrderedDict((i[IMG_FILENAME], i['id']) for i in self.images)
        return self._img_name_id_dict

    def id_imgname_dict(self, cache=True):
        if not getattr(self, '_id_img_name_dict', None) or not cache:
            self._id_img_name_dict = OrderedDict((i['id'], i[IMG_FILENAME]) for i in self.images)
        return self._id_img_name_dict

    def id_anns_dict(self, cache=True):
        if not getattr(self, '_id_anns_dict', None) or not cache:
            self._id_anns_dict = OrderedDict((i['id'], []) for i in self.images)
            for ann in self.anns:
                self._id_anns_dict[ann[ANN_IMG_ID]].append(ann)
        return self._id_anns_dict

    def get_anns_by_image_id(self, image_id, cache=True):
        return self.id_anns_dict(cache=cache)[image_id]

    def head(self, n=50, rand=False):
        self.split(n, rand=rand, return_new_part=False)
        return self

    def stat(self):
        return len(self.imgs), len(self.anns), len(self.cats)

    def print_stat(self):
        print(self.stat())
        return self

    def to_json(self, out_file=None, indent=None):
        out_str = json.dumps({
            IMAGES: self.images,
            ANNOTATIONS: self.annotations,
            CATEGORIES: self.categories,
        }, indent=indent)
        if out_file:
            if isinstance(out_file, str):
                with open(out_file, 'w') as of:
                    of.write(out_str)
            else:
                out_file.write(out_str)
            return self
        return out_str

    def __str__(self):
        return self.to_json().encode().decode('unicode_escape')  # for print Chinese

    def __repr__(self):
        return f"COCO(json_str={self.to_json()})"

    def print(self):
        print(self)
        return self

    @property
    def imgs(self) -> List[Dict]:
        return self.images

    @imgs.setter
    def imgs(self, new_imgs):
        self.imgs = new_imgs

    @property
    def anns(self) -> List[Dict]:
        return self.annotations

    @anns.setter
    def anns(self, new_ann):
        self.annotations = new_ann

    @property
    def cats(self) -> List[Dict]:
        return self.categories

    @cats.setter
    def cats(self, new_cat) -> List[Dict]:
        self.categories = new_cat

    def length(self) -> int:
        return len(self.images)

    def crop_boxes(self, image_dir='train', min_border=0, max_border=0, out_image_dir='crop_imgs'):
        if not os.path.exists(out_image_dir):
            os.makedirs(out_image_dir)
        id_name_dict = self.id_imgname_dict()
        new_imgs = []
        new_anns = []

        for ann_id, ann in enumerate(self.annotations):
            new_ann = copy.copy(ann)
            new_ann[ANN_IMG_ID] = ann_id
            old_img_name = os.path.join(image_dir, id_name_dict[ann[ANN_IMG_ID]])
            assert os.path.exists(old_img_name), f'{old_img_name} not exist'
            old_img_data = cv2.imread(old_img_name)
            box = list(map(int, ann['bbox']))
            xbias, ybias, x2bias, y2bias = (random.randint(min_border, max_border) for _ in range(4))
            img_rows, img_cols = old_img_data.shape
            crop_box = limit_box_in_image([
                box[0] - xbias,
                box[1] - ybias,
                box[2] + x2bias + xbias,
                box[3] + y2bias + ybias, ],
                cols=img_cols,
                rows=img_rows,
            )
            if crop_box[2] <= 0 or crop_box[3] <= 0:
                continue
            new_box = limit_box_in_image([xbias, ybias, box[2], box[3]],
                                         cols=crop_box[2], rows=crop_box[3])
            new_ann['old_bbox'] = box
            new_ann['bbox'] = new_box
            new_img_data = cv2.imread(old_img_name)[get_box(crop_box)]
            new_img_name = os.path.join(out_image_dir, id_name_dict[ann[ANN_IMG_ID]] + '.{}.jpg'.format(ann_id))
            cv2.imwrite(new_img_name, new_img_data)
            new_img = {
                IMG_FILENAME: os.path.basename(new_img_name),
                'width': crop_box[2],
                'height': crop_box[3],
                'id': ann_id
            }
            new_anns.append(new_ann)
            new_imgs.append(new_img)
        self.images = new_imgs
        self.annotations = new_anns
        return self

    def visualize(self, img_dir, out_img_dir='vis', skip_no_image=True):
        colors = ((255, 0, 255), (255, 255, 0), (0, 255, 255),
                  (255, 127, 127), (127, 127, 255), (127, 255, 127),
                  (255, 0, 0), (0, 255, 0), (0, 0, 255))
        ann_dict = group_by(self.annotations, ANN_IMG_ID)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        for image in self.images:
            img_name = os.path.join(img_dir, image[IMG_FILENAME])
            if skip_no_image:
                if not os.path.exists(img_name):
                    continue
            else:
                assert os.path.exists(img_name)
            out_img_name = os.path.join(out_img_dir, image[IMG_FILENAME])
            img = cv2.imread(img_name)
            count = 0

            # add legend
            x, y = 20, 20
            for cat in self.cats:
                name = cat.get('name', cat['id'])
                color = colors[cat['id'] % len(colors)]
                img = cv2.rectangle(img, (x, y), (x + 50, y + 50), color, -1)
                cv2.putText(img, name, (x + 70, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                y += 70

            for ann in ann_dict[image['id']]:
                color = colors[ann[ANN_CAT_ID] % len(colors)]
                box = limit_box_in_image(ann['bbox'])
                img = cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                    color, 3, cv2.LINE_AA)
                if 'score' in ann:
                    img = cv2.putText(img, f'{ann["score"]:.3f}', (box[0], box[1] - 2),
                                      cv2.FONT_HERSHEY_TRIPLEX, 1, color)
                count += 1
            if count != 0:
                cv2.imwrite(out_img_name, img)
        return self

    def keep(self, min_width=0, min_height=0, min_area=0):
        self.annotations = [i for i in self.annotations
                            if i['bbox'][2] > min_width
                            and i['bbox'][3] > min_height
                            and i['bbox'][2] * i['bbox'][3] > min_area]
        return self

    def remap_image_id(self, id_dict: Dict) -> 'COCO':
        mapping(self.images, id_dict)
        mapping(self.anns, id_dict, ANN_IMG_ID)
        return self

    def merge(self, coco_obj: 'COCO', same_cats=False, strict=True) -> 'COCO':
        """combine two coco dataset

        Args:
            coco_obj (COCO):
            same_cats (bool, optional): True: Use the old categories, False: Two categories are all different. Defaults to False.
            :param coco_obj: the other coco dataset
            :param same_cats: True: Use the same categories, False: Two categories are all different. Defaults to False.
            :param strict: check coco format before merge

        Returns:
            self
        """
        if strict:
            self.health_check()
            coco_obj.health_check()
        if same_cats:
            assert len(self.cats) == len(coco_obj.cats), \
                f'cats size {len(self.cats)} vs {len(coco_obj.cats)} don\'t match'
            print(
                f'make sure class in same order: {list(zip(get_list(self.cats, "name"), get_list(coco_obj.cats, "name")))}')
        else:
            cat_dict = group_by(self.cats)
            for cat in coco_obj.cats:
                if cat['id'] in cat_dict:
                    assert cat == cat_dict[cat['id']][0], f'Categories conflict, {cat} vs {cat_dict[cat["id"]][0]}'
                else:
                    self.cats.append(cat)
        self.to_str_id()
        coco_obj.to_str_id()
        self.anns += coco_obj.anns
        self.imgs += coco_obj.imgs
        self.remove_dump_imgs()
        self.to_num_id()
        self.health_check()
        return self

    def remap_cls_label(self, name_id_pairs):
        if isinstance(name_id_pairs, (list, tuple)):
            name_id_pairs = dict(name_id_pairs)
        label_mapping = {}
        for cat in self.categories:
            if name_id_pairs.get(cat['name'], cat['id']) != cat['id']:
                old_cat_id = cat['id']
                new_cat_id = name_id_pairs[cat['name']]
                cat['id'] = new_cat_id
                label_mapping[old_cat_id] = new_cat_id
            else:
                label_mapping[cat['id']] = cat['id']
        for ann in self.annotations:
            assert ann[ANN_CAT_ID] in label_mapping, f'{ann[ANN_CAT_ID]} not in {label_mapping}'
            if ann[ANN_CAT_ID] in label_mapping:
                ann[ANN_CAT_ID] = label_mapping[ann[ANN_CAT_ID]]
        return self

    def split(self, locate: Union[int, float], rand: bool = True, return_new_part: bool = True) -> 'COCO':
        """
        split a COCO into self [0, locate) and returned [locate, end)
        :param return_new_part:
        :param locate:
        :param rand: if random split or not
        :return:
        """
        if isinstance(locate, float):
            assert 1 > locate > 0, f'float locate {locate} should in (0, 1)'
            locate = int(len(self) * locate)
        all_id = get_list(self.imgs)
        if rand:
            random.shuffle(all_id)
        dst_ids = all_id[locate:]
        if return_new_part:
            self_ids = all_id[:locate]
            dst_coco = copy.deepcopy(self).remove_imgs(ids=self_ids)
            self.remove_imgs(ids=dst_ids)
            return dst_coco
        else:
            self.remove_imgs(ids=dst_ids)
            return self

    def split_dataset(self,
                      front: str = 'instances_train.json',
                      tail: str = 'instances_val.json',
                      front_num: float = 0.8,
                      indent=None,
                      image_dir=None):
        val_part = self.split(locate=front_num).to_json(tail, indent=indent)
        self.to_json(front, indent=indent)
        val_part.imgname_id_dict(cache=False)
        self.imgname_id_dict(cache=False)
        if image_dir:
            def copy_images(dataset: 'COCO', dst_dir: str):
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir)
                for img_name in dataset.imgname_id_dict():
                    img_src_path = os.path.join(image_dir, img_name)
                    img_dst_path = os.path.join(dst_dir, img_name)
                    shutil.copyfile(img_src_path, img_dst_path, follow_symlinks=False)

            val_dir = os.path.join(image_dir, '..', 'val')
            train_dir = os.path.join(image_dir, '..', 'train')
            copy_images(val_part, val_dir)
            copy_images(self, train_dir)
        return self

    def filter(self, images_rule=None, annotations_rule=None, categories_rule=None, keep_images=True):
        pre_stat = self.stat()
        if images_rule:
            self.images = list(filter(images_rule, self.images))
        if annotations_rule:
            self.annotations = list(filter(annotations_rule, self.annotations))
        if categories_rule:
            self.categories = list(filter(categories_rule, self.categories))
        if self.stat() == pre_stat:
            return self
        kept_img_ids = get_set(self.imgs) & get_set(self.anns, ANN_IMG_ID)
        return self.filter(
            images_rule=(lambda x: x['id'] in kept_img_ids) if not keep_images else None,
            annotations_rule=lambda x: x[ANN_IMG_ID] in kept_img_ids and x[ANN_CAT_ID] in get_set(self.categories))

    def filter_imgs(self, vals, key='id'):
        vals = promise_set(vals)
        return self.filter(images_rule=lambda x: x[key] in vals)

    def filter_anns(self, vals, key='id', keep_images=True):
        vals = promise_set(vals)
        return self.filter(annotations_rule=lambda x: x[key] in vals, keep_images=keep_images)

    def filter_cls(self, vals, key='id', keep_images=True):
        vals = promise_set(vals)
        return self.filter(categories_rule=lambda x: x[key] in vals, keep_images=keep_images)

    def remove_imgs(self,
                    names: Union[str, Iterable[str]] = (),
                    ids: Union[str, Iterable[str]] = ()) -> 'COCO':
        if isinstance(names, str):
            names = [names]
        if isinstance(ids, str):
            ids = [ids]
        removed_img_id = set(ids)
        if names:
            file_names = self.imgname_id_dict(cache=False)
            for name in names:
                for file_name in file_names:
                    if fnmatch.fnmatch(file_name, f'*{name}'):
                        removed_img_id.add(file_names[file_name])
        self.filter(images_rule=lambda x: x['id'] not in removed_img_id)
        return self

    def remove_dump_imgs(self):
        # clean images
        mem = OrderedDict()
        count = 0
        for i in self.images:
            if i['id'] not in mem:
                mem[i['id']] = i
            else:
                if mem[i['id']] != i:
                    raise Exception('same id but different image: %s %s', mem[i['id']], i)
                else:
                    count += 1
        if count > 0:
            print('remove dump images {}'.format(count))
        self.images = list(mem.values())
        return self

    def to_num_id(self) -> 'COCO':
        self.images.sort(key=itemgetter(IMG_FILENAME))
        image_id_mapping = {i['id']: loc for loc, i in enumerate(self.images)}
        self.remap_image_id(image_id_mapping)
        remap_num(self.anns, order_by=ANN_IMG_ID)
        return self

    def to_str_id(self) -> 'COCO':
        image_id_mapping = {i['id']: i[IMG_FILENAME] for i in self.images}
        self.remap_image_id(image_id_mapping)
        return self

    def mosaic(self, img_dir: str, out_dir: str, cats: Union[set, int] = 1,
               const_boxes: List[List[int]] = ()) -> 'COCO':
        if isinstance(cats, (int, str)):
            cats = {cats}
        self.filter_cls(cats)
        anns_dict = group_by(self.anns, ANN_IMG_ID)
        for img_obj in tqdm(self.imgs):
            anns = anns_dict[img_obj['id']]
            img_name = os.path.join(img_dir, img_obj[IMG_FILENAME])
            out_img_name = os.path.join(out_dir, img_obj[IMG_FILENAME])
            assert os.path.exists(img_name), f'image not exist: {img_name}'
            img = cv2.imread(img_name)
            blur = cv2.GaussianBlur(img, (19, 19), 0)
            for ann in anns:
                img[get_box(ann['bbox'])] = blur[get_box(ann['bbox'])]
            for const_box in const_boxes:
                img[get_box(const_box)] = blur[get_box(const_box)]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(out_img_name, img)
        return self

    def health_check(self, img_dir=None, remove_no_image_ann=False):
        exc_data = [i for i in Counter(i['id'] for i in self.images).items() if i[1] > 1]
        if len(exc_data) > 0:
            raise Exception(f'images id not uniq:\n{exc_data}')

        exc_data = [i for i in Counter(i['id'] for i in self.annotations).items() if i[1] > 1]
        if len(exc_data) > 0:
            raise Exception(f'anns id not uniq:\n{exc_data}')

        exc_data = [i for i in Counter(i['id'] for i in self.categories).items() if i[1] > 1]
        if len(exc_data) > 0:
            raise Exception(f'cats id not uniq:\n{exc_data}')

        img_ids = get_set(self.images)
        for i in self.anns:
            if i[ANN_IMG_ID] not in img_ids:
                raise Exception(f'not found image id {i[ANN_IMG_ID]} in {i}')

        cat_ids = get_set(self.cats)
        for i in self.anns:
            if i[ANN_CAT_ID] not in cat_ids:
                raise Exception(f'not found cat id {i["id"]} in {i}')

        if img_dir:
            no_img_set = {i for i in get_iter(self.images, IMG_FILENAME) if
                          not os.path.exists(os.path.join(img_dir, i))}
            if no_img_set:
                if not remove_no_image_ann:
                    raise Exception(f'no image: {no_img_set}')
                self.remove_imgs(names=no_img_set)
                print(f'remove {len(no_img_set)} images with no image')
        return self

    def evaluate(self, dt: str, cls: Optional[Union[dict, set]] = None) -> 'COCO':
        from .eval import Evaluator
        dt = make_dt_by_gt(dt, self)
        tmp_file_dt = tempfile.NamedTemporaryFile(mode='w+')
        tmp_file = tempfile.NamedTemporaryFile(mode='w+')
        logger.debug(f'tmp_file_dt = {tmp_file_dt.name}')
        logger.debug(f'tmp_file = {tmp_file.name}')
        if not isinstance(cls, dict):
            cls = promise_set(cls)
        if cls:
            with open(dt) as dt_file:
                dt_data = json.load(dt_file)
            dt_data = filter_with(dt_data, cls, ANN_CAT_ID)
            if isinstance(cls, dict):
                for i in dt_data:
                    i[ANN_CAT_ID] = cls[i[ANN_CAT_ID]]
            json.dump(dt_data, tmp_file_dt)
            tmp_file_dt.flush()
            dt = tmp_file_dt.name
        self.to_json(tmp_file)
        evaluator = Evaluator(tmp_file.name)
        evaluator.evaluate(dt)
        return self


_intro_str = f'''
    visualize coco data:
        COCO('instances_train_shoes.json').visualize('../train', 'out_dir')
    filter data:
        COCO('instances_train_shoes.json').head(1)
        COCO('instances_train_shoes.json').keep(min_area=300)
    filter classes:
        COCO('Objects365/Annotations/train/train.json').filter_class({{2, 18, 29, 43, 54, 61, 167}})
    output:
        COCO('instances_train.json').print()
        COCO('instances_train.json').to_json('out.json')
    health check:
        COCO('instances_train.json').health_check('../train/')
    dir(COCO): {dir(COCO)}" '''


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='*', help='input files')
    parser.add_argument('-d', '--debug', action='store_true', help='turn on debug mode')
    parser.add_argument('-c', '--command', action='store_true', help='commands in python')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate coco result, file1: gt; file2: dt')
    parser.add_argument('-m', '--merge', action='store_true', help='merge all inputs, file1: coco; file2: coco, ...')
    parser.add_argument('-v', '--visualize', action='store_true', help='visualize inputs, file1: coco; file2: img dir')
    parser.add_argument('-p', '--print_stat', action='store_true',
                        help='visualize coco stats(img len, ann len, cat len), file1: coco')
    #parser.add_argument('--help', action='store_true', help='print help')
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
            COCO(args.inputs[0]).visualize(img_dir=args.inputs[1])
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


if __name__ == '__main__.py':
    main()
