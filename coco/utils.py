from collections import defaultdict
from typing import Iterable, List, Dict, Set

import numpy as np

from coco.consts import MAX_INT


def promise_set(in_val=None) -> set:
    """
    try to convert anything to a set
    :param in_val: input data
    :return: result set from in_val
    """
    if in_val is None:
        return set()
    if not isinstance(in_val, str) and isinstance(in_val, Iterable):
        return set(in_val)
    return {in_val}


def get_slice(bbox):
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


def grep_set(data: Iterable[Dict], label: str = 'id') -> Set[str]:
    """
    get the unique set from the attribute **label** in data
    :param data:
    :param label:
    :return: a set with all **label** in **data**
    """
    return set([i[label] for i in data])


def grep_list(data: Iterable[Dict], label: str = 'id') -> List[str]:
    """
    get the ordered list from the attribute **label** in data
    :param data:
    :param label:
    :return: a list with all **label** in **data**
    """
    return [i[label] for i in data]


def grep_iter(data: Iterable[Dict], label: str = 'id') -> Iterable[str]:
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
    Change the selected field to integer sequence inplace for a List[dict] data type
    :param data: input list, all the child elements should be dict
    :param label: selected field, default to 'id'
    :param start: start number of the sequence, default to 0
    :param order_by: the sort key field before remapping
    :return: remapped list
    """
    if order_by:
        data.sort(key=lambda x: x[order_by])
    for loc, i in enumerate(data, start=start):
        i[label] = loc
    return data
