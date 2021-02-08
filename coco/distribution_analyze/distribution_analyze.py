import logging
import os
import re
from collections import defaultdict
from typing import Set, Optional, Sequence, Dict

from .. import COCO
from ..consts import IMG_FILENAME
from ..utils import grep_set


def parse_channel_from_img_name(img_name: str) -> Optional[Sequence[str]]:
    """
    return (store info, channel) from an image name
    :param img_name: any image name
    :return: (store info, channel) pair if success, or None if failed
    """
    regexp = re.compile('.*?([A-Z]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+|[A-Za-z0-9]+)_.*(ch[0-9]{5}).*')
    match_result = regexp.match(img_name)
    if not match_result or len(match_result.groups()) != 2:
        logging.warning(f'channel cannot parsed: {img_name}')
        return None
    return match_result.groups()


def update_channels(coco: COCO) -> COCO:
    if 'channel' not in coco.imgs[0]:
        for img in coco.imgs:
            channel = parse_channel_from_img_name(img[IMG_FILENAME])
            if channel is not None:
                img['channel'] = '-'.join(list(channel))
            else:
                img['channel'] = 'UNKNOWN'
    return coco


def get_all_channels(coco: COCO) -> Set[str]:
    update_channels(coco)
    return grep_set(coco.imgs, 'channel')


def calc_mean_width_height(coco: COCO, ignore_channel=False) -> Dict[str, float]:
    update_channels(coco)
    mean_width = defaultdict(list)
    mean_height = defaultdict(list)
    mean_w_h_per_channel = {}
    for img in coco.imgs:
        img_w = img['width']
        img_h = img['height']
        for ann in coco.get_anns_by_image_id(img['id']):
            mean_width[img['channel']].append(ann['bbox'][2] / img_w)
            mean_height[img['channel']].append(ann['bbox'][3] / img_h)
    for channel in mean_width:
        mean_w_h_per_channel[channel if not ignore_channel else 'ALL'] = \
            (sum(mean_width[channel]) / len(mean_width[channel]),
             sum(mean_height[channel]) / len(mean_height[channel]))
    return mean_w_h_per_channel


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('-c', '--coco')
    parser.add_argument('-t', '--threshold', default=0.03, type=float)
    args = parser.parse_args()
    ref = COCO(args.reference).health_check()
    ref_w, ref_h = calc_mean_width_height(ref, ignore_channel=True)['ALL']
    print(f'distribution for {args.reference}: {ref_w}, {ref_h}, ({ref_w * 1024}, {ref_h * 576}) in (1024x576 image)')
    if args.coco:
        coco = COCO(args.coco)
        mean_wh = calc_mean_width_height(coco)
        accepted_channels = set()
        for channel, (w, h) in mean_wh.items():
            threshold = args.threshold
            if abs(w - ref_w) < threshold \
                    and abs(h - ref_h) < threshold \
                    and abs(w - ref_w) + abs(h - ref_h) < threshold * 1.5:
                print(f'{channel} accepted, {mean_wh[channel]}')
                accepted_channels.add(channel)
        print(accepted_channels)
        coco.print_stat()
        coco.filter_imgs(accepted_channels, 'channel').print_stat().to_json(args.coco + '.dfid.json')


if __name__ == '__main__.py':
    main()
