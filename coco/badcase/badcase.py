import copy

from .. import COCO
import numpy as np
from scipy.optimize import linear_sum_assignment


def calc_iou_xywh(box_a, box_b):
    box_a_xyxy = [box_a[0], box_a[1], box_a[2] + box_a[0], box_a[3] + box_a[1]]
    box_b_xyxy = [box_b[0], box_b[1], box_b[2] + box_b[0], box_b[3] + box_b[1]]
    return calc_iou_xyxy(box_a_xyxy, box_b_xyxy)


def calc_iou_xyxy(box_a, box_b):
    a_l, a_t, a_r, a_b = box_a[0], box_a[1], box_a[2], box_a[3]
    b_l, b_t, b_r, b_b = box_b[0], box_b[1], box_b[2], box_b[3]

    x1 = max(a_l, b_l)
    y1 = max(a_t, b_t)
    x2 = min(a_r, b_r)
    y2 = min(a_b, b_b)
    box_a_area = (a_r - a_l + 1) * (a_b - a_t + 1)
    box_b_area = (b_r - b_l + 1) * (b_b - b_t + 1)

    intersect_area = max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0)
    union_area = max(box_a_area + box_b_area - intersect_area, 1)
    iou = intersect_area / float(union_area)
    return iou

from loguru import logger
@logger.catch
def get_badcase(gt:COCO, dt:COCO, cat=1, threshold:float = 0.5)->COCO:
    """
    get badcase subset if any gt has lower iou than threshold with all dt boxes
    :param gt: gt COCO object
    :param dt: dt COCO object, filted with threshold
    :param cat: badcase category id
    :param threshold: iou threshold
    :return: badcase subset
    """
    bad_case_cat = [
        {'id': 1, 'name': 'object'},
        {'id': 2, 'name': 'wrong detect'},
        {'id': 3, 'name': 'missed gt'},
    ]
    {i['id']: i['name'] for i in gt.cats}[cat]
    gt = gt.filter_cls(cat).remap_cls_label({{i['id']: i['name'] for i in gt.cats}[cat]: 1})
    dt = dt.filter_cls(cat).remap_cls_label({{i['id']: i['name'] for i in dt.cats}[cat]: 1})
    gt.cats = bad_case_cat
    dt.cats = bad_case_cat
    badcase_image_ids = set()
    fps = []
    fns = []
    for img in gt.imgs:
        keep_gt_ann_pos = set()
        keep_dt_ann_pos = set()
        fp = 0
        fn = 0
        image_id = img['id']
        if not image_id in dt.id_anns_dict():
            continue
        gt_anns = gt.get_anns_by_image_id(image_id)
        dt_anns = dt.get_anns_by_image_id(image_id)
        iou_matrix = np.array([[-calc_iou_xywh(g['bbox'], d['bbox']) for d in dt_anns] for g in gt_anns])
        if len(gt_anns) != 0 and len(dt_anns) != 0:
            gt_result, dt_result = linear_sum_assignment(iou_matrix)
            for i, j in zip(gt_result, dt_result):
                if iou_matrix[i, j] < -threshold:
                    keep_gt_ann_pos.add(i)
                    keep_dt_ann_pos.add(j)
            fp = (len(dt_anns) - len(keep_dt_ann_pos)) / len(dt_anns)
            fn = (len(gt_anns) - len(keep_gt_ann_pos)) / len(gt_anns)
        elif len(gt_anns) == 0 and len(dt_anns) == 0:
            pass
        elif len(gt_anns) == 0 and len(dt_anns) != 0:
            fn = 1
        elif len(gt_anns) != 0 and len(dt_anns) == 0:
            fp = 1
        else:
            raise Exception('Should not run here')
        fps.append(fp)
        fns.append(fn)
        if fp > 0.1 or fn > 0.1:
            badcase_image_ids.add(image_id)
            for i, dt_ann in enumerate(dt_anns):
                if i not in keep_dt_ann_pos:
                    dt_ann['category_id'] = 2
            for i, gt_ann in enumerate(gt_anns):
                if i not in keep_gt_ann_pos:
                    gt_ann['category_id'] = 3
                    dt.anns.append(gt_ann)
    print(f'evaluate:\n\tiou_threshold: {threshold}\n\tmean fp: {sum(fps)/len(fps):.3f}\n\tmean fn: {sum(fns)/len(fns):.3f}')
    print(f'mean error: {(sum(fps)/len(fps) + sum(fns)/len(fns))/2:.3f}')
    return dt.filter_imgs(badcase_image_ids)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt')
    parser.add_argument('dt')
    parser.add_argument('-i', '--img_dir', help='img dir for vis')
    parser.add_argument('-v', '--vis_dir', default='vis', help='output dir for vis')
    parser.add_argument('-c', '--cat', default=1, type=int, help='category for analyze')
    parser.add_argument('-s', '--score_th', default=0.4, type=float)
    parser.add_argument('-o', '--iou_th', default=0.5, type=float)
    args = parser.parse_args()
    score_threshold = args.score_th
    iou_threshold = args.iou_th
    gt = COCO(args.gt)
    dt = COCO.from_detect_file(args.dt, gt=args.gt, th=score_threshold)
    print(f'eval for score threshold: {args.score_th}')
    badcase = get_badcase(gt, dt, cat=args.cat, threshold=iou_threshold)
    print(f'Bad case stat(image, box, cls): {badcase.stat()}')
    if args.img_dir:
        assert args.img_dir, f'need -i/--img_dir argument for visualize'
        badcase.vis(args.img_dir, args.vis_dir)


if __name__ == '__main__.py':
    main()
