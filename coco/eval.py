import numpy as np
import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Evaluator(object):

    def __init__(self, anno_file):
        self.anno_file = anno_file
        self._COCO = COCO(self.anno_file)
        self._cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in self._cats])

    def _print_detection_eval_metrics(self, coco_eval):

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

        """
        precision has dims (iou, recall, cls, area range, max dets)
        area range index 0: all area ranges
        max dets index 2: 100 per image
        """
        # Print mAP per category
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        # ap_default = np.mean(precision[precision > -1])
        print('~~~~ Per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        # print '{:.1f}'.format(100 * ap_default)
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{}: {:.1f}'.format(cls, 100 * ap))

        # Print AP50 per category
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_lo + 1), :, :, 0, 2]
        # ap_default = np.mean(precision[precision > -1])
        print('~~~~ Per-category AP @ IoU=[{:.2f}] ~~~~'.format(IoU_lo_thresh))
        # print '{:.1f}'.format(100 * ap_default)
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_lo + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{}: {:.1f}'.format(cls, 100 * ap))

        # Summary Table
        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def evaluate(self, res_file):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dt', type=str, help='JSON format detection-results file.', default='bbox.json')
    parser.add_argument('-g', '--gt', type=str, help='JSON format annotation file.',)

    args = parser.parse_args()

    evaluator = Evaluator(args.anno)
    evaluator.evaluate(args.res)
