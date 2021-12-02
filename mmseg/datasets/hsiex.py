# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
import mmcv
import numpy as np
from numpy.random import default_rng
from mmcv.utils import print_log
from prettytable import PrettyTable
from mmseg.core import eval_metrics
from sklearn import metrics

@DATASETS.register_module()
class HSIExtraDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('noncancer', 'cancer')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, full_positive_dir, full_negative_dir, extra_rate=1.0, **kwargs):
        super(HSIExtraDataset, self).__init__(
            img_suffix='.hdr', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        self.full_positive_dir = full_positive_dir
        self.full_negative_dir = full_negative_dir
        if self.data_root is not None:
            if not osp.isabs(self.full_positive_dir):
                self.full_positive_dir = osp.join(self.data_root, self.full_positive_dir)
            if not osp.isabs(self.full_negative_dir):
                self.full_negative_dir = osp.join(self.data_root, self.full_negative_dir)
        self.full_positive_infos, self.full_negative_infos = \
            self.load_extra_annotations(self.full_positive_dir, self.full_negative_dir, self.img_suffix)
        self.extra_rate = extra_rate
        self.len_true = len(self.img_infos)
        self.len_false = int(self.len_true * self.extra_rate)
        self.len_positive = len(self.full_positive_infos)
        self.len_negative = len(self.full_negative_infos)
        self.extra_positive_idx = self.len_positive
        self.extra_negative_idx = self.len_negative
        self.extra_ann_idx = self.len_true
        self.shuffled_positive_map = np.arange(self.len_positive)
        self.shuffled_negative_map = np.arange(self.len_negative)
        self.shuffled_ann_map = np.arange(self.len_true)
        self.rng = default_rng()

    def __len__(self):
        return self.len_false

    def load_extra_annotations(self, full_positive_dir, full_negative_dir, img_suffix):
        full_positive_infos = []
        for img in mmcv.scandir(full_positive_dir, img_suffix, recursive=True):
            full_positive_infos.append(dict(filename=img))
        full_negative_infos = []
        for img in mmcv.scandir(full_negative_dir, img_suffix, recursive=True):
            full_negative_infos.append(dict(filename=img))
        print_log(f'Loaded {len(full_positive_infos)} positive images', logger=get_root_logger())
        print_log(f'Loaded {len(full_negative_infos)} negative images', logger=get_root_logger())
        return full_positive_infos, full_negative_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['full_positive_prefix'] = self.full_positive_dir
        results['full_negative_prefix'] = self.full_negative_dir
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def prepare_train_img(self, idx):
        if idx < self.len_true:
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
        else:
            if self.extra_positive_idx == self.len_positive:
                self.rng.shuffle(self.shuffled_positive_map)
                self.extra_positive_idx = 0
            if self.extra_negative_idx == self.len_negative:
                self.rng.shuffle(self.shuffled_negative_map)
                self.extra_negative_idx = 0
            if self.extra_ann_idx == self.len_true:
                self.rng.shuffle(self.shuffled_ann_map)
                self.extra_ann_idx = 0
            img_info = dict(positive=self.full_positive_infos[self.shuffled_positive_map[self.extra_positive_idx]],
                            negative=self.full_negative_infos[self.shuffled_negative_map[self.extra_negative_idx]],
                            ann=self.get_ann_info(self.shuffled_ann_map[self.extra_ann_idx]))
            ann_info = self.get_ann_info(self.extra_ann_idx)
            self.extra_positive_idx += 1
            self.extra_negative_idx += 1
            self.extra_ann_idx += 1
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)



    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):

        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)

            # get kappa
            con_mat = np.zeros((2, 2))
            for result, gt in zip(results, self.get_gt_seg_maps()):
                con_mat += metrics.confusion_matrix(gt.flatten(), result.flatten(), labels=[1, 0])

        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

            # get kappa
            con_mat = np.zeros((2, 2))
            pre_eval_results = tuple(zip(*results))

            total_area_intersect = sum(pre_eval_results[0])
            total_area_label = sum(pre_eval_results[3])
            con_mat[0][0] = total_area_intersect[0]
            con_mat[1][1] = total_area_intersect[1]
            con_mat[0][1] = total_area_label[1] - total_area_intersect[1]
            con_mat[1][0] = total_area_label[0] - total_area_intersect[0]

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        print_log('mIoU:{:.4f}'.format(eval_results['mIoU']), logger=logger)
        print_log('mDice:{:.4f}'.format(eval_results['mDice']), logger=logger)
        print_log('mAcc:{:.4f}'.format(eval_results['mAcc']), logger=logger)
        print_log('aAcc:{:.4f}'.format(eval_results['aAcc']), logger=logger)
        print_log('kappa:{:.4f}'.format(kappa(con_mat)), logger=logger)
        print_log('accuracy:{:.4f}'.format(accuracy(con_mat)), logger=logger)
        # print_log('precision:{:.4f}'.format(precision(con_mat)), logger=logger)
        # print_log('sensitivity:{:.4f}'.format(sensitivity(con_mat)), logger=logger)
        # print_log('specificity:{:.4f}'.format(specificity(con_mat)), logger=logger)

        return eval_results


def kappa(matrix):
    matrix = np.array(matrix)
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)
def sensitivity(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[1][0])
def specificity(matrix):
    return matrix[1][1]/(matrix[1][1]+matrix[0][1])
def precision(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[0][1])
def accuracy(matrix):
    return (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])