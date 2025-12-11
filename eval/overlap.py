import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import torch
from torchmetrics import Metric


def iou(interval_1, interval_2):
    """
    interval: list (2 float elements)
    """
    eps = 1e-8
    (s_1, e_1) = interval_1
    (s_2, e_2) = interval_2

    intersection = max(0.0, min(e_1, e_2) - max(s_1, s_2))
    union = min(max(e_1, e_2) - min(s_1, s_2), e_1 - s_1 + e_2 - s_2)
    iou = intersection / (union + eps)
    return iou


def get_vid_overlaps(refs, preds):
    """
    refs are the video ground truths
    preds are the video predictions
    """
    vid_overlaps = {}
    for ref_i, ref in enumerate(refs):
        for pred_j, pred in enumerate(preds):
            iou_ij = iou(ref, pred)
            if iou_ij > 0:
                vid_overlaps[(ref_i, pred_j)] = iou_ij
    return vid_overlaps

def vid_overlap_optimal_assignment(vid_overlaps):
    """
    vid_overlaps is a dictionary of video overlaps
    """

    # Initialize sets to keep track of covered references and predictions
    ref_set_covered = set()
    pred_set_covered = set()

    # Sort the IoUs in descending order based on their values
    sorted_vid_overlaps = sorted(vid_overlaps.items(), key=lambda x: x[1], reverse=True)
    connections = {}
    # Iterate through the sorted IoUs
    for (ref_i, pred_j), tiou_ij in sorted_vid_overlaps:
        # Check if the reference or prediction is already covered
        if ref_i not in ref_set_covered and pred_j not in pred_set_covered:
            # If not covered, mark them as covered
            ref_set_covered.add(ref_i)
            pred_set_covered.add(pred_j)
            # Output the chosen pair and its IoU

            # Calculate midpoints of predictions and references
            connections[(ref_i, pred_j)] = tiou_ij

    return connections

def compute_vid_avg_optimal(refs, preds):
    if not refs or not preds:
        return 0.0
    vid_overlaps = get_vid_overlaps(refs, preds)
    if not vid_overlaps:
        return 0.0
    vid_overlaps = vid_overlap_optimal_assignment(vid_overlaps)
    return np.mean(list(vid_overlaps.values()))


def vid_overlap_threshold_assignment(vid_overlaps, threshold):
    connections = {
        k: overlap for k, overlap in vid_overlaps.items() if overlap >= threshold
    }
    return connections

    

class OverlapMetrics:
    def __init__(self, pred, ref, verbose=False):

        self.pred = pred
        self.ref = ref
        self.verbose = verbose

        pred_keys = set(pred.keys())
        ref_keys = set(ref.keys())

        missing_in_pred = ref_keys - pred_keys # empty, good
        missing_in_ref = pred_keys - ref_keys # empty, good

        # self.vid_ids = sorted(list(pred_keys))
        self.vid_ids = sorted([x for x in pred_keys if x in ref_keys])
        

        self.process_pred()
        self.process_ref()

        # print(self.pred[self.vid_ids[0]])
        # print(self.ref[self.vid_ids[0]])


    def process_pred(self):
        for vid_id in self.vid_ids:
            pred = self.pred[vid_id]
            if type(pred) == dict:
                pred = pred["timestamps"]
            elif type(pred) == list:
                tmp = []
                for x in pred:
                    if type(x) == dict:
                        tmp.append(x["timestamp"])
                pred = tmp
            self.pred[vid_id] = pred

    def process_ref(self):
        for vid_id in self.vid_ids:
            ref = self.ref[vid_id]
            if type(ref) == dict:
                ref = ref["timestamps"]
            self.ref[vid_id] = ref


    def get_metrics(self, metrics=("F1", "Avg. TIoU")):
        results = {}
        
        if "F1" in metrics:
            avg_overlap_res  = self.compute_avg_thresholds()
            if "P" in metrics:
                results["P"] = avg_overlap_res["precision"]
            if "R" in metrics:
                results["R"] = avg_overlap_res["recall"]
            if "F1" in metrics:
                results["F1"] = avg_overlap_res["f1"]
        if "Avg. TIoU" in metrics:
            results["Avg. TIoU"] = self.compute_avg_optimal()
        
        print(results)

    def compute_avg_thresholds(self, tiou_thr=None):
        if tiou_thr is None:
            step = 0.05
            tiou_thr = np.arange(0.5, 0.95 + step, step)
            # tiou_thr = np.arange(0.75, 0.95 + step, step)
        thr2prf = {t: PRFMetric() for t in tiou_thr}
        for vid_id in self.vid_ids:
            vid_refs = self.ref[vid_id]
            vid_preds = self.pred[vid_id]
            vid_overlaps = get_vid_overlaps(vid_refs, vid_preds)

            for t in tiou_thr:
                vid_overlaps_t = vid_overlap_threshold_assignment(vid_overlaps, t)
                ref_set_covered = {ref_i for ref_i, _ in vid_overlaps_t}
                pred_set_covered = {pred_j for _, pred_j in vid_overlaps_t}
                vid_p = float(len(pred_set_covered)) / max(len(vid_preds), 1)
                vid_r = float(len(ref_set_covered)) / len(vid_refs)

                thr2prf[t].update(vid_p, vid_r)

        return {metric: np.mean([thr2prf[t].compute()[metric] for t in tiou_thr])
                for metric in ["precision", "recall", "f1"]}

    def compute_avg_optimal(self):
        res = []
        for vid_id in self.vid_ids:
            vid_refs = self.ref[vid_id]
            vid_preds = self.pred[vid_id]
            res.append(compute_vid_avg_optimal(vid_refs, vid_preds))
        return np.mean(res)


class PRFMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("t_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, vid_p, vid_r) -> None:
        self.t_precision += vid_p
        self.t_recall += vid_r
        self.t_f1 += 2 * (vid_p * vid_r) / (vid_p + vid_r) if vid_p + vid_r else 0.0
        self.n += 1

    def compute(self):
        avg_p = self.t_precision * 100 / self.n
        avg_r = self.t_recall * 100 / self.n
        avg_f1 = self.t_f1 * 100 / self.n
        return {"precision": avg_p, "recall": avg_r, "f1": avg_f1}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default='data/pred_prev.json')
    parser.add_argument('--ref_file', type=str, default='data/gt.json')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    args = parser.parse_args()

    with open(args.pred_file, 'r') as f:
        pred_data = json.load(f)
    pred_data = pred_data["results"]

    with open(args.ref_file, 'r') as f:
        ref_data = json.load(f)
    

    metrics = OverlapMetrics(pred_data, ref_data, verbose=args.verbose)
    metrics.get_metrics()

    # unimotion: /home/sxu/Unimotion/babel_results.json 