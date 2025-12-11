# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import json
import random
import string
import sys
sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np

def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def iou(interval_1, interval_2):
    """
    interval: list (2 float elements)
    """
    eps = 1e-8  # 防止除0
    (s_1, e_1) = interval_1
    (s_2, e_2) = interval_2

    intersection = max(0.0, min(e_1, e_2) - max(s_1, s_2))
    union = min(max(e_1, e_2) - min(s_1, s_2), e_1 - s_1 + e_2 - s_2)
    iou = intersection / (union + eps)
    return iou


def build_iou_matches_vid(vid_refs, vid_preds, tiou=0.5):
    """
    Computes IoU matches for a single video's references and predictions.

    Parameters:
    - vid_refs: Dictionary of reference ground truth captions with timestamps for one video.
    - vid_preds: Dictionary of predicted captions with timestamps for one video.
    - tiou: Temporal Intersection over Union threshold.

    Returns:
    - Two dictionaries containing matched references and predictions based on tIoU.
    """
    refs_iou = {}
    preds_iou = {}

    for vid_pred_segment, vid_pred_label in vid_preds.items():
        match_found = False
        for vid_ref_segment, vid_ref_label in vid_refs.items():
            # 使用第一版的 IoU 计算方法
            iou_value = iou(vid_pred_segment, vid_ref_segment)
            if iou_value >= tiou:
                # 匹配成功，添加到字典
                refs_iou[(vid_pred_segment, vid_ref_segment)] = [vid_ref_label]
                preds_iou[(vid_pred_segment, vid_ref_segment)] = [vid_pred_label]
                match_found = True

        if not match_found:
            # 如果没有匹配，则使用随机字符串作为占位符
            refs_iou[(vid_pred_segment)] = [random_string(10)]
            preds_iou[(vid_pred_segment)] = [vid_pred_label]

    return refs_iou, preds_iou


def build_iou_matches(vid2refs, vid2preds, tiou=0.5):
    """
    Matches ground truth captions to predicted captions based on tIoU threshold.

    Parameters:
    - vid2refs: Dictionary of reference ground truth captions with timestamps.
    - vid2preds: Dictionary of predicted captions with timestamps.
    - tiou: Temporal Intersection over Union threshold.

    Returns:
    - Two dictionaries containing matched references and predictions based on tIoU.
    """
    vid2refs_iou = {}
    vid2preds_iou = {}

    for vid_id, vid_refs in vid2refs.items():
        if vid_id not in vid2preds:
            continue

        vid_preds = vid2preds[vid_id]
        # 调用改进版 IoU 匹配函数
        vid2refs_iou[vid_id], vid2preds_iou[vid_id] = build_iou_matches_vid(
            vid_refs, vid_preds, tiou
        )

    return vid2refs_iou, vid2preds_iou

class ANETcaptions(object):
    # PREDICTION_FIELDS = ['results', 'version', 'external_data']
    PREDICTION_FIELDS = ['results']
    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 tious=None, max_proposals=1000,
                 prediction_fields=PREDICTION_FIELDS, verbose=False):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.tious = tious
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        ##### ADDED for CHAP-LLaMA Version Code #####
        self.vid2refs = self.process_ref(ground_truth_filenames)
        self.vid2preds = self.process_pred(prediction_filename)

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        if self.verbose:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            # self.scorers = [(Meteor(), "METEOR")]
            self.scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print( "| Loading submission...")
        submission = json.load(open(prediction_filename))
        if not all([field in submission.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid ground truth file.')
        # Ensure that every video is limited to the correct maximum number of proposals.
        results = {}
        for vid_id in submission['results']:
            results[vid_id] = submission['results'][vid_id][:self.max_proposals]
        return results

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(gt)
        if self.verbose:
            print("| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids)))
        return gts

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def check_gt_exists(self, vid_id):
        for gt in self.ground_truths:
            if vid_id in gt:
              return True
        return False

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        aggregator = {}
        self.scores = {}
        for tiou in self.tious:
            scores = self.evaluate_tiou(tiou)
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)
        if self.verbose:
            self.scores['Recall'] = []
            self.scores['Precision'] = []
            for tiou in self.tious:
                precision, recall = self.evaluate_detection(tiou)
                self.scores['Recall'].append(recall)
                self.scores['Precision'].append(precision)

    def process_pred(self, preds_filename):
        with open(preds_filename, 'r') as f:
            preds = json.load(f)
        preds = preds["results"]

        vids = list(preds.keys())
        for vid_id in vids:
            pred = preds[vid_id]
            if type(pred) == dict:
                pred = pred["timestamps"]
            elif type(pred) == list:
                tmp = []
                for x in pred:
                    if type(x) == dict:
                        tmp.append(x["timestamp"])
                pred = tmp
            preds[vid_id] = pred
        return preds

    def process_ref(self, refs_filename):
        print("Processing reference file: ", refs_filename)
        if type(refs_filename) == list:
            refs_filename = refs_filename[0]
        with open(refs_filename, 'r') as f:
            refs = json.load(f)

        vids = list(refs.keys())
        for vid_id in vids:
            ref = refs[vid_id]
            if type(ref) == dict:
                ref = ref["timestamps"]
            refs[vid_id] = ref
        return refs
    
    def evaluate_chap(self):
        """Chapter-LLaMA version"""
        aggregator = {}
        self.scores = {}
        for tiou in self.tious:
            scores = self.evaluate_tiou_chap(tiou)
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)

        # Print detailed scores if verbose
        if self.verbose:
            for metric, scores in self.scores.items():
                avg_score = np.mean(scores)
                print(f"| {metric}: {avg_score:.2f}%")

        # Compute average across all tIoUs
        avg_scores = {
            metric: np.mean(self.scores[metric]) for metric in self.scores
        }
        print("Average across all tIoUs:", avg_scores)
        return avg_scores

    def evaluate_tiou_chap(self, tiou, scorers=("CIDEr",)):
        """
        Evaluate video captions based on a single tIoU threshold using PRF.
        Chapter-LLaMA version

        Parameters:
        - tiou: Temporal Intersection over Union threshold for matching predictions to ground truths.
        - scorers: Tuple of scoring metrics to use for this evaluation.

        Returns:
        - Dictionary with PRF scores.
        """
        # Initialize PRF metric
        prf_metric = PRFMetric()
        
        # Create dictionaries of matched references and predictions based on tIoU
        vid2refs_iou, vid2preds_iou = build_iou_matches(
            self.vid2refs, self.vid2preds, tiou
        )

        video_ids = list(vid2refs_iou.keys())
        output = {}

        for vid_id in video_ids:
            refs_iou = vid2refs_iou[vid_id]
            preds_iou = vid2preds_iou[vid_id]
            
            # Calculate precision and recall for each video
            ref_set_covered = {ref_i for ref_i, _ in refs_iou}
            pred_set_covered = {pred_j for _, pred_j in preds_iou}

            vid_p = float(len(pred_set_covered)) / max(len(preds_iou), 1)
            vid_r = float(len(ref_set_covered)) / len(refs_iou)

            # Update PRF metric
            prf_metric.update(vid_p, vid_r)

        # Compute final PRF
        prf_results = prf_metric.compute()
        for metric in ["precision", "recall", "f1"]:
            output[metric] = prf_results[metric]

        return output

    def evaluate_detection(self, tiou):
        gt_vid_ids = self.get_gt_vid_ids()
        # Recall is the percentage of ground truth that is covered by the predictions
        # Precision is the percentage of predictions that are valid
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set([])
                pred_set_covered = set([])
                num_gt = 0
                num_pred = 0
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)

                    new_precision = float(len(pred_set_covered)) / (pred_i + 1) 
                    best_precision = max(best_precision, new_precision)
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def evaluate_tiou(self, tiou):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos 
        res = {}
        gts = {}
        gt_vid_ids = self.get_gt_vid_ids()
        
        unique_index = 0

        # video id to unique caption ids mapping
        vid2capid = {}
        
        cur_res = {}
        cur_gts = {}
        
        
        for vid_id in gt_vid_ids:
            
            vid2capid[vid_id] = []

            # If the video does not have a prediction, then we give it no matches
            # We set it to empty, and use this as a sanity check later on.
            if vid_id not in self.prediction:
                pass

            # If we do have a prediction, then we find the scores based on all the
            # valid tIoU overlaps.
            else:
                # For each prediction, we look at the tIoU with ground truth.
                for pred in self.prediction[vid_id]:
                    has_added = False
                    for gt in self.ground_truths:
                        if vid_id not in gt:
                            continue
                        gt_captions = gt[vid_id]
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                            if self.iou(pred['timestamp'], caption_timestamp) >= tiou:
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{'caption': remove_nonascii(gt_captions['sentences'][caption_idx])}]
                                vid2capid[vid_id].append(unique_index)
                                unique_index += 1
                                has_added = True

                    # If the predicted caption does not overlap with any ground truth,
                    # we should compare it with garbage.
                    if not has_added:
                        cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                        cur_gts[unique_index] = [{'caption': random_string(random.randint(10, 20))}]
                        vid2capid[vid_id].append(unique_index)
                        unique_index += 1

        # Each scorer will compute across all videos and take average score
        output = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print( 'computing %s score...'%(scorer.method()))
            
            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}
            
            # call tokenizer here for all predictions and gts
            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)
            
            # reshape back
            for vid in vid2capid.keys():
                res[vid] = {index:tokenize_res[index] for index in vid2capid[vid]}
                gts[vid] = {index:tokenize_gts[index] for index in vid2capid[vid]}
            
            for vid_id in gt_vid_ids:

                if len(res[vid_id]) == 0 or len(gts[vid_id]) == 0:
                    if type(method) == list:
                        score = [0] * len(method)
                    else:
                        score = 0
                else:
                    score, scores = scorer.compute_score(gts[vid_id], res[vid_id])
                all_scores[vid_id] = score

            # print(all_scores)
            if type(method) == list:
                # scores = np.mean(all_scores.values(), axis=0)
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
                    if self.verbose:
                        print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method[m], output[method[m]]))
            else:
                output[method] = np.mean(list(all_scores.values()))
                if self.verbose:
                    print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method, output[method]))
        return output

class PRFMetric:
    def __init__(self):
        self.t_precision = 0.0
        self.t_recall = 0.0
        self.t_f1 = 0.0
        self.n = 0

    def update(self, vid_p, vid_r):
        self.t_precision += vid_p
        self.t_recall += vid_r
        self.t_f1 += (
            2 * (vid_p * vid_r) / (vid_p + vid_r) if vid_p + vid_r else 0.0
        )
        self.n += 1

    def compute(self):
        avg_p = self.t_precision * 100 / max(self.n, 1)
        avg_r = self.t_recall * 100 / max(self.n, 1)
        avg_f1 = self.t_f1 * 100 / max(self.n, 1)
        return {"precision": avg_p, "recall": avg_r, "f1": avg_f1}


def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             tious=args.tious,
                             max_proposals=args.max_proposals_per_video,
                             verbose=args.verbose)
    evaluator.evaluate()

    # chap-llama version still have some bug... we just use the original version for now
    # evaluator.evaluate_chap()

    # Output the results
    if args.verbose:
        for i, tiou in enumerate(args.tious):
            print('-' * 80)
            print("tIoU: ", tiou)
            print('-' * 80)
            for metric in evaluator.scores:
                score = evaluator.scores[metric][i]
                print('| %s: %2.4f'%(metric, 100*score))

    # Print the averages
    print('-' * 80)
    print('Average across all tIoUs')
    print('-' * 80)

    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        print('| %s: %2.4f'%(metric, 100 * np.mean(score)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, nargs='+', default=['data/val_1.json', 'data/val_2.json'],
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('--tious', type=float,  nargs='+', default=[0.3, 0.5, 0.7, 0.9],
                        help='Choose the tIoUs to average over.')
    parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=1000,
                        help='maximum propoasls per video.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print intermediate steps.')
    args = parser.parse_args()

    main(args)