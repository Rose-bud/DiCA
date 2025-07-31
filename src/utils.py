# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os

import numpy as np
import torch

from .logger import create_logger, PD_Stats

import torch.distributed as dist
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.config import args

import nets.ImageNet as ImageNet

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path, map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def confidence_update(confidence, logits, plLabel, indexes):
    with torch.no_grad():
        logits = F.softmax(logits, dim=1)
        confidence[indexes, :] = plLabel * logits
        base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
        confidence = confidence / base_value
    return confidence

def compute_pda_loss(feats_q, prototypes_img, prototypes_txt):
    siml_q_img = torch.mm(feats_q, prototypes_img.t())
    siml_q_txt = torch.mm(feats_q, prototypes_txt.t())
    dist_siml_q = (siml_q_img - siml_q_txt).abs().sum(1).mean()

    return dist_siml_q

def set_prototype_update_weight(epoch, args):
    start = float(args.pro_weight_start)
    end = float(args.pro_weight_end)
    proto_weight = 1. * epoch / args.max_epochs * (end - start) + start

    return  proto_weight

def nbd_loss(outputs, partialY,confidence):
    partialY = torch.tensor(partialY)

    sm_outputs = F.softmax(outputs, dim=1).cuda()
    final_outputs = sm_outputs.to('cuda') *partialY.to('cuda') * confidence.to('cuda')
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()

    loss_noncon = torch.sum((1. - partialY.to('cuda')) * sm_outputs.to('cuda')**2)

    return average_loss + args.lamda * loss_noncon

def supervised_contrastive_loss(fea, labels_list, tau=1.):
    n_view = len(fea)
    batch_size = fea[0].shape[0]
    image_labels1, text_labels1 = labels_list
    image_labels, text_labels = torch.argmax(image_labels1, dim=1), torch.argmax(text_labels1, dim=1)
    labels = [image_labels, text_labels]
    all_fea = torch.cat(fea)
    all_labels = torch.cat(labels)
    sim = all_fea.mm(all_fea.t())
    sim = (sim / tau).exp()
    label_sim = all_labels.unsqueeze(1) == all_labels.unsqueeze(0)
    label_sim = label_sim.float()
    sim = sim * label_sim
    sim = sim - sim.diag().diag()
    eps = 1e-20
    sim = sim + eps
    log_sim = torch.log(sim)

    sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
    diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
    loss1 = -(diag1 / sim.sum(1)).log().mean()

    sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
    diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
    loss2 = -(diag2 / sim.sum(1)).log().mean()

    return loss1 + loss2






