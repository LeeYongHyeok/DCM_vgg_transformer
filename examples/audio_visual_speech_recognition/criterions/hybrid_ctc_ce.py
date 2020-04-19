#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from itertools import groupby

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion

from examples.audio_visual_speech_recognition.data.data_utils import encoder_padding_mask_to_lengths
from examples.audio_visual_speech_recognition.utils.wer_utils import Code, EditDistance, Token


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@register_criterion("hybrid_ctc_ce_loss")
class hybrid_ctc_ce_Criterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.blank_idx = task.target_dictionary.index("<ctc_blank>")
        self.pad_idx = task.target_dictionary.pad()
        self.task = task

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--use-source-side-sample-size",
            action="store_true",
            default=False,
            help=(
                "when compute average loss, using number of source tokens "
                + "as denominator. "
                + "This argument will be no-op if sentence-avg is used."
            ),
        )
        parser.add_argument(
                "--hybrid_ctc_alpha",
                type=float,
                default=0.2,
                help="alpha * log_prob_ctc + (1-alpha) * log_prob_att"
                )

    def forward(self, model, sample, reduce=True, log_probs=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        enc_output = model.forward_ctc_encoder(**sample["net_input"])
        enc_output["encoder_out"] = enc_output["encoder_out"].permute(1,0,2) # [T, B, F] -> [B, T, F]
        
        lprobs_enc = model.get_normalized_probs([enc_output["encoder_out"]], log_probs=log_probs)

        #CTC loss 
        if not hasattr(lprobs_enc, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the encoder output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )

        batch_first = getattr(lprobs_enc, "batch_first", True)

        if not batch_first:
            max_seq_len = lprobs_enc.size(0)
            bsz = lprobs_enc.size(1)
        else:
            max_seq_len = lprobs_enc.size(1)
            bsz = lprobs_enc.size(0)
        device = enc_output["encoder_out"].device

        input_lengths = encoder_padding_mask_to_lengths(
            enc_output["encoder_padding_mask"], max_seq_len, bsz, device
        )
        target_lengths = sample["target_lengths"]
        targets = sample["target"]

        if batch_first:
            # N T D -> T N D (F.ctc_loss expects this)
            lprobs_enc = lprobs_enc.transpose(0, 1)

        pad_mask = sample["target"] != self.pad_idx
        targets_flat = targets.masked_select(pad_mask)

        loss_ctc = F.ctc_loss(
            lprobs_enc,
            targets_flat,
            input_lengths,
            target_lengths,
            blank=self.blank_idx,
            reduction="sum",
            zero_infinity=True,
        )

        lprobs_enc = lprobs_enc.transpose(0, 1)  # T N D -> N T D
        errors, total = self.compute_ctc_uer(
            lprobs_enc, targets, input_lengths, target_lengths, self.blank_idx
        )

        if self.args.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            if self.args.use_source_side_sample_size:
                sample_size = torch.sum(input_lengths).item()
            else:
                sample_size = sample["ntokens"]

        #CrossEntropy loss
        target = model.get_targets(sample, net_output)
        lprobs_dec, loss_ce = self.compute_CE_loss(model=model, net_output=net_output, target=target, reduction="sum", log_probs=log_probs)
    
        loss = self.args.hybrid_ctc_alpha * loss_ctc + (1 - self.args.hybrid_ctc_alpha) * loss_ce

        _, logging_output_CE = self.get_logging_output(sample, target, lprobs_dec, loss)

        return loss, sample_size, logging_output_CE 

    def compute_ctc_uer(self, logprobs, targets, input_lengths, target_lengths, blank_idx):
        """
            Computes utterance error rate for CTC outputs

            Args:
                logprobs: (Torch.tensor)  N, T1, D tensor of log probabilities out
                    of the encoder
                targets: (Torch.tensor) N, T2 tensor of targets
                input_lengths: (Torch.tensor) lengths of inputs for each sample
                target_lengths: (Torch.tensor) lengths of targets for each sample
                blank_idx: (integer) id of blank symbol in target dictionary

            Returns:
                batch_errors: (float) errors in the batch
                batch_total: (float)  total number of valid samples in batch
        """
        batch_errors = 0.0
        batch_total = 0.0
        for b in range(logprobs.shape[0]):
            predicted = logprobs[b][: input_lengths[b]].argmax(1).tolist()
            target = targets[b][: target_lengths[b]].tolist()
            # dedup predictions
            predicted = [p[0] for p in groupby(predicted)]
            # remove blanks
            nonblanks = []
            for p in predicted:
                if p != blank_idx:
                    nonblanks.append(p)
            predicted = nonblanks

            # compute the alignment based on EditDistance
            alignment = EditDistance(False).align(
                self.arr_to_toks(predicted), self.arr_to_toks(target)
            )

            # compute the number of errors
            # note that alignment.codes can also be used for computing
            # deletion, insersion and substitution error breakdowns in future
            for a in alignment.codes:
                if a != Code.match:
                    batch_errors += 1
            batch_total += len(target)

        return batch_errors, batch_total
    
    def arr_to_toks(self, arr):
        toks = []
        for a in arr:
            toks.append(Token(str(a), 0.0, 0.0))
        return toks
    
    def compute_CE_loss(self, model, net_output, target, reduction, log_probs):
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs, target, ignore_index=self.padding_idx, reduction=reduction
        )
        return lprobs, loss

    def get_logging_output(self, sample, target, lprobs, loss):
        target = target.view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = (
            sample["target"].size(0) if self.args.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
            "nframes": torch.sum(sample["net_input"]["audio_src_lengths"]).item(),
        }
        return sample_size, logging_output
    
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nframes": nframes,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


