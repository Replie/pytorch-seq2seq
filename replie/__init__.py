#!/usr/bin/python
# -*- coding: <encoding name> -*-
"""
@author talm

Description:

"""
import json
import os
import argparse
import logging
import re

import torch
from os.path import dirname
import torchtext

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

raw_input = input  # Python 3


def read_question_answers(source_file, target_file, reverse=False):
    # Read the file and split into lines
    question = open(source_file).read().strip().split('\n')
    answers = open(target_file).read().strip().split('\n')

    # Split every line into pairs and normalize
    # pairs = [[normalize_string(s) for s in question] normalize_string(s) for s in answers]

    pairs = zip([normalize_string(s) for s in question], [normalize_string(s) for s in answers])
    # pairs = zip(question, answers)
    # pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        # output_lang = Lang('eng')
    else:
        pairs = [list((p)) for p in pairs]
        # output_lang = Lang('eng')

    return pairs


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?,])", r" \1", s)
    s = re.sub(r"[^a-zA-Zא-ת.!?']+", r"", s)
    return s


default_data_dir = os.path.join(dirname(dirname(os.path.abspath(__file__))), 'data')
# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
formatter = logging.Formatter(LOG_FORMAT)
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
f_handler = logging.FileHandler('log_seq2seq.log')
f_handler.setFormatter(formatter)
f_handler.setLevel(level=getattr(logging, opt.log_level.upper()))
logging._addHandlerRef(f_handler)
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 50
    source_file = os.path.join(default_data_dir, 'reddit/q_w01.txt')
    target_file = os.path.join(default_data_dir, 'reddit/a_w01.txt')
    data_file = os.path.join(default_data_dir, 'reddit/data.txt')

    pairs = read_question_answers(source_file=source_file, target_file=target_file)
    with open(data_file, 'w') as data:
        for pair in pairs:
            data.write(json.dumps({'src': pair[0], 'tgt': pair[1]}) + '\n')


    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len


    train = torchtext.data.TabularDataset(
        path=data_file, format='json',
        fields={'src': ('src', src), 'tgt': ('tgt', tgt)},
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=data_file, format='json',
        fields={'src': ('src', src), 'tgt': ('tgt', tgt)},
        filter_pred=len_filter
    )
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=50,
                          print_every=10, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=100, dev_data=None,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
