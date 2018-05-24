#!/usr/bin/python
# coding=utf-8
"""
Author: tal 
Created on 05/05/2018
"""
import os

from seq2seq.evaluator import Predictor
from seq2seq.models import Seq2seq, TopKDecoder
from seq2seq.util.checkpoint import Checkpoint


def get_model(checkpoint_path):
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    return seq2seq, input_vocab, output_vocab


def predict(expt_dir, seq_str, date, epoch, step, n=3):
    seq = seq_str.strip().split()
    checkpoint_path = os.path.join(Checkpoint.CHECKPOINT_DIR_NAME, date, epoch, step)
    seq2seq, input_vocab, output_vocab = get_model(checkpoint_path)
    beam_search = Seq2seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, 4))
    predictor = Predictor(beam_search, input_vocab, output_vocab)
    return predictor.predict_n(seq, n=n)
