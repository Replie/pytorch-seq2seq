#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author talm

Description:

"""
import json
import os
import logging

import torch
import torchtext

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

raw_input = input  # Python 3


def run_training(opt, default_data_dir):
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

        data_file = os.path.join(default_data_dir, opt.train_path, 'data.txt')

        logging.info("Starting new Training session on %s", data_file)

        def len_filter(example):
            return (len(example.src) <= max_len) and (len(example.tgt) <= max_len) \
                   and (len(example.src) > 0) and (len(example.tgt) > 0)

        def char_len_filter(example):  # TODO different LENGHT
            return (len(example.src) <= max_len) and (len(example.tgt) <= max_len) \
                   and (len(example.src) > 0) and (len(example.tgt) > 0)

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
            logging.info("Yayyy We got CUDA!!!")
            loss.cuda()
        else:
            logging.info("No cuda available device found running on cpu")

        seq2seq = None
        optimizer = None
        if not opt.resume:
            hidden_size = 128
            decoder_hidden_size = hidden_size * 2
            logging.info("EncoderRNN Hidden Size: %s", hidden_size)
            logging.info("DecoderRNN Hidden Size: %s", decoder_hidden_size)
            bidirectional = True
            encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                                 bidirectional=bidirectional,
                                 rnn_cell='lstm',
                                 variable_lengths=True)
            decoder = DecoderRNN(len(tgt.vocab), max_len, decoder_hidden_size,
                                 dropout_p=0, use_attention=True,
                                 bidirectional=bidirectional,
                                 rnn_cell='lstm',
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

        num_epochs = 1000
        batch_size = 32
        checkpoint_every = num_epochs / 10
        print_every = num_epochs / 100

        properties = dict(batch_size=num_epochs,
                          checkpoint_every=checkpoint_every,
                          print_every=print_every, expt_dir=opt.expt_dir,
                          num_epochs=num_epochs,
                          teacher_forcing_ratio=0.5,
                          resume=opt.resume)

        logging.info("Starting training with the following Properties %s", json.dumps(properties, indent=2))
        t = SupervisedTrainer(loss=loss, batch_size=num_epochs,
                              checkpoint_every=checkpoint_every,
                              print_every=print_every, expt_dir=opt.expt_dir)

        seq2seq = t.train(seq2seq, train,
                          num_epochs=num_epochs, dev_data=dev,
                          optimizer=optimizer,
                          teacher_forcing_ratio=0.5,
                          resume=opt.resume)

        evaluator = Evaluator(loss=loss, batch_size=batch_size)
        dev_loss, accuracy = evaluator.evaluate(seq2seq, dev)
        logging.info("Dev Loss: %s", dev_loss)
        logging.info("Accuracy: %s", dev_loss)

    beam_search = Seq2seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, 4))

    predictor = Predictor(beam_search, input_vocab, output_vocab)
    while True:
        try:
            seq_str = raw_input("Type in a source sequence:")
            seq = seq_str.strip().split()
            results = predictor.predict_n(seq, n=3)
            for i, res in enumerate(results):
                print('option %s: %s\n', i + 1, res)
        except KeyboardInterrupt:
            logging.info("Bye Bye")
            exit(0)
