#!/usr/bin/python
# coding=utf-8
"""
Author: tal 
Created on 05/05/2018

# Sample usage:
#
#
# Training python:
#     $ python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#
# Resuming from the latest checkpoint of the experiment:
#     $ python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#
# Resuming from a specific checkpoint python
#     $ python sample.py --train_path $TRAIN_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR
#

"""
import argparse
import logging
import os
from os.path import dirname

import replie


def set_loggers():
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = list()
    formatter = logging.Formatter(LOG_FORMAT)
    f_handler = logging.FileHandler('seq2seq.log')
    f_handler.setFormatter(formatter)
    f_handler.setLevel(level=getattr(logging, opt.log_level.upper()))
    handlers.append(f_handler)

    if opt.debug:
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(formatter)
        s_handler.setLevel(level=getattr(logging, opt.log_level.upper()))
        handlers.append(s_handler)
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()), handlers=handlers)
    logging.info(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', dest='train_path',
                        help='Path to train data', default='data')
    parser.add_argument('--dev_path', action='store', dest='dev_path',
                        help='Path to dev data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir',
                        default=os.path.join(dirname(dirname(os.path.abspath(__file__))), 'experiment'),
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint '
                             'directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--log-level', dest='log_level',
                        default='info',
                        help='Logging level.')

    parser.add_argument('--no-dev', action='store_true', dest='no_dev',
                        default=False,
                        help='Logging level.')

    parser.add_argument('--num_epochs', dest='num_epochs',
                        default=500,
                        help='num_epochs')

    opt = parser.parse_args()

    set_loggers()

    default_data_dir = os.path.join(dirname(dirname(os.path.abspath(__file__))))

    replie.run_training(opt, default_data_dir, int(opt.num_epochs))
