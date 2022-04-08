#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import yaml
import os
import sys


def none_or_num(value):
    if value == 'None':
        return None
    else:
        return int(value)


def none_or_str(value):
    if value.lower() == 'none':
        return None
    else:
        return value


def parser_config(config_path):
    with open(config_path, 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    # parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    # parser.add_argument('--dataset', type=str, default='mnist', help="name \
    #                     of dataset")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                            of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, type=none_or_num, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # parser.add_argument('--cpu_frac', type=float, default=0.0, help='the fraction of cpu users, should be in [0,1]')
    parser.add_argument('--cpu_frac', type=float, default=1.0, help='the fraction of cpu users, should be in [0,1]')
    parser.add_argument('--conf_file_name', type=none_or_str, default=None, help='use configuration in <configure> '
                                                                                 'folder, default: use argsparse as'
                                                                                 ' configuration ')
    parser.add_argument('--specify_users_idx', type=bool, default=False, help='whether involves specific users in '
                                                                            'training process. Only support yaml '
                                                                            'configuration to import users indexes.('
                                                                            'default is using fraction to randomly '
                                                                            'select)')
    args = parser.parse_args()

    # the root of this repository
    args.project_home = os.path.abspath(os.path.join(os.path.realpath(__file__), "..\\.."))

    if args.conf_file_name:
        config_path = os.path.join(args.project_home, 'configure', args.conf_file_name)
        conf = parser_config(config_path)

        args.epochs = conf['epochs']
        args.num_users = conf['num_users']
        args.frac = conf['frac']
        args.local_ep = conf['local_ep']
        args.local_bs = conf['local_bs']
        args.lr = conf['lr']
        args.momentum = conf['momentum']
        args.model = conf['model']
        args.kernel_num = conf['kernel_num']
        args.kernel_size = conf['kernel_size']
        args.num_channels = conf['num_channels']
        args.norm = conf['norm']
        args.num_filters = conf['num_filters']
        args.max_pool = conf['max_pool']
        args.dataset = conf['dataset']
        args.num_classes = conf['num_classes']
        args.gpu = conf['gpu']
        args.optimizer = conf['optimizer']
        args.iid = conf['iid']
        args.unequal = conf['unequal']
        args.stopping_rounds = conf['stopping_rounds']
        args.verbose = conf['verbose']
        args.seed = conf['seed']
        args.cpu_frac = conf['cpu_frac']

        if args.specify_users_idx:
            assert len(conf['users_idx']) > 0, 'the number of users indexes should be larger than zero!'
            args.users_idx = conf['users_idx']

    return args
