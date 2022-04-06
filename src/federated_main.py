#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, cal_benefit

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)  # print the experiment details

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(
        args)  # user_groups: a dict which key is the user index and values are training data index

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0


    epoch = 1
    # Number of selected users
    N = int(args.num_users * args.frac)

    idxs_users = np.arange(args.num_users, dtype=int)
    intervals = np.ones(len(idxs_users), dtype=int)

    benefits = np.zeros(len(idxs_users))
    acc_scores = []

    while epoch <= args.epochs:
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')

        global_model.train()

        # Randomly select clients (users) which use cpu and gpu
        assert 0 <= args.cpu_frac <= 1, 'cpu fraction should be in [0,1] !'
        idxs_users_cpu = np.random.choice(idxs_users, int(len(idxs_users) * args.cpu_frac), replace=False)
        idxs_users_gpu = np.setdiff1d(idxs_users, idxs_users_cpu)

        # Train the local models and use the average weights of local weights to update model in server
        for idx in idxs_users_cpu:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device='cpu')
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            benefits[idx] = cal_benefit(args, copy.deepcopy(global_model), global_weights, w, test_dataset)

        for idx in idxs_users_gpu:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device='cuda')
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            benefits[idx] = cal_benefit(args, copy.deepcopy(global_model), global_weights, w, test_dataset)

        cur_acc_scores = np.zeros(args.num_users)
        cur_acc_scores[idxs_users] = (benefits[idxs_users] / (benefits[idxs_users].max(axis=0) + np.spacing(0))).reshape(1, len(idxs_users))

        acc_scores.append(cur_acc_scores)

        max_interval = intervals.max(axis=0)
        scores = (np.array(acc_scores)[(epoch - max_interval):epoch] / max_interval).sum(axis=0) + (intervals / max_interval)
        # scores = np.minimum(np.ones(num_users, dtype=int), scores)
        scores = scores / scores.max()

        # select top N users based on scores for next epoch training
        selected_users_idx = scores.argsort()[-N:][::-1]
        unselected_users_idx = np.setdiff1d(np.arange(args.num_users), selected_users_idx)

        # renew the interval of selected users
        intervals = intervals + 1
        intervals[selected_users_idx] = 1

        idxs_users = selected_users_idx

        epoch = epoch + 1

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        # train loss it defined as the average of local losses, append every epoch
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all selected users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device='cuda' if args.gpu else 'cpu')
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))





    # for epoch in tqdm(range(args.epochs)):
    #     local_weights, local_losses = [], []
    #     print(f'\n | Global Training Round : {epoch+1} |\n')
    #
    #     global_model.train()
    #
    #     if args.specify_users_idx:
    #         idxs_users = args.users_idx
    #     else:
    #         m = max(int(args.frac * args.num_users), 1)  # number of selected users
    #         idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #
    #     # Randomly select clients (users) which use cpu and gpu
    #     assert args.cpu_frac >= 0 and args.cpu_frac <= 1, 'cpu fraction should be in [0,1] !'
    #     idxs_users_cpu = np.random.choice(idxs_users, int(len(idxs_users)*args.cpu_frac), replace=False)
    #     idxs_users_gpu = np.setdiff1d(idxs_users, idxs_users_cpu)
    #
    #     # Train the local models and use the average weights of local weights to update model in server
    #     for idx in idxs_users_cpu:
    #         local_model = LocalUpdate(args=args, dataset=train_dataset,
    #                                   idxs=user_groups[idx], logger=logger, device='cpu')
    #         w, loss = local_model.update_weights(
    #             model=copy.deepcopy(global_model), global_round=epoch)
    #         local_weights.append(copy.deepcopy(w))
    #         local_losses.append(copy.deepcopy(loss))
    #
    #     for idx in idxs_users_gpu:
    #         local_model = LocalUpdate(args=args, dataset=train_dataset,
    #                                   idxs=user_groups[idx], logger=logger, device='cuda')
    #         w, loss = local_model.update_weights(
    #             model=copy.deepcopy(global_model), global_round=epoch)
    #         local_weights.append(copy.deepcopy(w))
    #         local_losses.append(copy.deepcopy(loss))
    #
    #     # update global weights
    #     global_weights = average_weights(local_weights)
    #
    #     # update global weights
    #     global_model.load_state_dict(global_weights)
    #
    #     loss_avg = sum(local_losses) / len(local_losses)
    #     # train loss it defined as the average of local losses, append every epoch
    #     train_loss.append(loss_avg)
    #
    #     # Calculate avg training accuracy over all users at every epoch
    #     list_acc, list_loss = [], []
    #     global_model.eval()
    #     for c in range(args.num_users):
    #         # local_model = LocalUpdate(args=args, dataset=train_dataset,
    #         #                           idxs=user_groups[idx], logger=logger, device='cuda' if args.gpu else 'cpu')
    #         local_model = LocalUpdate(args=args, dataset=train_dataset,
    #                                   idxs=user_groups[c], logger=logger, device='cuda' if args.gpu else 'cpu')
    #         acc, loss = local_model.inference(model=global_model)
    #         list_acc.append(acc)
    #         list_loss.append(loss)
    #     train_accuracy.append(sum(list_acc) / len(list_acc))
    #
    #     # print global training loss after every 'i' rounds
    #     if (epoch + 1) % print_every == 0:
    #         print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
    #         print(f'Training Loss : {np.mean(np.array(train_loss))}')
    #         print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:

    file_name = os.path.join(args.project_home,
                             'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(args.dataset, args.model,
                                                                                          args.epochs, args.frac,
                                                                                          args.iid,
                                                                                          args.local_ep, args.local_bs))
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    logger.close()

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
