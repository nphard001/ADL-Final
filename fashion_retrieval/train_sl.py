from __future__ import print_function

import argparse
import math
import random

import ipdb
import torch
import torch.optim as optim

from sim_user import SynUser
from ranker import Ranker
from model import NetSynUser
from loss import TripletLossIP
from monitor import ExpMonitorSl as ExpMonitor


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Image Retrieval')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--model-folder', type=str, default="models/",
                        help='triplet loss margin ')
    # learning
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--triplet-margin', type=float, default=0.1, metavar='EV',
                        help='triplet loss margin ')
    # exp. control
    parser.add_argument('--train-turns', type=int, default=5,
                        help='dialog turns for training')
    parser.add_argument('--test-turns', type=int, default=5,
                        help='dialog turns for testing')
    args = parser.parse_args()

    return args


def train_sl():
    print('train epoch #{}'.format(epoch))
    model.train()
    triplet_loss.train()
    exp_monitor_candidate = ExpMonitor(args, user, train_mode=True)
    # train / test
    all_input = user.train_feature
    dialog_turns = args.train_turns

    user_img_idx = torch.empty(args.batch_size, dtype=torch.long)
    act_img_idx = torch.empty(args.batch_size, dtype=torch.long)
    neg_img_idx = torch.empty(args.batch_size, dtype=torch.long)
    num_batches = math.ceil(all_input.size(0) / args.batch_size)

    for batch_idx in range(1, num_batches + 1):
        # sample target images and first turn feedback images
        user.sample_idx(user_img_idx, train_mode=True)
        user.sample_idx(act_img_idx, train_mode=True)

        ranker.update_rep(model, all_input)
        model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            model.hx = model.hx.cuda()
        outs = []

        act_input = all_input[act_img_idx].to(device)
        act_emb = model.forward_image(act_input)

        # start dialog
        for k in range(dialog_turns):
            # get relative captions from user model given user target images and feedback images
            txt_input = user.get_feedback(act_idx=act_img_idx, user_idx=user_img_idx, train_mode=True).to(device)
            # txt_input = Variable(txt_input)

            # update the query action vector given feedback image and text feedback in this turn
            action = model.merge_forward(act_emb, txt_input)
            # obtain the next turn's feedback images
            act_img_idx = ranker.nearest_neighbor(action.data)

            # sample negative images for triplet loss
            user.sample_idx(neg_img_idx, train_mode=True)

            user_input = all_input[user_img_idx].to(device)
            neg_input = all_input[neg_img_idx].to(device)
            new_act_input = all_input[act_img_idx].to(device)
            # user_input, neg_input, new_act_input = Variable(user_input), Variable(neg_input), Variable(new_act_input)

            new_act_emb = model.forward_image(new_act_input)
            # ranking and loss
            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            user_emb = model.forward_image(user_input)
            neg_emb = model.forward_image(neg_input)
            loss = triplet_loss.forward(action, user_emb, neg_emb)

            outs.append(loss)
            act_emb = new_act_emb
            # log
            exp_monitor_candidate.log_step(ranking_candidate, loss.detach(), user_img_idx, neg_img_idx, act_img_idx, k)

        # finish dialog and update model parameters
        optimizer.zero_grad()
        outs = torch.stack(outs).mean()
        outs.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            exp_monitor_candidate.print_interval(epoch, batch_idx, num_batches)
    exp_monitor_candidate.print_all(epoch)


def eval():
    print('eval epoch #{}'.format(epoch))
    model.eval()
    triplet_loss.eval()
    exp_monitor_candidate = ExpMonitor(args, user, train_mode=False)
    # train / test
    all_input = user.test_feature
    dialog_turns = args.test_turns

    user_img_idx = torch.empty(args.batch_size, dtype=torch.long)
    act_img_idx = torch.empty(args.batch_size, dtype=torch.long)
    neg_img_idx = torch.empty(args.batch_size, dtype=torch.long)
    num_batches = math.ceil(all_input.size(0) / args.batch_size)

    ranker.update_rep(model, all_input)
    for batch_idx in range(1, num_batches + 1):
        # sample data index
        user.sample_idx(user_img_idx,  train_mode=False)
        user.sample_idx(act_img_idx, train_mode=False)

        model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            model.hx = model.hx.cuda()

        outs = []

        act_input = all_input[act_img_idx].to(device)
        act_emb = model.forward_image(act_input)

        for k in range(dialog_turns):
            txt_input = user.get_feedback(act_idx=act_img_idx, user_idx=user_img_idx, train_mode=False).to(device)
            user.sample_idx(neg_img_idx, train_mode=False)

            action = model.merge_forward(act_emb, txt_input)
            act_img_idx = ranker.nearest_neighbor(action.data)
            user_input = all_input[user_img_idx].to(device)
            neg_input = all_input[neg_img_idx].to(device)
            new_act_input = all_input[act_img_idx].to(device)
            new_act_emb = model.forward_image(new_act_input)

            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            user_emb = model.forward_image(user_input)
            neg_emb = model.forward_image(neg_input)
            loss = triplet_loss.forward(action, user_emb, neg_emb)

            outs.append(loss)
            act_emb = new_act_emb

            # log
            exp_monitor_candidate.log_step(ranking_candidate, loss.detach(), user_img_idx, neg_img_idx, act_img_idx, k)

        if batch_idx % args.log_interval == 0:
            exp_monitor_candidate.print_interval(epoch, batch_idx, num_batches)
    exp_monitor_candidate.print_all(epoch)


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        user = SynUser()
        ranker = Ranker()

        model = NetSynUser(user.vocabSize + 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
        triplet_loss = TripletLossIP(margin=args.triplet_margin).to(device)

        for epoch in range(1, args.epochs + 1):
            train_sl()
            with torch.no_grad():
                eval()
            torch.save(model.state_dict(), (args.model_folder+'sl-{}.pt').format(epoch))
