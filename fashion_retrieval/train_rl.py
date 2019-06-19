from __future__ import print_function

import argparse
import random

import ipdb
import torch
from torch.nn import functional as F
import torch.optim as optim

import math
from sim_user import SynUser
from ranker import Ranker
from model import NetSynUser
from loss import TripletLossIP
from monitor import ExpMonitorRl as ExpMonitor


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train')
    # learning
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--tau', type=float, default=1,
                        help='softmax temperature')
    parser.add_argument('--seed', type=int, default=7771,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--neg-num', type=int, default=5,
                        help='number of negative candidates in the denominator')
    parser.add_argument('--model-folder', type=str, default="models/",
                        help='triplet loss margin ')

    parser.add_argument('--top-k', type=int, default=4,
                        help='top k candidate for policy and nearest neighbors')
    parser.add_argument('--pretrained-model', type=str, default="models/sl-12.pt",
                        help='path to pretrained sl model')
    parser.add_argument('--triplet-margin', type=float, default=0.1, metavar='EV',
                        help='triplet loss margin ')
    # exp. control
    parser.add_argument('--train-turns', type=int, default=5,
                        help='dialog turns for training')
    parser.add_argument('--test-turns', type=int, default=5,
                        help='dialog turns for testing')
    return parser.parse_args()


def rollout_search(behavior_state, target_state, cur_turn, max_turn, user_img_idx, all_input):
    # 1. compute the top-k nearest neighbor for current state
    top_k_act_img_idx = ranker.k_nearest_neighbors(target_state.data, K=args.top_k)

    # 2. rollout for each candidate in top k
    target_hx_bk = target_model.hx
    rollout_values = []
    for i in range(args.top_k):
        target_model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            target_model.hx = target_model.hx.cuda()
        target_model.hx.data.copy_(target_hx_bk.data)
        act_img_idx = top_k_act_img_idx[:, i]
        score = 0
        for j in range(max_turn - cur_turn):
            txt_input = user.get_feedback(act_idx=act_img_idx, user_idx=user_img_idx,  train_mode=True)
            if torch.cuda.is_available():
                txt_input = txt_input.cuda()

            if torch.cuda.is_available():
                act_img_idx = act_img_idx.cuda()
            act_emb = ranker.feat[act_img_idx]

            with torch.no_grad():
                action = target_model.merge_forward(act_emb, txt_input)
            act_img_idx = ranker.nearest_neighbor(action.data)
            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            score = score + ranking_candidate
        rollout_values.append(score)
    rollout_values = torch.stack(rollout_values, dim=1)
    # compute greedy actions
    _, greedy_idx = rollout_values.min(dim=1)
    # recover target_state
    target_model.hx = target_hx_bk
    if torch.cuda.is_available():
        greedy_idx = greedy_idx.cuda()

    act_opt = torch.gather(top_k_act_img_idx, 1, greedy_idx.cpu().unsqueeze(1)).view(-1)

    # 3. compute loss
    # compute the log prob for candidates
    dist_action = []
    act_input = all_input[act_opt]
    if torch.cuda.is_available():
        act_input = act_input.cuda()
    act_emb = behavior_model.forward_image(act_input)
    dist = -torch.sum((behavior_state - act_emb) ** 2, dim=1) / args.tau
    dist_action.append(dist)
    for i in range(args.neg_num):
        neg_img_idx = torch.empty(args.batch_size, dtype=torch.long)
        user.sample_idx(neg_img_idx, train_mode=True)

        neg_input = all_input[neg_img_idx]
        if torch.cuda.is_available():
            neg_input = neg_input.cuda()
        neg_emb = behavior_model.forward_image(neg_input)
        dist = -torch.sum((behavior_state - neg_emb) ** 2, dim=1) / args.tau
        dist_action.append(dist)
    dist_action = torch.stack(dist_action, dim=1)
    label_idx = torch.zeros(args.batch_size, dtype=torch.long)
    if torch.cuda.is_available():
        label_idx = label_idx.cuda()
    loss = F.cross_entropy(input=dist_action, target=label_idx)
    # compute the reg following the pre-training loss
    if torch.cuda.is_available():
        user_img_idx = user_img_idx.cuda()
    target_emb = ranker.feat[user_img_idx]
    reg = torch.sum((behavior_state - target_emb) ** 2, dim=1).mean()

    return act_opt, reg + loss


def train_rl(epoch, train: bool):
    print(('Train' if train else 'eval') + f'epoch #{epoch}')
    if train:
        behavior_model.set_rl_mode()
        target_model.eval()
    else:
        behavior_model.eval()

    exp_monitor_candidate = ExpMonitor(args, user, train_mode=train)

    img_features = user.train_feature if train else user.test_feature
    dialog_turns = args.train_turns if train else args.test_turns

    target_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    candidate_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    false_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)

    ######################
    ranker.update_rep(target_model, img_features)
    ######################

    # update ranker
    num_epoch = math.ceil(img_features.size(0) / args.batch_size)

    for batch_idx in range(1, num_epoch + 1):
        # sample data index
        user.sample_idx(target_img_idx, train_mode=train)
        user.sample_idx(candidate_img_idx, train_mode=train)

        behavior_model.init_hid(args.batch_size)
        if train:
            target_model.init_hid(args.batch_size)

        ######################
        ranker.update_rep(target_model, img_features)
        ######################




        loss_sum = 0
        for k in range(dialog_turns):
            # construct data
            relative_text_idx = user.get_feedback(act_idx=candidate_img_idx,
                                                  user_idx=target_img_idx, train_mode=train)

            candidate_img_feat = ranker.feat[candidate_img_idx]
            behavior_hist_rep = behavior_model.merge_forward(candidate_img_feat, relative_text_idx)
            with torch.no_grad():
                target_hist_rep = target_model.merge_forward(candidate_img_feat, relative_text_idx)

            ranking_candidate = ranker.compute_rank(behavior_hist_rep.detach(), target_img_idx)

            act_img_idx_mc, loss = rollout_search(behavior_hist_rep, target_hist_rep, k,
                                                  dialog_turns, target_img_idx, img_features)

            loss_sum = loss + loss_sum

            candidate_img_idx.copy_(act_img_idx_mc)

            exp_monitor_candidate.log_step(ranking_candidate, loss.detach(), target_img_idx, candidate_img_idx, k)

        if train:
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0 and train:
            print('# candidate ranking #')
            exp_monitor_candidate.print_interval(epoch, batch_idx, num_epoch)

    print('# candidate ranking #')
    exp_monitor_candidate.print_all(epoch)


def eval(epoch):
    # train_mode = True
    print('eval epoch #{}'.format(epoch))
    behavior_model.eval()
    triplet_loss.eval()
    train_mode = False
    all_input = user.test_feature
    dialog_turns = args.test_turns

    exp_monitor_candidate = ExpMonitor(args, user, train_mode=train_mode)

    user_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    act_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    neg_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    num_epoch = math.ceil(all_input.size(0) / args.batch_size)

    ranker.update_rep(behavior_model, all_input)
    for batch_idx in range(1, num_epoch + 1):
        # sample data index

        user.sample_idx(user_img_idx,  train_mode=train_mode)
        user.sample_idx(act_img_idx, train_mode=train_mode)

        behavior_model.init_hid(args.batch_size)

        act_emb = ranker.feat[act_img_idx]

        for k in range(dialog_turns):
            txt_input = user.get_feedback(act_idx=act_img_idx,
                                          user_idx=user_img_idx,
                                          train_mode=train_mode).to(device)

            action = behavior_model.merge_forward(act_emb, txt_input)
            act_img_idx = ranker.nearest_neighbor(action.detach())

            user.sample_idx(neg_img_idx, train_mode=train_mode)

            user_emb = ranker.feat[user_img_idx]
            neg_emb = ranker.feat[neg_img_idx]
            new_act_emb = ranker.feat[act_img_idx]

            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            loss = triplet_loss.forward(action, user_emb, neg_emb)
            act_emb = new_act_emb

            # log
            exp_monitor_candidate.log_step(ranking_candidate, loss.detach(), user_img_idx,  act_img_idx, k)

    exp_monitor_candidate.print_all(epoch)
    return


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user = SynUser()
        ranker = Ranker()

        behavior_model = NetSynUser(user.vocabSize + 1).to(device)
        target_model = NetSynUser(user.vocabSize + 1).to(device)
        triplet_loss = TripletLossIP(margin=args.triplet_margin).to(device)
        # load pre-trained model
        behavior_model.load_state_dict(torch.load(args.pretrained_model, map_location=lambda storage, loc: storage))
        # load pre-trained model
        target_model.load_state_dict(torch.load(args.pretrained_model, map_location=lambda storage, loc: storage))

        optimizer = optim.Adam(behavior_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

        # user_img_idx_ = torch.empty(args.batch_size, dtype=torch.long)
        # act_img_idx_ = torch.empty(args.batch_size, dtype=torch.long)
        # user.sample_idx(user_img_idx_, train_mode=True)
        # user.sample_idx(act_img_idx_, train_mode=True)

        for epoch in range(20):
            with torch.no_grad():
                eval(epoch)
            train_rl(epoch, train=True)
            torch.save(behavior_model.state_dict(), (args.model_folder+'rl-{}.pt').format(epoch))

        with torch.no_grad:
            eval(20)
