from __future__ import print_function

import argparse
import random
import os

import ipdb
import torch
from torch.nn import functional as F
import torch.optim as optim

import math
from src.sim_user import SynUser
from src.ranker import Ranker
from src.model import ResponseEncoder, StateTracker
from src.loss import TripletLossIP
from src.monitor import ExpMonitorRl as ExpMonitor


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
    top_k_act_img_idx = ranker.k_nearest_neighbors(target_state.detach(), K=args.top_k)

    # 2. rollout for each candidate in top k
    target_hx_bk = target_tracker.history_rep
    rollout_values = []
    for i in range(args.top_k):
        target_tracker.history_rep = target_hx_bk.detach()
        act_img_idx = top_k_act_img_idx[:, i]
        score = 0
        for j in range(max_turn - cur_turn):
            txt_input = user.get_feedback(act_idx=act_img_idx, user_idx=user_img_idx,  train_mode=True)

            act_emb = ranker.feat[act_img_idx]

            with torch.no_grad():
                action = target_encoder(act_emb, txt_input)
                action, _ = target_tracker(action)

            act_img_idx = ranker.nearest_neighbor(action.data)
            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            score = score + ranking_candidate
        rollout_values.append(score)
    rollout_values = torch.stack(rollout_values, dim=1)
    # compute greedy actions
    _, greedy_idx = rollout_values.min(dim=1)
    # recover target_state
    target_tracker.history_rep = target_hx_bk

    act_opt = torch.gather(top_k_act_img_idx, 1, greedy_idx.cpu().unsqueeze(1)).view(-1)

    # 3. compute loss
    # compute the log prob for candidates
    dist_action = []
    act_input = all_input[act_opt]

    act_emb = behavior_encoder.encode_image(act_input)
    dist = -torch.sum((behavior_state - act_emb) ** 2, dim=1) / args.tau
    dist_action.append(dist)
    for i in range(args.neg_num):
        neg_img_idx = torch.empty(args.batch_size, dtype=torch.long)
        user.sample_idx(neg_img_idx, train_mode=True)

        neg_input = all_input[neg_img_idx]
        neg_emb = behavior_encoder.encode_image(neg_input)
        dist = -torch.sum((behavior_state - neg_emb) ** 2, dim=1) / args.tau
        dist_action.append(dist)
    dist_action = torch.stack(dist_action, dim=1)
    label_idx = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    loss = F.cross_entropy(input=dist_action, target=label_idx)
    # compute the reg following the pre-training loss
    target_emb = ranker.feat[user_img_idx]
    reg = torch.sum((behavior_state - target_emb) ** 2, dim=1).mean()

    return act_opt, reg + loss


def train_val_rl(epoch, train: bool):
    print(('Train' if train else 'Eval') + f'\tEpoch #{epoch}')
    if train:
        behavior_encoder.set_rl_mode()
    behavior_encoder.train(train)
    behavior_tracker.train(train)
    target_encoder.eval()
    target_tracker.eval()

    exp_monitor_candidate = ExpMonitor(args, user, train_mode=train)

    img_features = user.train_feature if train else user.test_feature
    dialog_turns = args.train_turns if train else args.test_turns

    target_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    candidate_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    false_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)

    ranker.update_rep(target_encoder, img_features)

    num_batch = math.ceil(img_features.size(0) / args.batch_size)
    for batch_idx in range(1, num_batch + 1):
        # sample data index
        user.sample_idx(target_img_idx, train_mode=train)
        user.sample_idx(candidate_img_idx, train_mode=train)

        history_rep_behavior = None
        if train:
            history_rep_target = None

        loss_sum = 0
        for k in range(dialog_turns):
            relative_text_idx = user.get_feedback(act_idx=candidate_img_idx, user_idx=target_img_idx, train_mode=train)

            # extract img features
            candidate_img_feat = ranker.feat[candidate_img_idx]
            # encode image and relative_text_ids
            response_rep_behavior = behavior_encoder(candidate_img_feat, relative_text_idx)
            # update history representation
            current_state_behavior, history_rep_behavior = behavior_tracker(response_rep_behavior, history_rep_behavior)

            if train:
                with torch.no_grad():
                    response_rep_target = target_encoder(candidate_img_feat, relative_text_idx)
                    current_state_target, history_rep_target = target_tracker(response_rep_target, history_rep_target)
                ranking_candidate = ranker.compute_rank(current_state_behavior.detach(), target_img_idx)

                act_img_idx_mc, loss = rollout_search(current_state_behavior, current_state_target, k,
                                                      dialog_turns, target_img_idx, img_features)

                loss_sum = loss + loss_sum
                candidate_img_idx.copy_(act_img_idx_mc)
            else:
                candidate_img_idx = ranker.nearest_neighbor(current_state_behavior)
                user.sample_idx(false_img_idx, train_mode=train)

                target_img_feat = ranker.feat[target_img_idx]
                false_img_feat = ranker.feat[false_img_idx]

                ranking_candidate = ranker.compute_rank(current_state_behavior, target_img_idx)
                # TODO: not the same space
                loss = triplet_loss(current_state_behavior, target_img_feat, false_img_feat)

            exp_monitor_candidate.log_step(ranking_candidate, loss.detach(), target_img_idx, candidate_img_idx, k)

        if train:
            optimizer_encoder.zero_grad()
            optimizer_tracker.zero_grad()
            loss_sum.backward(retain_graph=True)
            optimizer_encoder.step()
            optimizer_tracker.step()

        if batch_idx % args.log_interval == 0 and train:
            print('# candidate ranking #')
            exp_monitor_candidate.print_interval(epoch, batch_idx, num_batch)

    print('# candidate ranking #')
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

        encoder_config = {'num_emb': user.vocabSize + 1, 'hid_dim': 256, 'out_dim': 256, 'max_len': 16}
        tracker_config = {'input_dim': 256, 'hid_dim': 512, 'out_dim': 256}
        behavior_encoder = ResponseEncoder(**encoder_config).to(device)
        behavior_tracker = StateTracker(**tracker_config).to(device)
        target_encoder = ResponseEncoder(**encoder_config).to(device)
        target_tracker = StateTracker(**tracker_config).to(device)

        checkpoint = torch.load(args.pretrained_model, map_location=lambda storage, loc: storage)
        behavior_encoder.load_state_dict(checkpoint['encoder'])
        behavior_tracker.load_state_dict(checkpoint['tracker'])
        target_encoder.load_state_dict(checkpoint['encoder'])
        target_tracker.load_state_dict(checkpoint['tracker'])

        optimizer_encoder = optim.Adam(behavior_encoder.parameters(), lr=args.lr)
        optimizer_tracker = optim.Adam(behavior_tracker.parameters(), lr=args.lr)
        triplet_loss = TripletLossIP(margin=args.triplet_margin).to(device)

        for epoch in range(1, args.epochs+1):
            with torch.no_grad():
                train_val_rl(epoch, train=False)
            train_val_rl(epoch, train=True)
            torch.save({'encoder': behavior_encoder.state_dict(),
                        'tracker': behavior_tracker.state_dict()},
                       os.path.join(args.model_folder, f'rl-{epoch}.pt'))

        with torch.no_grad:
            train_val_rl(args.epochs, train=False)
