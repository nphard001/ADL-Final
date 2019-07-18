from __future__ import print_function

import argparse
import math
import os
import random
import pickle

import ipdb
import torch
import torch.optim as optim
from torch import nn

from src.loss import TripletLossIP
from src.model import ResponseEncoder, StateTracker, Reconstruct
from src.monitor import ExpMonitorSl as ExpMonitor
from src.ranker import Ranker
from src.sim_user import SynUser
from src.process_fasttext import load_embedding

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Image Retrieval')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--model-folder', type=str, default="models/",
                        help='triplet loss margin ')
    parser.add_argument('--fasttext', type=str, default="features/crawl-300d-2M.vec",
                        help='fasttext embedding ')
    parser.add_argument('--embedding', type=str, default="features/embedding.pkl",
                        help='processed embedding vectors')

    # learning
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--triplet-margin', type=float, default=0.1, metavar='EV',
                        help='triplet loss margin')
    # exp. control
    parser.add_argument('--train-turns', type=int, default=5,
                        help='dialog turns for training')
    parser.add_argument('--test-turns', type=int, default=5,
                        help='dialog turns for testing')
    return parser.parse_args()


def train_val_epoch(train: bool):
    print(('Train' if train else 'Eval') + f'\tepoch #{epoch}')
    encoder.train(train)
    tracker.train(train)
    reconstructor.train(train)
    triplet_loss.train(train)

    exp_monitor_candidate = ExpMonitor(args, user, train_mode=train)
    img_features = user.train_feature.to(device) if train else user.test_feature.to(device)
    dialog_turns = args.train_turns if train else args.test_turns

    target_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    candidate_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    false_img_idx = torch.empty(args.batch_size, dtype=torch.long, device=device)
    num_batches = math.ceil(img_features.size(0) / args.batch_size)

    if not train:
        ranker.update_rep(encoder, img_features)

    for batch_idx in range(1, num_batches + 1):
        # sample target images and first turn feedback images
        user.sample_idx(target_img_idx, train_mode=train)
        user.sample_idx(candidate_img_idx, train_mode=train)

        if train:
            ranker.update_rep(encoder, img_features)

        history_rep = None
        outs = []
        # start dialog
        for k in range(dialog_turns):
            candidate_img_feat = ranker.feat[candidate_img_idx].to(device)

            # get both original text and index for feedback
            relative_text_idx, relative_text = user.get_feedback_with_sent(act_idx=candidate_img_idx,
                                                                           user_idx=target_img_idx, train_mode=train)
            # encode image and relative_text_ids
            response_rep = encoder(candidate_img_feat, relative_text_idx.cuda())

            # update history representation
            current_state, history_rep = tracker(response_rep, history_rep)

            user.sample_idx(false_img_idx, train_mode=train)
            target_img_feat = ranker.feat[target_img_idx].to(device)
            false_img_feat = ranker.feat[false_img_idx].to(device)

            # loss
            loss = triplet_loss.forward(current_state,
                                        target_img_feat,
                                        false_img_feat)

            outs.append(loss)

            # obtain the next turn's feedback images
            candidate_img_idx = ranker.nearest_neighbor(current_state.detach())

            # ranking
            ranking_candidate = ranker.compute_rank(current_state.detach(), target_img_idx)

            # log
            exp_monitor_candidate.log_step(ranking_candidate, loss.detach(),
                                           target_img_idx, false_img_idx, candidate_img_idx, k)

        if train:
            logits = reconstructor(torch.mean(history_rep, 0))
            loss_cls = classification_criterion(logits, target_img_idx)

            # finish dialog and update model parameters
            optimizer_encoder.zero_grad()
            optimizer_tracker.zero_grad()
            optimizer_recon.zero_grad()
            mean_loss = torch.stack(outs).mean() + loss_cls
            mean_loss.backward(retain_graph=True)
            optimizer_encoder.step()
            optimizer_tracker.step()
            optimizer_recon.step()

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

        print(f"Using device: {device}")

        user = SynUser()
        ranker = Ranker()

        # load fasttext embedding vectors
        print("load fasttext",args.fasttext)
        if not os.path.isfile(args.embedding):
            embedding = load_embedding(args.fasttext,user.captioner_relative.vocab)
            with open(args.embedding, 'wb') as f:
                pickle.dump(embedding,f)
        else:
            with open(args.embedding, 'rb') as f:
                embedding = pickle.load(f)
                print(embedding.size())

        encoder = ResponseEncoder(user.vocabSize+1, hid_dim=256, out_dim=256, max_len=16, bert_dim=768,embedding=embedding).to(device)
        tracker = StateTracker(input_dim=256, hid_dim=512, out_dim=256).to(device)
        reconstructor = Reconstruct(256).to(device)

        optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr)
        optimizer_tracker = optim.Adam(tracker.parameters(), lr=args.lr)
        optimizer_recon = optim.Adam(reconstructor.parameters(), lr=args.lr)
        triplet_loss = TripletLossIP(margin=args.triplet_margin).to(device)
        classification_criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            train_val_epoch(train=True)
            with torch.no_grad():
                train_val_epoch(train=False)
            torch.save({'encoder': encoder.state_dict(),
                        'tracker': tracker.state_dict()},
                       os.path.join(args.model_folder, f'sl-{epoch}.pt'))
