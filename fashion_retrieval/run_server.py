from __future__ import print_function

import argparse
import random
import time
import random

import ipdb
import torch
from collections import defaultdict, Counter

from src.sim_user import SynUser
from src.ranker import Ranker
from src.model import ResponseEncoder, StateTracker
from nphard001.api_chatbot import HostDataChatbotAPI


def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--top-k', type=int, default=4,
                        help='top k candidate for policy and nearest neighbors')
    parser.add_argument('--pretrained-model', type=str, default="models/rl-3.pt",
                        help='path to pretrained sl model')
    # exp. control
    parser.add_argument('--turns', type=int, default=5,
                        help='dialog turns')
    return parser.parse_args()


def convert_sent2idx(sent: str, max_sent_len=16) -> torch.Tensor:
    idx2word = user.captioner_relative.vocab
    word2idx = {word: int(idx) for idx, word in idx2word.items()}
    sent_idx = []
    for token in sent.strip().split():
        try:
            sent_idx.append(word2idx[token])
        except KeyError:
            pass

    return torch.tensor([sent_idx + [0] * (max_sent_len - len(sent_idx))], dtype=torch.long, device=device)


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        max_sent_len = 16
        max_dialog_time = 30
        last_k_turns = 5

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user = SynUser()
        ranker = Ranker()

        encoder_config = {'num_emb': user.vocabSize + 1, 'hid_dim': 256, 'out_dim': 256, 'max_len': 16}
        tracker_config = {'input_dim': 256, 'hid_dim': 512, 'out_dim': 256}
        behavior_encoder = ResponseEncoder(**encoder_config).to(device)
        behavior_tracker = StateTracker(**tracker_config).to(device)

        checkpoint = torch.load(args.pretrained_model, map_location=lambda storage, loc: storage)
        behavior_encoder.load_state_dict(checkpoint['encoder'])
        behavior_tracker.load_state_dict(checkpoint['tracker'])

        behavior_encoder.eval()
        behavior_tracker.eval()

        img_features = user.train_feature

        ranker.update_rep(behavior_encoder, img_features)

        user_hist = defaultdict(lambda: None)
        user_dialog_counter = Counter()
        user_last_reply = defaultdict(lambda: [random.randrange(10000)])

        api = HostDataChatbotAPI()

        with torch.no_grad():
            while(True):
                pending_list = api.get_pending_list()
                if len(pending_list) == 0:
                    time.sleep(1)
                    continue
                print(f"Received requests: {len(pending_list)}")
                for i, json_in in enumerate(pending_list):
                    user_id = json_in['line_userId']
                    relative_text = json_in['text_list'][-1]

                    # if user_dialog_counter[user_id] >= max_dialog_time:
                    #     continue

                    if relative_text.lower().startswith('restart'):
                        user_dialog_counter[user_id] = 0
                        user_hist[user_id] = None
                        user_last_reply[user_id] = []

                    relative_text_idx = convert_sent2idx(relative_text, max_sent_len)

                    candidate_img_idx = user_last_reply[user_id][-1] if len(user_last_reply[user_id]) > 0 \
                        else random.randrange(10000)
                    candidate_img_feat = ranker.feat[candidate_img_idx]
                    response_rep_behavior = behavior_encoder(candidate_img_feat, relative_text_idx)

                    current_state_behavior, user_hist[user_id] = behavior_tracker(
                        response_rep_behavior,
                        user_hist[user_id][:last_k_turns] if user_hist[user_id] is not None else None)

                    reply_img_idx = ranker.k_nearest_neighbors(current_state_behavior, 10).squeeze()

                    for reply_idx in reply_img_idx:
                        if reply_idx.item() not in user_last_reply[user_id]:
                            reply_img_idx = reply_idx.item()
                            break
                    else:
                        reply_img_idx = reply_img_idx[0].item()

                    json_out = api.send_reply_index(user_id, reply_img_idx)
                    user_last_reply[user_id].append(reply_img_idx)
                    user_dialog_counter[user_id] += 1
