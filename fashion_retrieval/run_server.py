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
    parser.add_argument('--pretrained-model', type=str, default="models/rl-1.pt",
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

        random.seed(633)
        torch.manual_seed(633)
        torch.cuda.manual_seed_all(633)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user = SynUser()
        ranker = Ranker()

        encoder_config = {'num_emb': user.vocabSize + 1, 'hid_dim': 256, 'out_dim': 256, 'max_len': 16}
        tracker_config = {'input_dim': 256, 'hid_dim': 512, 'out_dim': 256}
        behavior_encoder = ResponseEncoder(**encoder_config).to(device)
        behavior_tracker = StateTracker(**tracker_config).to(device)

        checkpoint = torch.load(args.pretrained_model, map_location=lambda storage, loc: storage)
        # behavior_encoder.load_state_dict(checkpoint['encoder'])
        # behavior_tracker.load_state_dict(checkpoint['tracker'])

        behavior_encoder.eval()
        behavior_tracker.eval()

        img_features = user.train_feature

        ranker.update_rep(behavior_encoder, img_features)

        user_hist = defaultdict(None)
        user_prev_img_idx = defaultdict(None)
        user_dialog_counter = Counter()

        api = HostDataChatbotAPI()

        with torch.no_grad():
            while(True):
                time.sleep(1)
                # polling
                for json_in in api.get_pending_list():
                    user_id = json_in['line_userId']
                    relative_text = json_in['text_list'][-1]

                    user_dialog_counter[user_id] += 1

                    if user_id is None:
                        # sample data index
                        candidate_img_idx = torch.empty(1, dtype=torch.long, device=device)
                        user.sample_idx(candidate_img_idx, train_mode=False)
                        user_prev_img_idx[user_id] = candidate_img_idx

                    # relative_text_idx = convert_sent2idx(input("Please input relative feedback:\n"))
                    # relative_text_idx = convert_sent2idx(relative_text)
                    # extract img features
                    # candidate_img_feat = ranker.feat[candidate_img_idx]
                    # encode image and relative_text_ids
                    # response_rep_behavior = behavior_encoder(candidate_img_feat, relative_text_idx)
                    # update history representation
                    # current_state_behavior, user_hist[user_id] = behavior_tracker(response_rep_behavior,
                    #                                                               user_hist[user_id])

                    # candidate_img_idx = ranker.nearest_neighbor(current_state_behavior).item()
                    # candidate_img_idx = ranker.nearest_neighbor(current_state_behavior).item()
                    # json_out = api.send_reply_index(user_id, candidate_img_idx)
                    json_out = api.send_reply_index(user_id, random.randint(0, 9999))
