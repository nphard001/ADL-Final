import sys
import time
import torch


class ExpMonitorSl:
    def __init__(self, args, user, train_mode):
        self.args = args
        self.user = user
        self.train_mode = train_mode
        self.num_turns = self.args.train_turns if self.train_mode else self.args.test_turns
        self.num_act = self.user.train_feature.size(0) if self.train_mode else self.user.test_feature.size(0)

        self.loss = torch.zeros(self.num_turns)
        self.all_loss = torch.zeros(self.num_turns)
        self.rank = torch.zeros(self.num_turns)
        self.all_rank = torch.zeros(self.num_turns)
        self.count = 0.0
        self.all_count = 0.0
        self.start_time = time.time()
        self.pos_idx = torch.zeros(self.num_act)
        self.neg_idx = torch.zeros(self.num_act)
        self.act_idx = torch.zeros(self.num_act)

    def log_step(self, ranking, loss, user_img_idx, neg_img_idx, act_img_idx, k):
        tmp_rank = ranking.float().mean()
        self.rank[k] += tmp_rank
        self.all_rank[k] += tmp_rank
        self.loss[k] += loss.item()
        self.all_loss[k] += loss.item()
        for i in range(user_img_idx.size(0)):
            self.pos_idx[user_img_idx[i]] += 1
            self.neg_idx[neg_img_idx[i]] += 1
            self.act_idx[act_img_idx[i]] += 1
        self.count += 1
        self.all_count += 1
        return

    def print_interval(self, epoch, batch_idx, num_epoch):
        output_string = 'Train Epoch:' if self.train_mode else 'Eval Epoch:'
        num_input = self.num_act

        output_string += '{} [{}/{} ({:.0f}%)]\tTime:{:.2f}\tNumAct:{}\n'.format(
            epoch, batch_idx, num_epoch, 100. * batch_idx / num_epoch, time.time() - self.start_time, self.pos_idx.sum()
        )
        output_string += 'pos:({:.0f}, {:.0f}) \tneg:({:.0f}, {:.0f}) \tact:({:.0f}, {:.0f})\n'.format(
            self.pos_idx.max(), self.pos_idx.min(), self.neg_idx.max(), self.neg_idx.min(), self.act_idx.max(), self.act_idx.min()
        )

        self.rank.mul_(self.num_turns / self.count)
        self.loss.mul_(1.0 / self.count)
        output_string += 'rank:'
        for i in range(self.num_turns):
            output_string += '{:.4f}\t '.format(self.rank[i] / num_input)
        output_string += '\nloss:'
        for i in range(self.num_turns):
            output_string += '{:.4f}\t '.format(self.loss[i])
        print(output_string)
        self.loss.zero_()
        self.rank.zero_()
        self.count = 0.0
        sys.stdout.flush()
        return

    def print_all(self, epoch):
        self.all_rank.mul_(self.num_turns/ self.all_count)
        self.all_loss.mul_(1.0 / self.all_count)
        output_string = '{} #rank:'.format(epoch)
        for i in range(self.num_turns):
            output_string += '{:.4f}\t '.format(self.all_rank[i] / self.num_act)
        output_string += '\n{} #loss:'.format(epoch)
        for i in range(self.num_turns):
            output_string += '{:.4f}\t '.format(self.all_loss[i])
        print(output_string)

        self.all_loss.zero_()
        self.all_rank.zero_()
        self.all_count = 0.0
        self.loss.zero_()
        self.rank.zero_()
        self.count = 0.0
        sys.stdout.flush()
        return


class ExpMonitorRl:
    def __init__(self, args, user, train_mode):
        self.args = args
        self.user = user
        self.train_mode = train_mode

        self.num_dialog_turns = self.args.train_turns if self.train_mode else self.args.test_turns
        self.num_data = self.user.train_feature.size(0) if self.train_mode else self.user.test_feature.size(0)

        self.loss = torch.zeros(self.num_dialog_turns)
        self.all_loss = torch.zeros(self.num_dialog_turns)
        self.rank = torch.zeros(self.num_dialog_turns)
        self.all_rank = torch.zeros(self.num_dialog_turns)
        self.count = 0.0
        self.all_count = 0.0
        self.start_time = time.time()
        self.pos_idx = torch.zeros(self.num_data)
        self.act_idx = torch.zeros(self.num_data)

    def log_step(self, ranking, loss, user_img_idx, act_img_idx, k):
        tmp_rank = ranking.float().mean()
        self.rank[k] += tmp_rank
        self.all_rank[k] += tmp_rank
        self.loss[k] += loss.item()
        self.all_loss[k] += loss.item()
        for i in range(user_img_idx.size(0)):
            self.pos_idx[user_img_idx[i]] += 1
            self.act_idx[act_img_idx[i]] += 1
        self.count += 1
        self.all_count += 1

    def print_interval(self, epoch, batch_idx, num_epoch):
        output_string = 'Train Epoch:' if self.train_mode else 'Eval Epoch:'

        output_string += '{} [{}/{} ({:.0f}%)]\tTime:{:.2f}\tNumAct:{}\n'.format(
            epoch, batch_idx, num_epoch, 100. * batch_idx / num_epoch, time.time() - self.start_time, self.pos_idx.sum()
        )
        output_string += 'pos:({:.0f}, {:.0f}) \tact:({:.0f}, {:.0f})\n'.format(
            self.pos_idx.max(), self.pos_idx.min(), self.act_idx.max(), self.act_idx.min()
        )

        self.rank.mul_(self.num_dialog_turns / self.count)
        self.loss.mul_(1.0 / self.count)
        output_string += 'rank:'
        for i in range(self.num_dialog_turns):
            output_string += '{:.4f}\t '.format(self.rank[i] / self.num_data)
        output_string += '\nloss:'
        for i in range(self.num_dialog_turns):
            output_string += '{:.4f}\t '.format(self.loss[i])
        print(output_string)
        self.loss.zero_()
        self.rank.zero_()
        self.count = 0.0
        sys.stdout.flush()

    def print_all(self, epoch):
        self.all_rank.mul_(self.num_dialog_turns / self.all_count)
        self.all_loss.mul_(1.0 / self.all_count)
        output_string = '{} #rank:'.format(epoch)
        for i in range(self.num_dialog_turns):
            output_string += '{:.4f}\t '.format(self.all_rank[i] / self.num_data)
        output_string += '\n{} #loss:'.format(epoch)
        for i in range(self.num_dialog_turns):
            output_string += '{:.4f}\t '.format(self.all_loss[i])
        print(output_string)

        self.all_loss.zero_()
        self.all_rank.zero_()
        self.all_count = 0.0
        self.loss.zero_()
        self.rank.zero_()
        self.count = 0.0
        sys.stdout.flush()
