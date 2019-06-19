import torch


class Ranker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feat = None

    def update_rep(self, encoder, input, batch_size=64):
        self.feat = torch.empty(input.size(0), encoder.out_dim, device=self.device, dtype=torch.float)

        for start in range(0, input.size(0), batch_size):
            end = start + batch_size
            x = input[start: end]
            out = encoder.encode_image(x)
            self.feat[start: end].copy_(out.data)
        self.feat.to(self.device)

    def compute_rank(self, input, target_idx):
        """

        :param input: size(N,) proposed indexes of images
        :param target_idx: size(N,) groud truth indexes of images
        :return: rank[i] = number of images (all images) farther than
                 the distance between input[i] and its ground truth (img index is target_idx[i])
        """
        # input <---- a batch of vectors
        # targetIdx <----- ground truth index
        # return rank of input vectors in terms of rankings in distance to the ground truth

        # target_idx = target_idx.to(self.device)
        target = self.feat[target_idx]

        value = torch.pow(target - input, 2).sum(1)
        rank = torch.empty(value.size(0), dtype=torch.long)
        for i in range(value.size(0)):
            val = self.feat - input[i].expand(self.feat.size(0), self.feat.size(1))
            val = torch.pow(val, 2).sum(1)
            rank[i] = val.lt(value[i]).sum()

        return rank

    def nearest_neighbor(self, target):
        target.to(self.device)
        idx = torch.empty(target.size(0), dtype=torch.long)

        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = torch.pow(val, 2).sum(1)
            min_idx = val.argmin(0)
            idx[i] = min_idx.item()
        return idx

    def k_nearest_neighbors(self, target, K=10):
        target.to(self.device)
        idx = torch.empty(target.size(0), K, dtype=torch.long)

        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = torch.pow(val, 2).sum(1)
            v, id = torch.topk(val, k=K, dim=0, largest=False)
            idx[i].copy_(id.view(-1))
        return idx
