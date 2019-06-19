from nphard001.api_data import *
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset, random_split
from torch.utils.data.dataloader import default_collate
class SingleCategoryDataset(Dataset):
    api = HostDataAPI()
    meta = get_metadata_static_dict()
    def __init__(self, category: str='womens_high_heels', num: Optional[int]=None):
        self.num = num or self.meta[category]['num']
        self.ctg = category
        self.cache_dict = {}
    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        row = self.cache_dict.get(idx, None)
        if not row:
            # cache miss
            img_pillow = self.api.get_jpg(self.ctg, idx+1)
            caption_string = self.api.get_txt(self.ctg, idx+1)
            row = (self.ctg, img_pillow, caption_string)
            self.cache_dict[idx] = row
        return row
    @staticmethod
    def MakeCollate(to_tensor=False):
        r'''return raw collate_fn (to_tensor WIP)'''
        def collate_fn(batch):
            ctg = [b[0] for b in batch]
            img = [b[1] for b in batch]
            cap = [b[2] for b in batch]
            return ctg, img, cap
        return collate_fn
    @staticmethod
    def demo():
        from IPython.display import display
        ctg1 = SingleCategoryDataset('womens_high_heels')
        ctg2 = SingleCategoryDataset('womens_rain_boots')
        ctg3 = SingleCategoryDataset('womens_sneakers')
        
        # pytorch Dataset+Dataset = ConcatDataset
        batch_size = 8
        DL = DataLoader(ctg1+ctg2+ctg3, batch_size, shuffle=True, collate_fn=SingleCategoryDataset.MakeCollate())
        for ctg, img, cap in DL:
            # display first batch
            for i in range(batch_size):
                print(i, ctg[i], cap[i])
                display(img[i])
            break

