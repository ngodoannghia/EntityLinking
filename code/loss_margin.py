import torch
import torch.nn as nn


class MyMarginLoss(nn.Module):
    def __init__(self):
        super(MyMarginLoss, self).__init__() 
        self.marginloss = nn.MultiMarginLoss()
    
    def forward(self, source, target, mask_mentions, mask_candidates):
        loss = 0
        batch, num_men, num_cand = source.shape
        for idx_sample in range(batch):
            for idx_men in range(num_men):
                if not mask_mentions[idx_sample, idx_men]:
                    continue
                tmp = torch.masked_select(source[idx_sample, idx_men], mask_candidates[idx_sample, idx_men])

                print(tmp)
                id_target = target[idx_sample, idx_men]
                
                size = tmp.shape[0]
                loss += (torch.sum(1 - (tmp[id_target] - tmp)) - 1) / size

        return loss