import torch
import torch.nn as nn


class MyMarginLoss(nn.Module):
    def __init__(self, device):
        super(MyMarginLoss, self).__init__() 
        self.marginloss = nn.MarginRankingLoss(margin=1.5 ,reduction='mean')
        self.y = torch.tensor([1]).to(device)

    
    def forward(self, source, target, mask_mentions, mask_candidates):
        loss = 0
        batch, num_men, num_cand = source.shape

        # print(source.shape, target.shape, mask_mentions.shape, mask_candidates.shap


        for idx_sample in range(batch):
            for idx_men in range(num_men):
                if not mask_mentions[idx_sample, idx_men]:
                    continue

                tmp = source[idx_sample, idx_men].masked_fill(mask_candidates[idx_sample, idx_men]==False, -float('inf'))

                # print(mask_candidates[idx_sample, idx_men])
                # print(tmp)
                # print(target[idx_sample, idx_men])
                # print(torch.argmax(tmp, dim=-1))

                id_target = target[idx_sample, idx_men]  
                v_target = tmp[id_target].reshape(1)

                tmp = torch.cat((tmp[:id_target], tmp[id_target+1:]))

                loss += self.marginloss(v_target, tmp, self.y)

        return loss