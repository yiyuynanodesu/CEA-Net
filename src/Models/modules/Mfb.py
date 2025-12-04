from torch import nn
import torch
import torch.nn.functional as F

from einops import rearrange
import numbers

import math

class MFB(nn.Module):
    def __init__(self, img_feat_size, ques_feat_size, is_first=True, MFB_K=5, MFB_O=2048, DROPOUT_R=0.1):
        super(MFB, self).__init__()
        self.is_first = is_first
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R
        self.proj_i = nn.Linear(img_feat_size, self.MFB_K * self.MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, self.MFB_K * self.MFB_O)
        self.dropout = nn.Dropout(self.DROPOUT_R)
        self.pool = nn.AvgPool1d(self.MFB_K, stride=self.MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z, exp_out
