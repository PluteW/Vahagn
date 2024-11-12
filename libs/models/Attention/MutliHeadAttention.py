import torch.nn as nn
import torch
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul  # 16 4 4
        u = u / self.scale # 2.Scale                    # 16 4 4

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax      # [16, 4, 4]
        output = torch.bmm(attn, v) # 5.Output

        return attn, output

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head    # 8
        self.d_k = d_k          # 128
        self.d_v = d_v          # 64
        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头 # 2, 4, 1024
        k = self.fc_k(k)                # 2, 4, 1024
        v = self.fc_v(v)                # 2, 4, 512
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q) # 16, 4, 128
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k) # 16, 4, 128
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v) # 16, 4, 64

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)    # 16 4 4
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出
        # 12, 7, 7      12, 7, 64
        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat      # 16, 4, 64
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """
    
    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))  # [80, 128] 
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))  # [80, 128]
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))  # [80, 64]

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)    # 2, 4, 128
        k = torch.matmul(x, self.wk)    # 2, 4, 128
        v = torch.matmul(x, self.wv)    # 2, 4, 64

        attn, output = self.mha(q, k, v, mask=mask)

        return attn, output
    
if __name__ == "__main__":
    n_x = 7
    d_x = 357
    batch = 2

    d_out = 357
    
    # 1 7 357 5000
    # batch 批大小 n_x 对象 d_x 单个对象特征
    x = torch.randn(batch, n_x,d_x, 5000).reshape(-1,n_x,d_x)            # 2 4 80
    # mask = torch.zeros(batch, n_x, n_x).bool()  # 2 4 4
    mask = None
    # d_k 
    selfattn = SelfAttention(n_head=6, d_k=128, d_v=128, d_x=d_x, d_o=d_out) 
    attn, output = selfattn(x, mask=mask)  # [16, 4, 4]  # [2, 4, 80]

    print(attn.size())
    print(output.size())
