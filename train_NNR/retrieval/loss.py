####################################################################################################
# TANSmodels: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2, contrast=False):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax
        self.contrast = True

    def forward(self, m, q, matched=None):
        q=q.squeeze(1)
        scores = torch.mm(m, q.t()) # (160, 160)
        diag = scores.diag() # (160,)
        scores = (scores - 1 * torch.diag(scores.diag()))
        # Sort the score matrix in the query dimension
        sorted_query, _ = torch.sort(scores, 0, descending=True)
        # Sort the score matrix in the model dimension
        sorted_model, _ = torch.sort(scores, 1, descending=True)
        # Select the nmax score
        max_q = sorted_query[:self.nmax, :] # (1, 160)
        max_m = sorted_model[:, :self.nmax] # (160, 1)
        neg_q = torch.sum(torch.clamp(max_q + 
            (self.margin - diag).view(1, -1).expand_as(max_q), min=0))
        neg_m = torch.sum(torch.clamp(max_m + 
            (self.margin - diag).view(-1, 1).expand_as(max_m), min=0))

        if self.contrast:
            loss = neg_m + neg_q
        else:
            loss = neg_m

        return loss

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def rbf_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
        # 将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # 调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)  # /len(kernel_val)

    def mmd_rbf(self,source, target):
        """
        计算源域数据和目标域数据的MMD距离
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            loss: MMD loss
        """
        batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
        kernels = self.rbf_kernel(source, target)
        # 根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算
    def forward(self, q_emb, m_emb):

        mmdloss=self.mmd_rbf(q_emb,m_emb)

        return mmdloss





