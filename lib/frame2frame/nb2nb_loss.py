"""

Neighborhood2Neighborhood



"""


import torch as th
import torch.nn as nn
import numpy as np


class Neighbor2Neighbor():

    def __init__(self,lambda1,lambda2,nepochs,epoch_ratio):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nepochs = nepochs
        self.epoch_ratio = epoch_ratio

    def compute(self,model,noisy,epoch):

        # -- misc --
        Lambda = (epoch / (1.*self.nepochs)) * self.epoch_ratio

        # -- noisy sub --
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)

        # -- denoised sub --
        with torch.no_grad():
            deno_d = model(noisy).detach()
            deno_sub1 = generate_subimages(deno_d, mask1)
            deno_sub2 = generate_subimages(deno_d, mask2)
        deno_diff = deno_sub1 - deno_sub2

        # -- denoise --
        deno = model(noisy_sub1)
        diff = deno - noisy_sub2
        loss1 = torch.mean(diff**2)
        loss2 = Lambda * torch.mean((diff - deno_diff)**2)
        loss_all = self.lambda1 * loss1 + self.lambda2 * loss2

        # -- return --
        return loss_all
        # loss_all.backward()
        # optimizer.step()
        # print(
        #     '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
        #     .format(epoch, iteration, np.mean(loss1.item()), Lambda,
        #             np.mean(loss2.item()), np.mean(loss_all.item()),
        #             time.time() - st))

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


