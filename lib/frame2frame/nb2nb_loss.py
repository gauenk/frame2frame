"""

Neighborhood2Neighborhood



"""


import torch as th
import torch.nn as nn
import numpy as np
from einops import rearrange

class Nb2NbLoss():

    def __init__(self,lambda1,lambda2,nepochs,epoch_ratio):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nepochs = nepochs
        self.epoch_ratio = epoch_ratio
        self.name = "nb2nb"

    def compute(self,model,noisy,epoch):


        # -- misc --
        # print("noisy.shape: ",noisy.shape)
        B = noisy.shape[0]
        noisy = rearrange(noisy,'b t c h w -> (b t) c h w')
        # clean = rearrange(clean,'b t c h w -> (b t) c h w')
        Lambda = (epoch / (1.*self.nepochs)) * self.epoch_ratio

        # -- noisy sub --
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)

        # -- denoised sub --
        with th.no_grad():
            deno_d = model(noisy).detach()
            deno_sub1 = generate_subimages(deno_d, mask1)
            deno_sub2 = generate_subimages(deno_d, mask2)
        deno_diff = deno_sub1 - deno_sub2

        # -- denoise --
        deno = model(noisy_sub1)
        diff = deno - noisy_sub2
        loss1 = th.mean(diff**2)
        loss2 = Lambda * th.mean((diff - deno_diff)**2)
        loss_all = self.lambda1 * loss1 + self.lambda2 * loss2

        # -- reshape --
        deno = rearrange(deno_d,'(b t) c h w -> b t c h w',b=B)

        # -- return --
        return deno,loss_all
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
    mask1 = th.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=th.bool,
                        device=img.device)
    mask2 = th.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=th.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = th.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=th.int64,
        device=img.device)
    rd_idx = th.zeros(size=(n * h // 2 * w // 2, ),
                      dtype=th.int64,
                      device=img.device)
    th.randint(low=0,
               high=8,
               size=(n * h // 2 * w // 2, ),
               generator=get_generator(),
               out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += th.arange(start=0,
                             end=n * h // 2 * w // 2 * 4,
                             step=4,
                             dtype=th.int64,
                             device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = th.zeros(n, c, h // 2, w // 2,
                        dtype=img.dtype,
                        layout=img.layout,
                        device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        # print(mask.shape,img_per_channel.shape)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = th.nn.functional.unfold(x, block_size, stride=block_size)
    # print((n, c, h, w), unfolded_x.shape)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = th.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

