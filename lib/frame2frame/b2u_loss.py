"""

Neighborhood2Neighborhood



"""


import torch as th
import torch.nn as nn
import numpy as np
from einops import rearrange
import torchvision.transforms.functional as tvF

class B2ULoss():

    def __init__(self,lambda1,lambda2,nepochs,epoch_ratio,ninfo):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nepochs = nepochs
        self.epoch_ratio = epoch_ratio
        self.masker = Masker(width=4, mode='interpolate', mask_type='all')
        if "g-30" in ninfo or "pg-30" in ninfo:
            self.Thread1 = 0.8
            self.Thread2 = 1.0
        else:
            self.Thread1 = 0.4
            self.Thread2 = 1.0

    def compute(self,model,noisy,epoch):

        # -- reshape --
        B = noisy.shape[0]
        noisy = rearrange(noisy,'b t c h w -> (b t) c h w')

        # -- run --
        deno,loss = [],0
        for b in range(B):
            deno_b,loss_b = self.compute_sample(model,noisy[[b]],epoch)
            loss += loss_b / B
            deno.append(deno_b.detach())

        # -- reshape --
        deno = th.cat(deno)
        deno = rearrange(deno,'(b t) c h w -> b t c h w',b=B)

        return deno,loss

    def compute_sample(self,model,noisy,epoch):

        # -- step [diff] --
        # print("noisy.shape: ",noisy.shape)
        net_input, mask = self.masker.train(noisy)
        # print("net_input.shape: ",net_input.shape)
        # print("mask.shape: ",mask.shape)
        noisy_output = model(net_input)
        n, c, h, w = noisy.shape
        noisy_output = (noisy_output*mask).view(n, -1, c, h, w).sum(dim=1)
        diff = noisy_output - noisy

        # -- view --
        from dev_basics.utils import vid_io
        def normz(vid):
            vid0 = vid - vid.min()
            vid0 /= vid0.max()
            vid0 = th.clamp(vid0,0,1)
            return vid0
        # vid_io.save_video(normz(diff),"output/b2u/","diff")
        # vid_io.save_video(normz(noisy_output),"output/b2u/","noisy_output")

        # -- step [exp_diff] --
        with th.no_grad():
            exp_output = model(noisy)
        exp_diff = exp_output - noisy
        # vid_io.save_video(normz(exp_output),"output/b2u/","exp_output")


        # -- get [alpha,beta] --
        Lambda = epoch / self.nepochs
        if Lambda <= self.Thread1:
            beta = self.lambda2
        elif self.Thread1 <= Lambda <= self.Thread2:
            beta = Lambda2 + (Lambda - self.Thread1) * \
                (self.epoch_ratio-self.lambda2) / (self.Thread2-self.Thread1)
        else:
            beta = self.epoch_ratio
        alpha = self.lambda1
        # print(beta,alpha)

        # -- compute loss --
        revisible = diff + beta * exp_diff
        loss_reg = alpha * th.mean(diff**2)
        loss_rev = th.mean(revisible**2)
        loss_all = loss_reg + loss_rev

        # -- return --
        return noisy_output,loss_all

    def test(self,model,noisy):

        # -- reshape --
        B = noisy.shape[0]
        noisy = rearrange(noisy,'b t c h w -> (b t) c h w')

        with th.no_grad():

            # -- ensure multiple of 32 --
            H = noisy.shape[-2]
            W = noisy.shape[-1]
            val_size = (max(H, W) + 31) // 32 * 32
            device = noisy.device
            padH,padW = val_size - H,val_size - W
            noisy = tvF.pad(noisy,[0,0,padW,padH],padding_mode='reflect')

            # -- fwd [with mask....? idk copied from their code.] --
            n, c, h, w = noisy.shape
            net_input, mask = self.masker.train(noisy)
            noisy_output = (model(net_input,None)*mask).view(n, -1, c, h, w).sum(dim=1)
            dn_output = noisy_output.detach().clone()

            # -- back to original resolution --
            deno = dn_output[:, :, :H, :W]

        # -- reshape --
        deno = rearrange(deno,'(b t) c h w -> b t c h w',b=B)

        return deno

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#            Create a Mask
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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


def generate_mask(img, width=4, mask_type='random'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = th.zeros(size=(n * h // width * w // width * width**2, ),
                       dtype=th.int64,
                       device=img.device)
    idx_list = th.arange(
        0, width**2, 1, dtype=th.int64, device=img.device)
    rd_idx = th.zeros(size=(n * h // width * w // width, ),
                         dtype=th.int64,
                         device=img.device)

    if mask_type == 'random':
        th.randint(low=0,
                      high=len(idx_list),
                      size=(n * h // width * w // width, ),
                      device=img.device,
                      generator=get_generator(device=img.device),
                      out=rd_idx)
    elif mask_type == 'batch':
        rd_idx = th.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(h // width * w // width)
    elif mask_type == 'all':
        rd_idx = th.randint(low=0,
                               high=len(idx_list),
                               size=(1, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:
        index = mask_type.split('_')[-1]
        index = th.from_numpy(np.array(index).astype(
            np.int64)).type(th.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += th.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,
                                dtype=th.int64,
                                device=img.device)

    mask[rd_pair_idx] = 1

    mask = depth_to_space(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(th.int64)

    return mask

def depth_to_space(x, block_size):
    return th.nn.functional.pixel_shuffle(x, block_size)

def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = th.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = th.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv


class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = th.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = th.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = th.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


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


