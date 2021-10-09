import torch
import numpy as np
from torch import einsum
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff
from torch.autograd import Variable

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h, d = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h, d)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 3:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h, d = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h, d)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot2hd_dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                    dtype=None) -> np.ndarray:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap
    """
    # Relasx the assertion to allow computation live on only a
    # subset of the classes
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            res[k] = distance(posmask, sampling=resolution)

    return res


class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1, 2]  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        pc.requires_grad == True
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        # print('pc', pc)
        # print('dc', dc)

        multipled = einsum("bcwhd,bcwhd->bcwhd", pc, dc)

        loss = multipled.mean()

        return loss

class BoundaryLoss(nn.Module):
    def __init__(self) :
        super(BoundaryLoss, self).__init__()
    def forward(self, probs: Tensor, seg: Tensor) -> Tensor:
        seg = seg[:,0,:,:,:]          #bwHd
        seg2 = class2one_hot(seg, 3)   #bcwHd

        logits = F.softmax(probs, dim=1)  #bcwHd
        res = 0
        Loss = SurfaceLoss()
        for i in range(2):
            pred_seg = logits[i].data.max(0)[1] #wHd
            pred_seg1 = pred_seg.unsqueeze(0) #bwHd
            pred_seg2 = class2one_hot(pred_seg1, 3)  #bCwHd


            seg3 = seg2[i].cpu().numpy()  # chwd
            seg4 = one_hot2dist(seg3)
            seg4 = torch.tensor(seg4).unsqueeze(0)
            seg4 = seg4.cuda()

            res += Loss(pred_seg2, seg4).item()
        return res

class HausdorffLoss():
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self):
        super(HausdorffLoss, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing

    def __call__(self, probs, target):
        assert simplex(probs)
        #assert simplex(target)
        assert probs.shape == target.shape

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs.type(torch.float32))
        tc = cast(Tensor, target.type(torch.float32))
        #assert pc.shape == tc.shape == (B, len(self.idc), *xyz)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkwhd,bkwhd->bkwhd", delta, dtm)

        loss = multipled.mean()

        return loss


if __name__ == "__main__":
    data = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 2, 0, 0],
                           [0, 1, 1, 0, 2, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]]]])

    data2 = class2one_hot(data, 3)
    print(data2.shape)
    data2 = data2[0].numpy()
    data3 = one_hot2dist(data2)  # bcwh

    # print(data3)
    print("data3.shape:", data3.shape)

    logits = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 2, 2, 0, 0]]]])

    logits = class2one_hot(logits, 3)

    Loss = HausdorffLoss()
    data3 = torch.tensor(data3).unsqueeze(0)
    print(data3.shape)

    res = Loss(logits, data3)
    print('loss:', res)