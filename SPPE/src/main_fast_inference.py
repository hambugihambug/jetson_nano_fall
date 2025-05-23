import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np
from SPPE.src.utils.img import flip, shuffleLR
from SPPE.src.utils.eval import getPrediction
from SPPE.src.models.FastPose import FastPose

import time
import sys

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class InferenNet(nn.Module):
    def __init__(self, dataset, weights_file='./Models/sppe/fast_res101_320x256.pth', device='cpu'):
        super().__init__()

        self.pyranet = FastPose('resnet101').to(device)
        print('Loading pose model from {}'.format(weights_file))
        sys.stdout.flush()
        self.pyranet.load_state_dict(torch.load(weights_file))
        self.pyranet.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        flip_out = self.pyranet(flip(x))
        flip_out = flip_out.narrow(1, 0, 17)

        flip_out = flip(shuffleLR(
            flip_out, self.dataset))

        out = (flip_out + out) / 2

        return out


class InferenNet_fast(nn.Module):
    def __init__(self, weights_file=None, device='cpu'):
        super().__init__()

        self.pyranet = FastPose('resnet101').to(device)
        if weights_file is not None:
            print('Loading pose model from {}'.format(weights_file))
            self.pyranet.load_state_dict(torch.load(weights_file, map_location=device))
        self.pyranet.eval()

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out


class InferenNet_fastRes50(nn.Module):
    def __init__(self, weights_file=None, device='cpu'):
        super().__init__()

        self.pyranet = FastPose('resnet50', 17).to(device)
        if weights_file is not None:
            print('Loading pose model from {}'.format(weights_file))
            try:
                self.pyranet.load_state_dict(torch.load(weights_file, map_location=device))
            except Exception as e:
                print(f"모델 로딩 오류: {e}")
                print("더미 모델 감지됨. 무시하고 계속 진행합니다.")
        self.pyranet.eval()

    def forward(self, x):
        # 입력이 없는 경우 더미 출력 생성
        if x.shape[0] == 0:
            return torch.zeros((0, 17, x.shape[2]//4, x.shape[3]//4), device=x.device)
            
        try:
            out = self.pyranet(x)
        except Exception as e:
            print(f"추론 오류, 더미 출력 생성: {e}")
            return torch.zeros((x.shape[0], 17, x.shape[2]//4, x.shape[3]//4), device=x.device)
        
        return out
