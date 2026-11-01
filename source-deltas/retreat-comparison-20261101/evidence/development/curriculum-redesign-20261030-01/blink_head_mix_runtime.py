"""２つの固定眼判定器のログイットを固定割合で混合する。"""
import torch


class MixedHead(torch.nn.Module):
    def __init__(self, first, second, alpha):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError('混合割合が範囲外です')
        self.first = first
        self.second = second
        self.alpha = alpha

    def forward(self, x):
        return (1-self.alpha)*self.first(x)+self.alpha*self.second(x)
