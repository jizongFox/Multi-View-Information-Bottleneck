from typing import Union, List, Optional, OrderedDict, Dict

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class Decoder(nn.Module):
    def __init__(self, z_dim, scale=0.39894):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.scale = scale

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28)
        )

    def forward(self, z):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)


# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


# IIC estimator
class IICEstimator(nn.Module):
    def __init__(self, size1, size2, num_clusters=10, projector_type="linear", lamda=1.0) -> None:
        super().__init__()
        self.lamb = float(lamda)
        assert projector_type in ("mlp", "linear"), projector_type
        if projector_type == "linear":
            self._projector = nn.Sequential(
                nn.Linear(size1, num_clusters),
                nn.Softmax(1)
            )
        else:
            self._projector = nn.Sequential(
                nn.Linear(size1, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_clusters),
                nn.Softmax(1)
            )

    def forward(self, z1, z2):
        p1, p2 = torch.chunk(self._projector(torch.cat([z1, z2], dim=0)), 2)
        iic_loss, p_i_j = self._iic_mi(p1, p2)
        return iic_loss, (p1, p2), p_i_j

    def _iic_mi(self, p1, p2):
        _, k = p1.size()
        assert self.simplex(p1) and self.simplex(p2)
        p_i_j = self._joint(p1, p2)
        p_i = (
            p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        )  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        loss = -p_i_j * (
            torch.log(p_i_j + 1e-10) - self.lamb * torch.log(p_j + 1e-10) - self.lamb * torch.log(p_i + 1e-10)
        )
        loss = loss.sum()

        return loss, p_i_j

    def _joint(self, p1, p2):

        assert self.simplex(p1), f"x_out not normalized."
        assert self.simplex(p2), f"x_tf_out not normalized."

        bn, k = p1.shape
        assert p1.size(0) == bn and p2.size(1) == k

        p_i_j = p1.unsqueeze(2) * p2.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k aggregated over one batch

        p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetric
        p_i_j /= p_i_j.sum()  # normalise

        return p_i_j

    @staticmethod
    def simplex(probs, dim=1) -> bool:
        sum = probs.sum(dim)
        return torch.allclose(sum, torch.ones_like(sum))


class KL_div(nn.Module):
    r"""
    KL(p,q)= -\sum p(x) * log(q(x)/p(x))
    where p, q are distributions
    p is usually the fixed one like one hot coding
    p is the target and q is the distribution to get approached.

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16, weight: Union[List[float], torch.Tensor] = None, verbose=True):
        super().__init__()
        self._eps = eps
        self._reduction = reduction
        self._weight: Optional[torch.Tensor] = weight
        if weight is not None:
            assert isinstance(weight, (list, torch.Tensor)), type(weight)
            if isinstance(weight, list):
                self._weight = torch.Tensor(weight).float()
            else:
                self._weight = weight.float()
            # normalize weight:
            self._weight = self._weight / self._weight.sum() * len(self._weight)
        if verbose:
            print(
                f"Initialized {self.__class__.__name__} \nwith weight={self._weight} and reduction={self._reduction}.")

    def forward(self, prob: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        if not kwargs.get("disable_assert"):
            assert prob.shape == target.shape
        b, c, *hwd = target.shape
        kl = (-target * torch.log((prob + self._eps) / (target + self._eps)))
        if self._weight is not None:
            assert len(self._weight) == c
            weight = self._weight.expand(b, *hwd, -1).transpose(-1, 1).detach()
            kl *= weight.to(kl.device)
        kl = kl.sum(1)
        if self._reduction == "mean":
            return kl.mean()
        elif self._reduction == "sum":
            return kl.sum()
        else:
            return kl

    def __repr__(self):
        return f"{self.__class__.__name__}\n, weight={self._weight}"

    def state_dict(self, *args, **kwargs):
        save_dict = super().state_dict(*args, **kwargs)
        save_dict["weight"] = self._weight
        save_dict["reduction"] = self._reduction
        return save_dict

    def load_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]], *args,
                        **kwargs):
        self._reduction = state_dict["reduction"]
        self._weight = state_dict["weight"]
