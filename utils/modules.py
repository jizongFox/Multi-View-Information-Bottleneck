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


# InfoNCE estimator
class InfoNCEEstimator(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(InfoNCEEstimator, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # SIMCLR
        elif labels is not None:
            if isinstance(labels, list):
                labels = torch.Tensor(labels).long()
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.t()).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 32 128
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.t()),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-16)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# InfoNCE projector
class NCEProjector(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self._z_dim = z_dim
        self._projector = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, z_dim)
        )

    def forward(self, z):
        return self._projector(z)


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
