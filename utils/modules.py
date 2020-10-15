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
        iic_loss = self._iic_mi(p1, p2)
        return iic_loss

    def _iic_mi(self, p1, p2):
        _, k = p1.size()
        assert self.simplex(p1) and self.simplex(p2)
        p_i_j = self._joint(p1, p2)
        p_i = (
            p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        )  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        # p_i = x_out.mean(0).view(k, 1).expand(k, k)
        # p_j = x_tf_out.mean(0).view(1, k).expand(k, k)
        #

        loss = -p_i_j * (
            torch.log(p_i_j + 1e-10) - self.lamb * torch.log(p_j + 1e-10) - self.lamb * torch.log(p_i + 1e-10)
        )
        loss = loss.sum()

        return loss

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
