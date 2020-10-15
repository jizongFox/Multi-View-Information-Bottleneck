from training.multiview_infomax import MVInfoMaxTrainer
from utils.modules import IICEstimator
from utils.schedulers import ExponentialScheduler


###############
# MIB Trainer #
###############
class IICTrainer(MVInfoMaxTrainer):
    def __init__(self, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000, num_clusters=10, resample=False, **params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(IICTrainer, self).__init__(**params)
        self.mi_estimator = IICEstimator(self.z_dim, self.z_dim, num_clusters=num_clusters)
        self.opt.add_param_group(
            {'params': self.mi_estimator.parameters(), 'lr': params["miest_lr"]}
        )
        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)
        self.resample = resample

    def _compute_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        v1, v2, _ = data

        # Encode a batch of data
        z1 = self.encoder_v1(v1).mean
        z2 = self.encoder_v2(v2).mean

        if self.resample:
            z1 = self.encoder_v1(v1).rsample()
            z2 = self.encoder_v2(v2).rsample()

        # Mutual information estimation
        mi_gradient = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()

        # Symmetrized Kullback-Leibler divergence
        # there is no estimator for this since it is supposed to be gaussian
        # kl_1_2 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1)
        # kl_2_1 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
        # skl = (kl_1_2 + kl_2_1).mean() / 2.

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/IIC', mi_gradient.item())
        # self._add_loss_item('loss/SKL_z1_z2', skl.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = mi_gradient  # + beta * skl

        return loss
