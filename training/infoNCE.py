import torch

from training.base import RepresentationTrainer
from utils.modules import InfoNCEEstimator


######################
# InfoNCE Trainer #
######################
class InfoNCETrainer(RepresentationTrainer):
    def __init__(self, miest_lr=1e-4, resample=False, **params):
        super(InfoNCETrainer, self).__init__(**params)

        # Initialization of the mutual information estimation network
        self.mi_estimator = InfoNCEEstimator(temperature=0.07, base_temperature=0.07)
        self.encoder_v1 = self.encoder
        self.encoder_v2 = self.encoder_v1

        # Adding the parameters of the estimator to the optimizer
        self.opt.add_param_group(
            {'params': self.mi_estimator.parameters(), 'lr': miest_lr}
        )
        self._resample = resample

    def _get_items_to_store(self):
        items_to_store = super(InfoNCETrainer, self)._get_items_to_store()

        # Add the mutual information estimator parameters to items_to_store
        items_to_store['mi_estimator'] = self.mi_estimator.state_dict()
        return items_to_store

    def _compute_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        v1, v2, _ = data

        # Encode a batch of data
        p_z1_given_v1 = self.encoder_v1(v1)
        p_z2_given_v2 = self.encoder_v2(v2)

        z1 = p_z1_given_v1.mean
        z2 = p_z2_given_v2.mean

        # Sample from the posteriors with reparametrization
        if self._resample:
            z1 = p_z1_given_v1.rsample()
            z2 = p_z2_given_v2.rsample()

        # Mutual information estimation
        mi_estimation = self.mi_estimator(torch.stack([z1, z2], dim=1))

        # Logging Mutual Information Estimation
        self._add_loss_item('loss/infonce', mi_estimation.item())

        # Computing the loss function
        loss = mi_estimation

        return loss
