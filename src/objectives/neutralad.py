import math
import torch
import numpy as np

from viewmaker.src.utils.utils import l2_normalize


class NeuTraLADLoss(object):

    def __init__(self, outputs_orig, outputs_views, t=0.07):
        super().__init__()
        self.outputs = l2_normalize(outputs_views, dim=2)
        self.outputs_orig = l2_normalize(outputs_orig, dim=1)
        self.t = t

    def get_loss(self, debug=False):
        # https://arxiv.org/pdf/2103.16440.pdf

        batch_size = self.outputs_orig.size(0)  # batch_size x out_dim
        num_views = self.outputs.size(0)

        sim_x_x_tags = torch.sum(self.outputs * self.outputs_orig, dim=-1).T / self.t  # [256]
        # sim_x_tags_x_tags = torch.bmm(self.outputs.permute(1, 0, 2),
        #                               self.outputs.permute(1, 2, 0)) / self.t
        x_x_tags = torch.cat([self.outputs_orig.unsqueeze(0), self.outputs]).permute(1,0,2)
        ######
        # currently similarty of x and its views appear in the denominator, consider removing it
        ######
        sim_x_tags_x_tags = torch.bmm(x_x_tags, x_x_tags.permute(0,2,1))
        # set the lower triangle to huge negative (when l<=k)
        # this is done so it doesn't effect the sum (goes to 0 when exponentiated)
        mask = torch.tril(torch.ones_like(sim_x_tags_x_tags, device=self.outputs.device)) * 1e20
        sim_x_tags_x_tags = sim_x_tags_x_tags * (1 - mask)
        sim_x_tags_x_tags = sim_x_tags_x_tags[:,:,0]
        # sim_x_tags_x_tags = sim_x_tags_x_tags.reshape(batch_size, -1)

        sim_x_x = self.outputs_orig @ self.outputs_orig.T
        sim_x_x = sim_x_x * (1 - torch.eye(*sim_x_x.shape, device=self.outputs.device) * 1e20)
        denom_sim = torch.cat([sim_x_x, sim_x_tags_x_tags], dim=1)

        c = denom_sim.max()  # stabilization factor
        denom = c + torch.log(torch.sum(torch.exp(denom_sim - c), dim=-1))  # [256]

        loss = -torch.mean((sim_x_x_tags.sum(dim=-1) - denom))
        if debug:
            return loss, {"dcl_denom": denom,
                          "sim_x_x_tags": sim_x_x_tags.sum(dim=-1, keepdim=True),
                          "sim_x_tags_x_tags": sim_x_tags_x_tags,
                          # "sim_x_tags_x_tags_with_x_sim": sim_x_tags_x_tags_with_x_sim
                          }
        return loss


# with negative examples kept for reference contains some errors
class OldNeuTraLADLoss(object):

    def __init__(self, outputs_orig, outputs_views, t=0.07):
        super().__init__()
        self.outputs = l2_normalize(outputs_views, dim=2)
        self.outputs_orig = l2_normalize(outputs_orig, dim=1)
        self.t = t

    def get_loss(self, debug=False):
        # https://arxiv.org/pdf/2103.16440.pdf

        batch_size = self.outputs_orig.size(0)  # batch_size x out_dim
        num_views = self.outputs.size(0)

        sim_x_x_tags = torch.sum(self.outputs * self.outputs_orig, dim=-1).T / self.t  # [256]
        # sim_x_tags_x_tags = torch.bmm(self.outputs.permute(1, 0, 2),
        #                               self.outputs.permute(1, 2, 0)) / self.t
        x_x_tags = torch.cat([self.outputs_orig, self.outputs.flatten(0, 1)], dim=0)
        ######
        # currently similarty of x and its views appear in the denominator, consider removing it
        ######
        sim_x_tags_x_tags = self.outputs_orig @ x_x_tags.T


        # set the main diagonal to huge negative (when l==k in the NeuTraLAD paper)
        # this is done so it doesn't effect the sum (goes to 0 when exponentiated)
        sim_x_tags_x_tags = sim_x_tags_x_tags * (1 - torch.eye(*sim_x_tags_x_tags.shape, device=self.outputs.device) * 1e20)
        # sim_x_tags_x_tags_with_x_sim = torch.cat([sim_x_x_tags, sim_x_tags_x_tags], dim=-1)  # [256, 2]
        # dcl_denom = torch.log(torch.sum(torch.exp(sim_x_tags_x_tags_with_x_sim), dim=-1))  # [256]

        c = sim_x_tags_x_tags.max()  # stabilization factor
        dcl_denom = c + torch.log(torch.sum(torch.exp(sim_x_tags_x_tags - c), dim=-1))  # [256]

        loss = -torch.mean((sim_x_x_tags.sum(dim=-1, keepdim=True) - dcl_denom).sum(dim=1))
        if debug:
            return loss, {"dcl_denom": dcl_denom,
                          "sim_x_x_tags": sim_x_x_tags.sum(dim=-1, keepdim=True),
                          "sim_x_tags_x_tags": sim_x_tags_x_tags,
                          # "sim_x_tags_x_tags_with_x_sim": sim_x_tags_x_tags_with_x_sim
                          }
        return loss

if __name__ == '__main__':
    x = torch.Tensor([[1, 1]])
    X_ = torch.ones(12,1,2)
    nad_loss = NeuTraLADLoss(x, X_, t=0.02)
    crit = nad_loss.get_loss()
    # assert np.isclose(crit.item(), -2 * np.log(0.5), rtol=1e-4)
    # print("test passed")
    # nad = NeuTraLADLoss(torch.zeros(1, 2) + 1, torch.zeros(1, 2, 2) + 1, t=1)
