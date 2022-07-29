import torch
import torch_dct as dct

class LowFreqPenaltyLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(delta,spatial_shape=(32,32)):
        if len(spatial_shape) == 2:
            res = dct.dct_2d(delta)
            a = spatial_shape[0]//4
            b = spatial_shape[1]//4
            return (res[:,:,:a,:b].abs()).mean()
        
        else:
            raise ValueError(f'LowFreqPenaltyLoss not implemented for spatial_shape: {spatial_shape}')


class HighFreqPenaltyLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(delta,spatial_shape=(32,32)):
        if len(spatial_shape) == 2:
            res = dct.dct_2d(delta)
            a = spatial_shape[0]//4
            b = spatial_shape[1]//4
            return (res[:,:,-a:,-b:].abs()).mean()
        
        else:
            raise ValueError(f'HighFreqPenaltyLoss not implemented for spatial_shape: {spatial_shape}')