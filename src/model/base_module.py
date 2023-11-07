from torch import nn

class TorchModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _scale_loss(self,loss_dict):
        # Scale the losses and add them up
        loss = None
        for key in loss_dict.keys():
            current = loss_dict[key] * self.loss_scaling.get(key,1.0)
            if loss is None:
                loss = current
            else:
                loss += current
        return loss

