from mmdet.models import ResNet, BACKBONES
import torch


@BACKBONES.register_module()
class MVResNet(ResNet):
    def forward(self, x):
        """Forward function."""
        # x is now B x V x C x W x H
        x = x.transpose(0, 1)
        mv_outs = []  # list of V items of N items of B x C' x W' x H' tensors
        for xv in x:
            mv_outs.append(super().forward(xv))
        outs = []
        for i in self.out_indices:
            single_layer_mv_out = [sv_outs[i] for sv_outs in mv_outs]
            single_layer_mv_out = torch.stack(single_layer_mv_out).transpose(0, 1)
            outs.append(single_layer_mv_out)
        return tuple(outs)
