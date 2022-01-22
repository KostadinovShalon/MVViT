from mmdet.models import NECKS, ChannelMapper


@NECKS.register_module()
class MVChannelMapper(ChannelMapper):

    def forward(self, inputs):
        """Forward function."""
        views = inputs[0].size(1)
        mv_inputs = [[inp[:, v] for inp in inputs] for v in range(views)]
        mv_outputs = []
        for sv_inputs in mv_inputs:
            sv_outs = super().forward(sv_inputs)  # tuple of sv outputs
            mv_outputs.append(sv_outs)
        return mv_outputs
