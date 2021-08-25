from mmcv.runner import HOOKS, Hook
import torch


@HOOKS.register_module()
class GradHook(Hook):

    def after_train_iter(self, runner):
        for name, param in runner.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = torch.norm(param.grad.detach(), 2.0)
                runner.log_buffer.update({name.split(".", 1)[1]: float(g)}, runner.outputs['num_samples'])
        runner.optimizer.step()
