
from torch.optim.lr_scheduler import LambdaLR

class WarmupConstantScheduler(LambdaLR):
    """Linear warmup + constant LR."""
    def __init__(self, optimizer, warmup_steps: int, last_epoch=-1):
        def lr_lambda(step):
            return min(1.0, step / max(1, warmup_steps))
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)