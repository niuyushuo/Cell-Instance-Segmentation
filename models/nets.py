from torch.optim import lr_scheduler


def get_scheduler(optimizer, max_epochs):
    def lambda_rule(epoch):
        return 1.0 - epoch / float(max_epochs + 1)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
