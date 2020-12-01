from torch import nn


class ResnetC4(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(resnet.children())[:-3]
        )

    def forward(self, x):
        x = self.features(x)
        return x
