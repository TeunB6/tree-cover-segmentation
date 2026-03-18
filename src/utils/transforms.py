import torch
import torchvision.transforms.v2 as v2

class ToTensor(torch.nn.Module):
    def forward(self, img, label=None):
        # Do some transformations
        if label is not None:
            return (v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img), label)   
        else:
            return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img)