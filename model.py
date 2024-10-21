import torchvision
import torch
from torch import nn

def create_effnetb2_model(num_class:int, seed:int=42):
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights=weights)

  model.transforms = transforms
  for param in model.parameters():
    param.requires_grad = False


  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1408, out_features=num_class)
  )
  return model, transforms
