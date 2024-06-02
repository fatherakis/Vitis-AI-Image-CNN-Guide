import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="/path/to/imagenet/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="/path/to/trained_model/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: '
)
parser.add_argument(
    '--model_name',
    default="resnet20_cifar",
    help='Name of the model to be loaded'
)
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')
parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()

#--------------- MODEL DEFINITIONS ------------------------------------------------------------------

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional
from torch.ao.nn.quantized.modules.functional_modules import FloatFunctional

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.ff = torch.nn.quantized.FloatFunctional()
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.ff.add(out, identity)
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

resnet20_model_cifar = CifarResNet(BasicBlock,[3]*3)

#--------------- END MODEL DEFINITIONS ---------------------------------------


def load_data(data_dir='dataset/imagenet',
              batch_size=128,
              **kwargs):

  #prepare data
  # random.seed(12345)
  valdir = data_dir + '/val'
  normalize = transforms.Normalize(
      mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
  dataset = torchvision.datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.ToTensor(),
          normalize,
      ]))
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False, **kwargs)
  return data_loader


def evaluate(model, val_loader):

    model.eval()
    model = model.to(device)
    for iteration, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        labels = labels.to(device)
        #pdb.set_trace()
        outputs = model(images)


def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

  data_dir = args.data_dir
  quant_mode = args.quant_mode
  deploy = args.deploy
  batch_size = args.batch_size
  inspect = args.inspect
  config_file = None
  target = args.target
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = resnet20_model_cifar.cpu()
  model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

  #change only the shape of the data
  input = torch.randn([batch_size, 3, 32, 32])
  ####################################################################################
  # This function call will create a quantizer object and setup it. 
  # Eager mode model code will be converted to graph model. 
  # Quantization is not done here if it needs calibration.
  quantizer = torch_quantizer(
      quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

  # Get the converted model to be quantized.
  quant_model = quantizer.quant_model
  #####################################################################################


  val_loader, _ = load_data(
      batch_size=batch_size,
      data_dir=data_dir,
      )
   
  #calibration
  evaluate(quant_model, val_loader)


  # handle quantization result
  if quant_mode == 'calib':
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
    quantizer.export_quant_config()
  if deploy:
    #quantizer.export_torch_script()
    #quantizer.export_onnx_model()
    quantizer.export_xmodel()


if __name__ == '__main__':

  model_name = args.model_name
  file_path = os.path.join(args.model_dir, model_name + '.pkl')

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path)

  print("-------- End of {} test ".format(model_name))
