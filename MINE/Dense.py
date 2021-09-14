import torch
import torch.nn as nn

#This file contains the DenseBlock and the Transition block

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))
    def forward(self, input):
        output = self.conv(self.relu(self.norm(input)))
        output = self.pool(output)
        return output

class _PreBlock(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_PreBlock, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=3, stride=1, bias=False))
        self.add_module('pool', nn.MaxPool3d(kernel_size=2, stride=2))

    def forward(self, input):
        output = self.conv(self.relu(self.norm(input)))
        output = self.conv(self.relu(self.norm(output)))
        output = self.pool(output)
        return output



class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        "Bottleneck function"
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_Denseblock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(('denselayer{}'.format(i + 1), layer))

    def forward(self, input):
        output = self.layers(input)
        return output

class _Upsample(nn.Module):
    def __init__(self, num_input_features, bn_size, growth_rate, drop_rate):
        super(_Upsample, self).__init__()
        self.add_module('tran', nn.ConvTranspose3d(num_input_features,
                                                   num_input_features * 2,
                                                   kernel_size=2, stride=1,
                                                   bias=False, dilation=1))
        num_input_features = num_input_features * 2
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1',  nn.Conv3d(num_input_features, num_input_features // 2, kernel_size=1, stride=1,
                                           bias=False))
        self.add_module('dense', _DenseBlock(num_layers=2,
                                    num_input_features=num_input_features,
                                    bn_size=self.bn_size,
                                    growth_rate=self.growth_rate,
                                    drop_rate=self.drop_rate))
        num_input_features = num_input_features * growth_rate
        self.add_module('norm2', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2',  nn.Conv3d(num_input_features, num_input_features // 2, kernel_size=1, stride=1,
                                           bias=False))

    def forward(self, input, previous):
        output = self.tran(input)
        # I want to concatenate the channels (CJECK)
        output = torch.cat((output, previous), dim=1)
        output = self.conv1(self.relu1(self.norm1(output)))
        output = self.dense(output)
        output = self.conv2(self.relu2(self.norm2(output)))
        return output
