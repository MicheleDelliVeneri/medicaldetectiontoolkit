import torch.nn as nn
import torch.nn.functional as F
import torch
import Dense as dense

class UDN(nn.Module):
    """
    Dense U-Net
    """
    def __init__(self,  conv_size, conv_stride,
                self.num_input_channels=1,
                 growth_rate=32, num_input_features=32,
                 bn_size=4, drop_rate=0):
        super(UDN, self).__init__()
        self.num_input_channels = num_input_channels
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.growth_rate = grow_rate
        self.num_input_features = num_input_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate

        # Down Path

        self.P0 = dense._PreBlock(num_input_features=self.num_input_channels,
                                        num_output_features=self.num_input_features)
        self.num_dense_layers = 1
        # This is the first block out of 3
        self.C0 = dense._DenseBlock(num_layers=2,
                                    num_input_features=self.num_input_features,
                                    bn_size=self.bn_size,
                                    growth_rate=self.growth_rate,
                                    drop_rate=self.drop_rate)
        self.num_input_features = self.num_input_features + self.num_dense_layers * self.growth_rate
        self.T0 = dense._Transition(num_input_features=self.num_input_features,
                                    num_output_features=self.num_input_features // 2)
        self.num_dense_layers += 1
        self.num_input_features = self.num_input_features // 2

        # this is the second block out of 3
        self.C1 = dense._DenseBlock(num_layers=2,
                                    num_input_features=self.num_input_features,
                                    bn_size=self.bn_size,
                                    growth_rate=self.growth_rate,
                                    drop_rate=self.drop_rate)
        self.num_input_features = self.num_input_features + self.num_dense_layers * self.growth_rate
        self.T1 = dense._Transition(num_input_features=self.num_input_features,
                                    num_output_features=self.num_input_features // 2)
        self.num_dense_layers += 1
        self.num_input_features = self.num_input_features // 2

        # this is the third block out of 3
        self.C2 = dense._DenseBlock(num_layers=2,
                                    num_input_features=self.num_input_features,
                                    bn_size=self.bn_size,
                                    growth_rate=self.growth_rate,
                                    drop_rate=self.drop_rate)
        self.num_input_features = self.num_input_features + self.num_dense_layers * self.growth_rate
        self.T2 = dense._Transition(num_input_features=self.num_input_features,
                                    num_output_features=self.num_input_features // 2)
        # T2 goes into first RPN head
        self.num_dense_layers += 1
        self.num_input_features = self.num_input_features // 2

        # Up parth
        #Add upsample 
