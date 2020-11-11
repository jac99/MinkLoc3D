# Author: Jacek Komorowski
# Warsaw University of Technology

import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from models.resnet import ResNetBase


class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64)):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks
        self.tconvs = nn.ModuleList()   # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                             dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - i], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(self.lateral_dim, self.lateral_dim, kernel_size=2,
                                                                stride=2, dimension=D))
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - self.num_top_down], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[0], self.lateral_dim, kernel_size=1,
                                                        stride=1, dimension=D))

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        feature_maps = []
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)     # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)        # Upsample using transposed convolution
            x = x + self.conv1x1[ndx+1](feature_maps[-ndx - 1])

        return x
