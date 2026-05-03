import torch.nn as nn

class Conv2dBatchPPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            PPReLU(self.out_channels)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x



# mobilenet
class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, isPadding=True, isBias=False):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.isBias = isBias
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        if isPadding == True:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 0, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )


    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        #print("*******************************begin")
        #print(self.layers[0].weight)
        #print(self.layers[1].weight)
        #print(self.layers[1].bias)
       # print(self.layers[1].running_mean)
        #print(self.layers[1].running_var)
        #print("******************************end")
        x = self.layers(x)
        #print("x: ", x.size())
        #print("x.size", x.size())
       # print("*************************", self.__class__.__name__)
        #print("************************** Conv2dBatchReLU weight begin")
        #print("***", self.layers[2])

        return x



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, isBias=False):
        super(InvertedResidual,self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.expand_ratio = expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup
        self.isBias = isBias
        #print("InvertedResidual in : ", inp, oup, expand_ratio)
        if abs(expand_ratio - 1) < .01:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=self.isBias),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=self.isBias),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=self.isBias),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=self.isBias),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=self.isBias),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        #print("in", x.size())
        if self.use_res_connect:
            out =  x + self.conv(x)
        else:
            out = self.conv(x)
        return out 


def buildInvertedResBlock(residual_setting, input_channel):
    # building inverted residual blocks
    features = []
    t, c, n, s = residual_setting
    output_channel = int(c)
    layerNameList = []
    weightList = []
    #print("******************************************\n")
    for i in range(n):
        print("channel: ", input_channel, output_channel)
        if i == 0:
            curLayer = InvertedResidual(input_channel, output_channel, s, t)
            #features.append(InvertedResidual(input_channel, output_channel, s, t))
            features.append(curLayer)
          #  curLayerNameList, curweightList = curLayer.toCaffe()
        else:
            curLayer = InvertedResidual(input_channel, output_channel, 1, t)
            #features.append(InvertedResidual(input_channel, output_channel, 1, t))
            features.append(curLayer)
           # curLayerNameList, curweightList = curLayer.toCaffe()
        input_channel = output_channel
    layers = nn.Sequential(*features)
    return layers, output_channel