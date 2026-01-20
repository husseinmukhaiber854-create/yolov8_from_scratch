import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
		super().__init__()
		if padding is None:
			padding = kernel_size // 2
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.act = nn.SiLU()

	def forward(self, x):
		return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 3)
        self.use_add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        if self.use_add:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n_bottlenecks = 1, 
                 shortcut = False):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.conv2 = Conv((2 + n_bottlenecks) * hidden_channels, out_channels, 1, 1)
        
        self.bottlenecks = nn.ModuleList(
            [Bottleneck(hidden_channels, hidden_channels, shortcut) 
                        for _ in range(n_bottlenecks)]
        )
    
    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend([m(y[-1]) for m in self.bottlenecks])
        return self.conv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, 
                                 padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))

class Detect(nn.Module):
    def __init__(self, num_classes = 80, channels = (256, 512, 1024)):
        super().__init__()
        self.num_classes = num_classes
        self.nl = len(channels)  # number of detection layers
        self.reg_max = 16  # DFL channels
        
        # Build strides
        self.stride = torch.ones(self.nl)
        
        # Build convolution layers
        c2 = max((16, channels[0] // 4, self.reg_max * 4))
        c3 = max(channels[0], self.num_classes)
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        for i in range(self.nl):
            # Classification branch
            self.cls_convs.append(
                nn.Sequential(
                    Conv(channels[i], c2, 3),
                    Conv(c2, c2, 3)
                )
            )
            self.cls_preds.append(nn.Conv2d(c2, self.num_classes, 1)
            )
            
            # Regression branch
            self.reg_convs.append(
                nn.Sequential(
                    Conv(channels[i], c2, 3),
                    Conv(c2, c2, 3)
                )
            )
            self.reg_preds.append(
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
    
    def forward(self, x):
        cls_outputs = []
        reg_outputs = []
        
        for i in range(self.nl):
            # Classification
            cls_feat = self.cls_convs[i](x[i])
            cls_output = self.cls_preds[i](cls_feat)
            cls_outputs.append(cls_output)
            
            # Regression
            reg_feat = self.reg_convs[i](x[i])
            reg_output = self.reg_preds[i](reg_feat)
            reg_outputs.append(reg_output)
        
        return cls_outputs, reg_outputs