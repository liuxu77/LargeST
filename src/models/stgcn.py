import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.base.model import BaseModel

class STGCN(BaseModel):
    '''
    Reference code: https://github.com/hazdzz/STGCN
    '''
    def __init__(self, gso, blocks, Kt, Ks, dropout, **args):
        super(STGCN, self).__init__(**args)
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(Kt, Ks, self.node_num, blocks[l][-1], blocks[l+1], gso, dropout))
        self.st_blocks = nn.Sequential(*modules)
        Ko = self.seq_len - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], self.node_num, dropout)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0])
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0])
            self.relu = nn.ReLU()


    def forward(self, x, label=None):  # (b, t, n, f)
        x = x.permute(0, 3, 1, 2)
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x.transpose(2, 3)


class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, node_num, last_block_channel, channels, gso, dropout):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], node_num)
        self.graph_conv = GraphConvLayer(channels[0], channels[1], Ks, gso)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], node_num)
        self.tc2_ln = nn.LayerNorm([node_num, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, node_num, dropout):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], node_num)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1])
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel)
        self.tc1_ln = nn.LayerNorm([node_num, channels[0]])
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, node_num):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.node_num = node_num
        self.align = Align(c_in, c_out)
        self.sigmoid = nn.Sigmoid()
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, \
                            kernel_size=(Kt, 1), enable_padding=False, dilation=1)


    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), self.sigmoid(x_q))
        return x


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super(GraphConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso)


    def forward(self, x):
        x_gc_in = self.align(x)
        x_gc = self.cheb_graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        return x_gc_out


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)


    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))

        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])
        
        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        return cheb_graph_conv


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))


    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, node_num = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, node_num]).to(x)], dim=1)
        else:
            x = x
        return x


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)


    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result