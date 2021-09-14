import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# def conv_relu(x, filters, kernel, padding='same',use_bias = True,  dilation_rate=1, strides=(1,1)):
#     channel = x.size(1)
#     if padding == 'same':
#         if dilation_rate == 0:
#            # conv = nn.conv2D(channel, 1,kernel,padding = padding,dilation =0,stride = strides)
#             conv = nn.conv2D(channel, filters,kernel,padding = (kernel-1)/2,bias=use_bias, dilation =0,             stride = strides)
#         else:
#             conv = nn.conv2D(channel, filters,kernel,padding = (kernel-1)/2,bias=use_bias, dilation =dilation_rate,stride = strides)
#     else :
#         if dilation_rate == 0:
#             conv = nn.conv2D(channel, filters,kernel,padding = 0,           bias=use_bias, dilation =0,             stride = strides)
#         else:
#             conv = nn.conv2D(channel, filters,kernel,padding = 0,           bias=use_bias, dilation =dilation_rate,stride = strides)
#
#     return nn.ReLU(conv(x))

class conv_relu(nn.Module):
    def __init__(self, channel, filters, kernel, padding='same', use_bias = True, dilation_rate=1, strides=(1,1)):
        super(conv_relu, self).__init__()
        # self.channel = x.size(1)
        self.channel = channel
        self.filters = filters
        self.kernel = kernel
        self.padding = padding
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.strides = strides[0]
        self.relu = nn.ReLU()

        self.kenerl_2 = self.kernel + ( dilation_rate-1 * (self.kernel-1) )

        if self.kernel == 3:
            if self.padding == 'same':
                if self.strides ==1:
                    if self.dilation_rate == 1:
                        self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=1, bias=self.use_bias,
                                         stride=strides)
                    elif self.dilation_rate == 2:
                        self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=2  , bias=self.use_bias,
                                         dilation=dilation_rate, stride=strides)
                    elif self.dilation_rate == 3:
                        self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding= 3  , bias=self.use_bias,
                                         dilation=dilation_rate, stride=strides)
                elif self.strides ==2:
                    if self.dilation_rate == 1:
                        self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding= 1  , bias=self.use_bias,
                                         dilation=dilation_rate, stride=strides)
            elif self.padding == 'valid':
                if self.strides ==2:
                    self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding= 0  , bias=self.use_bias,
                                     dilation=dilation_rate, stride=strides)

        elif self.kernel == 1:
            if self.padding =='same':
                if self.strides == 1:
                    if self.dilation_rate==1:
                        self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=0, bias=self.use_bias,
                                         stride=strides)


    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y



# def conv(x, filters, kernel, padding='same', use_bias=True, dilation_rate=1, strides = (1,1)):
#     channel = x.size(1)
#
#     if padding == 'same':
#         conv = nn.Conv2d(channel, filters, kernel, padding=(kernel-1)/2, bias=use_bias, dilation=dilation_rate, stride=strides)
#     return conv(x)


class conv(nn.Module):
    def __init__(self,channel, filters, kernel, padding='same', use_bias=True, dilation_rate=1, strides = (1,1)):
        super().__init__()
        self.channel = channel
        # self.x = x
        self.filters = filters
        self.kernel = kernel
        self.padding = padding
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.strides = strides

        if self.padding == 'same':
            self.conv = nn.Conv2d(self.channel, self.filters, self.kernel, padding=(self.kernel-1)//2,
                                 bias=self.use_bias, dilation=self.dilation_rate, stride=self.strides)

    def forward(self, x):
        y= self.conv(x)
        return y


# def conv_bn_relu(x, filters, kernel, padding='same', use_bias = True, dilation_rate=1):
#     return nn.ReLU(conv(x))
# def conv_prelu(x, filters, kernel, padding='same', use_bias=False, dilation_rate=1, strides = (1,1)):
#     return x


####
class pre_block(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self,x, d_list, enable = True):
        super().__init__()
        self.t = x
        self.nFilters = 64
        self.enable = enable

        #128,192,256,320,384
        self.conv_func1 = conv_relu(128, self.nFilters, 3, dilation_rate=d_list[0]) # 사이즈유지
        self.conv_func2 = conv_relu(192, self.nFilters, 3, dilation_rate=d_list[1])
        self.conv_func3 = conv_relu(256, self.nFilters, 3, dilation_rate=d_list[2])
        self.conv_func4 = conv_relu(320, self.nFilters, 3, dilation_rate=d_list[3])
        self.conv_func5 = conv_relu(384, self.nFilters, 3, dilation_rate=d_list[4])

        self.conv1 = conv(448, self.nFilters, 3)
        self.conv2 = conv(64, self.nFilters*2, 1)

        # self.adaptive_implicit_trans = adaptive_implicit_trans()(t)# 여기 바꾸기

        # self.ScaleLayer = ScaleLayer(s=0.1)(t) # 이거해야함

    def forward(self, x):
        t=x
        # print('\nthis is t.shape',t.shape)
        _t = self.conv_func1(t )

        # print('t.shape',t.shape)
        # print('_t.shape',_t.shape)
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_func2(t )


        # print('136t.shape',t.shape)
        # print('127_t.shape',_t.shape)
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_func3(t )
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_func4(t )
        t = torch.cat( [_t, t ] , dim=-3)
        _t = self.conv_func5(t )
        t = torch.cat( [_t, t ] , dim=-3)

        t = self.conv1(t)
        # t = self.adaptive_implicit_trans(t)

        t = self.conv2(t)
        # t = self.ScaleLayer(t)

        if not self.enable:
            t = layers.Lambda(lambda x: x*0)(t) # 여기 바꾸기

        t = torch.add( x,t )

        return t


####
class pos_block(nn.Module):
    def __init__(self,x, d_list):
        super().__init__()
        self.t = x
        self._t = x
        self.nFilters = 64

        self.conv_func1 = conv_relu(128, self.nFilters, 3, dilation_rate=d_list[0])
        self.conv_func2 = conv_relu(192, self.nFilters, 3, dilation_rate=d_list[2])
        self.conv_func3 = conv_relu(256, self.nFilters, 3, dilation_rate=d_list[1])
        self.conv_func4 = conv_relu(320, self.nFilters, 3, dilation_rate=d_list[3])
        self.conv_func5 = conv_relu(384, self.nFilters, 3, dilation_rate=d_list[4])
        self.conv_func_last = conv_relu(448, self.nFilters*2, 1, dilation_rate=d_list[4]) # 448

    def forward(self, x):
        t=x
        _t = self.conv_func1(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 여기서는 channel을 concat해야함. c,h,w

        _t = self.conv_func2(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel

        _t = self.conv_func3(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel

        _t = self.conv_func4(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel

        _t = self.conv_func5(t)
        t = torch.cat( [_t,t], dim=-3 ) # Default는 맨앞 아마channel

        t = self.conv_func_last(t)
        return t


####
class global_block(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.size = (x+2) // 2

        self.nFilters = 64
        self.conv_func1 = conv_relu(128, self.nFilters*4, 3, strides=(2,2))
        self.GlobalAveragePooling2D = nn.AvgPool2d((self.size,self.size))

        self.dense1 = nn.Linear(256, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)

        self.conv_func2 = conv_relu(128, self.nFilters*4, 1)
        self.conv_func3 = conv_relu(256, self.nFilters*2, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # print('\nbefore padding 223   ',x.shape) # 128, 128, 128
        t = F.pad(x, (1,1,1,1))
        # print('after padding  224 224 ',t.shape) # 128,130,130,
        t = self.conv_func1(t)
        # print('Befroe Global average pooling 231  ',t.shape) # 256,65,65

        # height, width = t.shape[-2:]
        # Averagepooling = nn.AvgPool2d((height,width))

        # t = self.GlobalAveragePooling2D(t)
        t = self.GlobalAveragePooling2D(t)
        # print('After Global average pooling ',t.shape)  #4, 256,1,1
        # t = torch.squeeze(t)
        t = t.squeeze(dim=2)
        t = t.squeeze(dim=2)
        # print('After squeeze ',t.shape)             #4, 256

        t = self.dense1(t)
        # print('After dense1 ',t.shape)      #4, 1024
        t =self.relu(t)
        # print('After dense2 ',t.shape)      #4, 1024
        t = self.dense2(t)
        t =self.relu(t)
        t = self.dense3(t)


        # print('\ninput of conv function = ',x.shape)        #4,128,128,128
        _t = self.conv_func2(x)
        # print('after conv_func = ',_t.shape)                #256,128,128
        # print('input of mul = ',_t.shape)   #4,256,128,128
        # print('input of mul = ',t.shape)       #4,256
        t = t.unsqueeze(dim=2)
        # print('unsqueeze of t = ',t.shape)       #4,256
        t = t.unsqueeze(dim=2)
        # print('unsqueeze of t = ',t.shape)       #4,256

        _t = torch.mul(_t,t)
        # _t = torch.mv()
        # print('\ninput of conv function second = ',_t.shape) # 256,128,128
        # exit()
        # self._t = x.mul(_t,self.t.reshape(-1,-1,1,1))
        _t = self.conv_func3(_t)
        # print('after conv_func second = ', _t.shape) # 128,128,128  512,512,128
        return _t


class adaptive_implicit_trans(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        conv_shape = (64, 64, 1, 1)
        self.it_weights = nn.Parameter( torch.ones( (64, 1, 1, 1) ) )

        # self.it_weights = self.add_weight(
        #     shape = (64,1,1,1),
        #     initializer = initializers.get('ones'),
        #     constraint = constraints.NonNeg(),
        #     name = 'ait_conv')

        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/8)
        r2 = sqrt(2.0/8)
        for i in range(8):
            _u = 2*i+1
            for j in range(8):
                _v = 2*j+1
                index = i*8+j
                for u in range(8):
                    for v in range(8):
                        index2 = u*8+v
                        t = cos(_u*u*pi/16)*cos(_v*v*pi/16)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t
        self.kernel = torch.autograd.Varible( kernel.type(torch.FloatTensor), requires_grad=False)

    def forward(self, inputs):
        # print(self.it_weights.weight.data)
        self.kernel = self.kernel * self.it_weights     # 출력도 kernel사이즈랑 동일
        #input.shape batch,CHW
        #weight.shape outchannel,CHW
        y = F.conv2d(inputs, self.kernel)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)
            module.weight.data = w

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        y = input * self.scale
        return y

    def compute_output_shape(self, input_shape):
        return input_shape
