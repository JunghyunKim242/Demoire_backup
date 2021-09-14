import torch
import torch.nn as nn
import torch.nn.functional as F

# from ops import *
from Net.UNet_class import *
from Net.Mbcnn_class import *
# from torch.nn import PixelShuffle

""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F


class MBCNN(nn.Module):
    def __init__(self,nFilters, multi=True):
        super().__init__()
        self.ttmp = 0;
        # self.imagesize = 1024
        self.imagesize = 256

        self.output_list = []

        self.d_list_a = (1, 2, 3, 2, 1)
        self.d_list_b = (1, 2, 3, 2, 1)
        self.d_list_c = (1, 2, 2, 2, 1)

        # self.x = x
        # self.Space2Depth1 =  nn.PixelUnshuffle(2)

        self.conv_func1 = conv_relu(12,nFilters*2,3, padding='same')
        self.pre_block1 = pre_block(128, self.d_list_a, enable = True)

        # layers.ZeroPadding2D(padding=(1, 1))(t2)
        self.conv_func2 = conv_relu(128,nFilters*2,3, padding='valid',strides=(2,2))
        self.pre_block2 = pre_block(128, self.d_list_a, enable = True)

        # layers.ZeroPadding2D(padding=(1, 1))(t2)
        self.conv_func3 = conv_relu(128,nFilters*2,3, padding='valid',strides=(2,2))
        self.pre_block3 = pre_block(128,self.d_list_c, True)

        # self.global_block1 = global_block(128)
        self.global_block1 = global_block(self.imagesize//8)
        self.pos_block1 = pos_block(self.ttmp, self.d_list_c)
        self.conv1 = conv(128, 12, 3)
        self.Depth2space1 =  nn.PixelShuffle(2)
        self.conv_func4 = conv_relu(131, nFilters*2, 1)
        # self.global_block2 = global_block(256)
        self.global_block2 = global_block(self.imagesize//4)
        self.pre_block4 = pre_block(128, self.d_list_b,True)
        # self.global_block3 = global_block(256)
        self.global_block3 = global_block(self.imagesize//4)
        self.pos_block2 = pos_block(self.ttmp, self.d_list_b)
        self.conv2 = conv(128, 12, 3)
        self.Depth2space2 =  nn.PixelShuffle(2)

        # output_list.append(t2_out)
        # _t1 = layers.Concatenate()([t1, t2_out])
        self.conv_func5 = conv_relu(131, nFilters*2, 1)
        # self.global_block4 = global_block(512)
        self.global_block4 = global_block(self.imagesize//2)
        self.pre_block5 = pre_block(128, self.d_list_a,True)
        # self.global_block5 = global_block(512)
        self.global_block5 = global_block(self.imagesize//2)
        self.pos_block3 = pos_block(self.ttmp, self.d_list_a)
        self.conv3 = conv(128, 12, 3)
        self.Depth2space3 =  nn.PixelShuffle(2)

    def forward(self, x):
        output_list = []
        d_list_a = (1, 2, 3, 2, 1)
        d_list_b = (1, 2, 3, 2, 1)
        d_list_c = (1, 2, 2, 2, 1)

        shape = list(x.shape)
        # print('input shape = ',shape)
        batch,channel,height,width = shape
        print('batch , channel, height, width \t',batch, channel, height, width)

        # _x = self.Space2Depth1(x)
        _x = x.view(batch,channel*4,height//2,width//2)
        # nn.PixelUnshuffle(2)

        print('line78line78line78t1.shape',_x.shape)
        t1 = self.conv_func1(_x)

        # print('line80line80line80t1.shape',t1.shape)
        t1 = self.pre_block1(t1)
        # print('line83   line83  line83  t1.shape',t1.shape)
        t2 = F.pad(t1, (1,1,1,1))

        # print('\nFlag1 convfunc22 line 86, line 86.line 86',t2.shape)  #4,128,514,514
        t2 = self.conv_func2(t2)
        # print('line 88, line 88.line 88',t2.shape)  #4,128,256,256,
        t2 = self.pre_block2(t2)
        # print('line 90, line 90.line 90',t2.shape)#4,128,256,256,
        t3 = F.pad(t2, (1,1,1,1))


        # print('\nFlag2 convfunc33 line 91, line 91.line 91',t3.shape) # 128,258,258
        t3 = self.conv_func3(t3)
        # print('line 93, line 93.line 93',t3.shape)      # 128,128,128 check
        t3 = self.pre_block3(t3)
        # print('line 95,global block input',t3.shape)      #128,128,128
        t3 = self.global_block1(t3)
        # print('line 97, line 97.line 97',t3.shape)
        t3 = self.pos_block1(t3)
        t3_out = self.conv1(t3)
        # print('\nline 103 ')
        t3_out = self.Depth2space1(t3_out)
        #_x = x.view(4,channel*4,height//2,width//2)
        # print('\nt3_out.shape ',t3_out.shape)
        output_list.append(t3_out)
        # print('\nline 107 ')

        _t2 = torch.cat( [t3_out, t2] , dim=-3) # channel을 concat하기
        # print('\noutput torch.cat ',_t2.shape) # 4,131,512,512
        _t2 = self.conv_func4(_t2 )
        # print('\noutput conv_func4 ',_t2.shape) # 4,131,512,512
        _t2 = self.global_block2(_t2)
        # print('\noutput global_block2 ',_t2.shape) # 4,131,512,512
        _t2 = self.pre_block4(_t2)
        # print('\noutput pre_block4 ',_t2.shape) # 4,131,512,512
        _t2 = self.global_block3(_t2)
        # print('\noutput global_block3 ',_t2.shape) # 4,131,512,512
        _t2 = self.pos_block2(_t2)
        # print('\noutput pos_block2 ',_t2.shape) # 4,131,512,512
        t2_out = self.conv2(_t2)
        # print('\noutput conv2 ',t2_out.shape) # 4,131,512,512
        t2_out = self.Depth2space2(t2_out)
        # print('\noutput Depth2space2 ',t2_out.shape) # 4,131,512,512
        output_list.append(t2_out)
        # print('\nline 118 ')

        _t1 = torch.cat( [t1, t2_out] , dim=-3) # channel을 concat하기
        # print('\noutput torch.cat ',_t1.shape) # 4,131,512,512
        _t1 = self.conv_func5(_t1)
        # print('\noutput conv_func5  ',_t1.shape)    #4,128,512,512
        _t1 = self.global_block4(_t1)
        # print('\noutput global_block4 ',_t1.shape)
        _t1 = self.pre_block5(_t1)
        # print('\noutput pre_block5  ',_t1.shape)
        _t1 = self.global_block5(_t1)
        # print('\noutput global_block5 ',_t1.shape)
        _t1 = self.pos_block3(_t1)
        # print('\noutput global_block5 ',_t1.shape)
        _t1 = self.conv3(_t1)
        # print('\noutput conv3 ',_t1.shape)
        y   = self.Depth2space3(_t1)

        output_list.append(y)
        print('line 130 ')

        # 여기 확인 하기
        # if multi != True:
        #     return models.Model(x, y)
        # else:
        #     return models.Model(x, output_list)

        return y #output_list


