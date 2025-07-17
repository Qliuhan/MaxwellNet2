import torch
from multiprocessing.spawn import import_main_path
from turtle import forward
import torch.nn as nn
from unet.util import *
from unet.net_part import *
import copy

class UNetModel(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        dims,
        num_res_blocks,
        src_channels,
        ):
        super().__init__()

        self.cw = nn.Parameter(torch.randn(1, 1, 1, 100))
        self.rw = nn.Parameter(torch.randn(1, 1, 100, 1))
        
        hidden_channel = [128, 256, 512, 512]

        
        self.input_blocks = nn.ModuleList(
            [
                conv_nd(dims, in_channels, hidden_channel[0], 3, padding=1)
            ]
        )

        # input layers
        for i in range(len(hidden_channel)):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=hidden_channel[i],
                        dims=dims,
                        src_channels=src_channels,
                    )
                ]
                self.input_blocks.append(*layers)

            
            if i != len(hidden_channel)-1:
                self.input_blocks.append(
                    Down(hidden_channel[i], hidden_channel[i+1])
                )

        self.middel_block = nn.ModuleList([
            ResBlock(
                channels=hidden_channel[-1],
                dims=dims,
                src_channels=src_channels
            ),
            ResBlock(
                channels=hidden_channel[-1],
                dims=dims,
                src_channels=src_channels
            )
        ])

        # up sampling
        hs_shape = [100, 50, 25]
        self.output_block1 = nn.ModuleList([])
        for i in range(len(hidden_channel)-1, -1, -1):
            
            for k in range(num_res_blocks+1):
                if i != 0 and k == num_res_blocks:
                    layers_up = nn.Sequential(
                        ResBlock(
                        channels=hidden_channel[i]*2,
                        out_channels=hidden_channel[i],
                        dims=dims,
                        src_channels=src_channels),
                        Up(
                            in_channels=hidden_channel[i],
                            out_channels=hidden_channel[i-1],
                            shape = hs_shape[i-1],
                            bilinear=True)
                    )

                else:
                    layers_up = nn.Sequential(
                        ResBlock(
                        channels=hidden_channel[i]*2,
                        out_channels=hidden_channel[i],
                        dims=dims,
                        src_channels=src_channels)
                    )
                self.output_block1.append(layers_up)

        self.output_block2 = nn.ModuleList([])
        for i in range(len(hidden_channel)-1, -1, -1):
            for k in range(num_res_blocks+1):
                if i != 0 and k == num_res_blocks:
                    layers_up = nn.Sequential(
                        ResBlock(
                        channels=hidden_channel[i]*2,
                        out_channels=hidden_channel[i],
                        dims=dims,
                        src_channels=src_channels),
                        Up(
                            in_channels=hidden_channel[i],
                            out_channels=hidden_channel[i-1],
                            shape = hs_shape[i-1],
                            bilinear=True)
                    )

                else:
                    layers_up = nn.Sequential(
                        ResBlock(
                        channels=hidden_channel[i]*2,
                        out_channels=hidden_channel[i],
                        dims=dims,
                        src_channels=src_channels)
                    )
                self.output_block2.append(layers_up)

        self.output_block3 = nn.ModuleList([])
        for i in range(len(hidden_channel)-1, -1, -1):
            for k in range(num_res_blocks+1):
                if i != 0 and k == num_res_blocks:
                    layers_up = nn.Sequential(
                        ResBlock(
                        channels=hidden_channel[i]*2,
                        out_channels=hidden_channel[i],
                        dims=dims,
                        src_channels=src_channels),
                        Up(
                            in_channels=hidden_channel[i],
                            out_channels=hidden_channel[i-1],
                            shape = hs_shape[i-1],
                            bilinear=True)
                    )

                else:
                    layers_up = nn.Sequential(
                        ResBlock(
                        channels=hidden_channel[i]*2,
                        out_channels=hidden_channel[i],
                        dims=dims,
                        src_channels=src_channels)
                    )
                self.output_block3.append(layers_up)

        self.output_block_list = [self.output_block1, self.output_block2, self.output_block3]

        # out layers
        self.out1 = nn.Sequential(
            nn.BatchNorm2d(hidden_channel[0]),
            nn.LeakyReLU(),
            conv_nd(dims, hidden_channel[0], out_channels, 1)
        )

        self.out2 = nn.Sequential(
            nn.BatchNorm2d(hidden_channel[0]),
            nn.LeakyReLU(),
            conv_nd(dims, hidden_channel[0], out_channels, 1)
        )

        self.out3 = nn.Sequential(
            nn.BatchNorm2d(hidden_channel[0]),
            nn.LeakyReLU(),
            conv_nd(dims, hidden_channel[0], out_channels, 1)
        )

        self.out_list = [self.out1, self.out2, self.out3]

    def forward(self, Ezz, u_init_hat, para_src, src_i):  
        out_total_list = []
        para_src = para_src*(self.rw @ self.cw)  # [B, 5, 100, 100]  # 2_train9

        h = torch.cat((Ezz, u_init_hat, para_src), 1)  # [B, 9, 100, 100]  # 2_train9
        
        # down sampling
        hs = []
        for i, module in enumerate(self.input_blocks):
            if i == 0:
                h = module(h)
            else:
                h_src_i = {'h': h, 'src_i': src_i}
                h = module(h_src_i)  # [1, 1024, 12, 12]
            hs.append(h)
        
        # middle block
        for i ,module in enumerate(self.middel_block):
            h_src_i = {'h': h, 'src_i': src_i}
            h = module(h_src_i)
        h_middel = h

        
        # up sampling & output module
        hs1 = hs.copy()
        hs2 = hs.copy()
        hs_list = [hs, hs1, hs2]

        for k in range(3):
            hs = hs_list[k]
            h = h_middel
            for i, module in enumerate(self.output_block_list[k]):
                h = torch.cat((h, hs.pop()), dim=1)
                h_src_i = {'h': h, 'src_i': src_i}
                h = module(h_src_i)
            out = self.out_list[k](h)  # [1, 20, 100, 100]
            out = out.unsqueeze(1)
            out_total_list.append(out)

        out_total = torch.cat(out_total_list, 1)  # [B, 3, 20, 100, 100]
        
        return out_total


        


        


