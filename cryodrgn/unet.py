import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from . import fft
from . import lie_tools
from . import utils
from . import lattice
from . import mrc
from . import models
from . import attention
from . import ctf
from . import settings
from . import str_models

class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DownBlock, self).__init__()

        down_block = []
        down_block.append(nn.Conv3d(inchannels, outchannels, 4, 2, 1))
        down_block.append(nn.LeakyReLU(0.2))
        #down_block.append(nn.Conv3d(outchannels, outchannels, 3, 1, 1))
        #down_block.append(nn.LeakyReLU(0.2))
        self.down = nn.Sequential(*down_block)
        utils.initseq(self.down)

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()

        up_block = []
        #up_block.append(nn.Conv3d(inchannels, outchannels, 3, 1, 1))
        #up_block.append(nn.LeakyReLU(0.2))
        up_block.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
        up_block.append(nn.LeakyReLU(0.2))

        self.up = nn.Sequential(*up_block)
        utils.initseq(self.up)

    def forward(self, x):
        return self.up(x)

class Unet(nn.Module):
    def __init__(self, D, zdim=256):

        super(Unet, self).__init__()
        self.start_size = D
        self.down_out_dim = (self.start_size)//32
        self.zdim = zdim
        self.transformer = models.SpatialTransformer(self.start_size)
        self.instance_norm = nn.InstanceNorm3d(32)

        self.up_outchannels =   {2:128, 4:192, 8:160, 16:128, 32:64, 64:32}
        self.down_outchannels = {2:256, 4:192, 8:160, 16:128, 32:64, 64:32}
        self.concats = [2, 4, 8, 16]
        #block = []
        #block.append(nn.Conv3d(1, 32, 4, 2, 1))
        #block.append(nn.LeakyReLU(0.2))
        #self.init_conv = nn.Sequential(*block)
        #utils.initseq(self.init_conv)

        downs = []
        resolution = 64
        n_layers = int(np.log2(resolution) - 1)
        inchannels = 1

        for i in range(n_layers):
            outchannels = self.down_outchannels[resolution]
            downs.append(DownBlock(inchannels, outchannels))
            inchannels = outchannels
            resolution //= 2

        self.downs = nn.ModuleList(downs)
        self.downz = nn.Sequential(
                nn.Linear(self.down_outchannels[4] * self.down_out_dim ** 3, 512), nn.LeakyReLU(0.2))

        self.mu = nn.Linear(512, self.zdim)

        utils.initseq(self.downz)

        utils.initmod(self.mu)

        self.upz = nn.Sequential(nn.Linear(self.zdim, 1024), nn.LeakyReLU(0.2))
        utils.initseq(self.upz)
        ups = []
        inchannels = 128
        resolution = 2
        self.templateres = 64

        for i in range(int(np.log2(self.templateres)) - 1):
            up_block = []
            resolution *= 2
            if resolution in self.concats:
                inchannels += self.down_outchannels[resolution]
            outchannels = self.up_outchannels[resolution]
            ups.append(UpBlock(inchannels, outchannels))
            #if inchannels == outchannels:
            inchannels = outchannels
            #outchannels = max(inchannels // 2, 16)
            #else:
            #inchannels = outchannels
        self.ups = nn.ModuleList(ups)

    def forward(self, img, losslist=["kldiv"]):
        B = img.shape[0]
        down32 = self.downs[0](img)
        down32 = self.instance_norm(down32)
        down16 = self.downs[1](down32)  #16, 128
        down8  = self.downs[2](down16) #8, 160
        down4  = self.downs[3](down8) #4, 192
        down2  = self.downs[4](down4) #2, 128
        downz  = self.downz(down2.view(-1, self.down_out_dim ** 3 * self.down_outchannels[4]))
        z      = self.mu(downz)
        #upsample x
        up2   = self.upz(z).view(-1, 128, 2, 2, 2) #512
        #print(up2.shape, down2.shape)
        feat2 = torch.cat([down2, up2], dim=1)
        up4   = self.ups[0](feat2) #512
        #print(up4.shape)
        feat4 = torch.cat([down4, up4], dim=1)
        up8   = self.ups[1](feat4) #256
        feat8 = torch.cat([down8, up8], dim=1)
        up16  = self.ups[2](feat8) #128
        #feat16 = torch.cat([down16, up16], dim=1)
        up32  = self.ups[3](up16) #64
        up64  = self.ups[4](up32) #32

        return up64

class UnetFormer(nn.Module):
    def __init__(self, D):

        super(UnetFormer, self).__init__()
        self.start_size = D
        self.transformer = models.SpatialTransformer(self.start_size)
        self.instance_norm = nn.InstanceNorm3d(32)

        self.up_outchannels =   {2:128, 4:192, 8:256, 16:64, 32:32, 64:16}
        self.down_outchannels = {2:256, 4:192, 8:256, 16:128, 32:64, 64:32}
        self.concats = [2, 4, 8, 16]

        downs = []
        resolution = 64
        n_layers = int(np.log2(resolution) - 1)
        inchannels = 1

        for i in range(3):
            outchannels = self.down_outchannels[resolution]
            downs.append(DownBlock(inchannels, outchannels))
            inchannels = outchannels
            resolution //= 2
        self.downs = nn.ModuleList(downs)
        self.up_dims = nn.Sequential(nn.Conv3d(inchannels, 256, 1, 1,),)

        ups = []
        inchannels = 384 #256
        resolution = 8
        self.templateres = 64

        for i in range(3):
            up_block = []
            resolution *= 2
            outchannels = self.up_outchannels[resolution]
            ups.append(UpBlock(inchannels, outchannels))
            #if inchannels == outchannels:
            inchannels = outchannels
            #outchannels = max(inchannels // 2, 16)
            #else:
            #inchannels = outchannels
        ups.append(nn.ConvTranspose3d(inchannels, 3, 4, 2, 1))
        utils.initmod(ups[-1])

        self.ups = nn.ModuleList(ups)
        pos_embeddings = ctf.FourierFeatures(8,).freqs.view(-1, 256,)
        #pos_embeddings = (torch.permute(pos_embeddings, [1, 0])) #.reshape(-1, 256)
        self.register_buffer('pos_embeddings', pos_embeddings)
        transformers = []
        for i in range(8):
            transformers.append(attention.Transformer(config=settings.CONFIG["evoformer"]))
        self.transformers = nn.ModuleList(transformers)
        self.layer_norm = nn.LayerNorm(256,) #(1, L, c)
        self.layer_norm_down = nn.LayerNorm(256,) #(1, L, c)
        #self.bn1 = nn.BatchNorm3d(32)
        #self.bn2 = nn.BatchNorm3d(64)
        #self.bn3 = nn.BatchNorm3d(128)

        self.bn1 = nn.GroupNorm(4, 32)
        self.bn2 = nn.GroupNorm(2, 64)
        self.bn3 = nn.GroupNorm(1, 128)

        self.layer_norm_pe = nn.LayerNorm(256,)
        #self.map_pred = nn.Conv3d(32, 2, 3, 1, 1)
        #self.pre_norm = nn.LayerNorm(1024)
        #self.mlp1 = nn.Linear(1024, 1024)
        #self.pre_norm1 = nn.LayerNorm(1024)
        #self.mlp2 = nn.Linear(1024, 1024)
        #self.pos_net = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2), nn.Linear(512, 6))
        self.trans_net = nn.Linear(384, 3)
        #torch.nn.init.zeros_(self.pos_net[2].weight)
        #torch.nn.init.zeros_(self.pos_net[2].bias)

        self.render_size = 8
        self.x_size = self.render_size//2+1
        y_idx = torch.arange(-self.x_size+1, self.x_size-1)/float(self.render_size) #[-0.5, 0.5)
        x_idx = torch.arange(self.x_size)/float(self.render_size) #(0, 0.5]
        grid  = torch.meshgrid(y_idx, y_idx, x_idx, indexing='ij')
        grid_x = grid[2] #change fast [[0,1,2,3]]
        grid_y = grid[1]
        grid_z = grid[0]
        freqs3d = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        self.register_buffer("freqs3d", freqs3d)
        x_idx = torch.linspace(-1., 1., 8) #[-s, s)
        grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        xgrid = grid[2]
        ygrid = grid[1]
        zgrid = grid[0]
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0) #(1, D, H, W, 3)
        self.register_buffer("grid", grid)

        params = {}
        params["n_str_layers"] = 1
        params["n_msa_feat"] = 256
        params["n_structure_msa_feat"] = 384
        params["n_pair_feat"] = 128
        self.structure_layer = str_models.CubicModule(config=params,
                                               n_att_head=12,
                                               dropout=0.1,)
        evoformers = []
        self.evoformer = attention.EvoCubformer(config=settings.CONFIG["evoformer"])
        self.evoformer1 = attention.EvoCubformer(config=settings.CONFIG["evoformer"])
        #self.evoformer2 = attention.EvoCubformer(config=settings.CONFIG["evoformer"])
        #for i in range(3):
        #    evoformers.append(attention.EvoCubformer(config=settings.CONFIG["evoformer"]))
        #self.evoformers = nn.ModuleList(evoformers)

    def bfactor_blurring(self, img, bfactor):
        s2 = self.freqs3d.pow(2).sum(dim=-1)
        img = img*torch.exp(-s2*bfactor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*np.pi**2)
        return img

    def forward(self, img, losslist=["kldiv"]):
        #blurring the map
        B = img.shape[0]
        #reshape image to (8, 8, 8, 1, d, h, w)
        #img = img.view(B, 8, 8, 8, 8, 8, 8)
        img = img.view(B, 64, 64, 64)
        img = img.unfold(1, 8, 8).unfold(2, 8, 8).unfold(3, 8, 8)
        img = img.reshape(B*8*8*8, 1, 8, 8, 8)
        #drop img that is negative
        img_bool = (img < 0.4).view(B*8*8*8, -1)
        img_bool = (img_bool).sum(dim=-1) > 506
        all_pos  = ~img_bool
        #all_pos = ~torch.all(img_bool, dim=1)
        all_pos_idx = torch.nonzero(all_pos).squeeze()
        #print(all_pos_idx, all_pos[all_pos_idx])
        all_pos_idx = all_pos_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #select img that is positive
        img = img[all_pos]
        if img.shape[0] < 2:
            return
        x_fft = fft.torch_rfft3_center(img, center=True)
        b_factor = 4.*(torch.rand(img.shape[0]).to(img.get_device()))
        #rand_mask = torch.rand(img.shape[0]).to(img.get_device()) > 0.3
        #print(b_factor.shape)
        x_fft_b = self.bfactor_blurring(x_fft, b_factor)
        img = fft.torch_irfft3_center(x_fft_b, center=True) #* rand_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #img = torch.permute(img, [0, 1, 3, 5, 2, 4, 6]).reshape(B*8*8*8, 1, 8, 8, 8)
        down32 = self.bn1(self.downs[0](img))
        #down32 = self.instance_norm(down32)
        down16 = self.bn2(self.downs[1](down32))  #16, 128
        up8 = self.up_dims(self.bn3(self.downs[2](down16))).view(B, -1, 256) #+ self.pos_embeddings #.view(B, 256, -1) #8, 160
        #up8 = torch.permute(up8, [0, 2, 1])
        up8 = self.layer_norm_down(up8) + self.layer_norm_pe(self.pos_embeddings[all_pos]) #torch.permute(self.pos_embeddings, [1, 0])
        for i in range(len(self.transformers)):
            up8 = self.transformers[i](up8,)# self.pos_embeddings[all_pos])
        #using a layer of evoformer
        msa, pair = self.evoformer(up8)
        msa, pair = self.evoformer1(msa, pair)
        #msa, pair = self.evoformer2(msa, pair)
        #for i in range(len(self.evoformers)):
        #    msa, pair = self.evoformers[i](msa, pair)
        #using a layer of IPA
        msa, pred_rots, pred_trans, up64s, atoms, local_trans = self.structure_layer(msa, pair.unsqueeze(0))
        #R = pred_rots[-1].squeeze() #(1, L, 3, 3) -> (L, 3, 3)
        #trans = pred_trans[-1].squeeze() #(1, L, 1, 3) -> (L, 3)
        #global_trans = self.trans_net(msa).mean(dim=1, keepdim=True)*0.01 #(B, 1, 3)
        #print(global_trans.shape, trans.shape)
        # add global translation to trans
        #trans = trans + global_trans[-1]
        up8 = msa #(1, L, 384)
        #print(R.shape, trans.shape)

        #up8 = self.layer_norm(up8)
        #estimate pose
        #pose = self.pos_net(up8).view(B*8*8*8, 6)
        #one = torch.ones_like(pose[:, :1])*2.
        #rotation = torch.cat([one, pose[:, :3]], dim=-1)
        #R = lie_tools.quaternions_to_SO3_wiki(rotation)
        #trans = pose[:, 3:]*0.1

        #up8 = F.relu(self.mlp(up8))
        #up8 = self.pre_norm(up8)
        #up8 = F.relu(self.mlp1(up8))
        #up8 = up8 + self.mlp2(self.pre_norm1(up8))
        #up8 = up8.view(B*8*8*8, msa.shape[-1], 1, 1, 1)
        ##up8 = torch.permute(up8, [0, 1, 2]).reshape(B, -1, 8, 8, 8)
        #up16  = self.ups[0](up8) #128
        #up32  = self.ups[1](up16) #64
        #up64  = self.ups[2](up32) #32
        #up64  = self.ups[3](up64)
        up64_idx = all_pos_idx.repeat((1, 6, 8, 8, 8))
        img_idx = all_pos_idx.repeat((1, 1, 8, 8, 8))
        imgs = []
        for i in range(up64s.shape[0]):
            R = pred_rots[i].squeeze() #(1, L, 3, 3) -> (L, 3, 3)
            trans = pred_trans[i].squeeze() #(1, L, 1, 3) -> (L, 3)
            L = atoms.shape[1]
            D = atoms.shape[2]
            local_tran = local_trans[i].squeeze()
            #local_trans = local_trans[i].squeeze().view(L*D, 3) #(L, 4, 3)

            #up64 = up64s[i]#[:, 1:2, ...]
            up64 = atoms[i].view(L*D, 1, 4, 4, 4)
            #print(up64.shape)
            #grid_R = self.grid @ R.unsqueeze(1).unsqueeze(1) + trans.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            #print(local_tran.unsqueeze(2).unsqueeze(2).shape, R.shape, atoms.shape)
            grid_R = self.grid @ R.unsqueeze(1).unsqueeze(1)
            grid_R = grid_R.unsqueeze(1) + 0.1*local_tran.unsqueeze(2).unsqueeze(2).unsqueeze(2) #(L, 4, 1, 1, 3)
            grid_R = grid_R.view(L*D, 8, 8, 8, 3)
            #print(up64.shape, grid_R.shape)
            up64 = F.grid_sample(up64, grid_R, mode='bilinear', align_corners=utils.ALIGN_CORNERS) #(L, 3, 8, 8, 8)
            #up64 = F.interpolate(up64, size=[8]*3, mode='trilinear', align_corners=utils.ALIGN_CORNERS)
            #cat sidechains and solvent
            up64 = torch.cat([up64s[i], up64.view(L, D, 8, 8, 8)], dim=1) #(L, 6, 8, 8, 8)
            #gather cubes
            up64_full = torch.zeros(B*8*8*8, up64.shape[1], 8, 8, 8).to(up64.get_device())
            #print(up64_full.shape)
            up64_full[all_pos, 0] = 5.
            #print(up64.shape, all_pos_idx.shape)
            up64 = torch.scatter(up64_full, 0, up64_idx, up64)
            up64 = up64.view(B, 8, 8, 8, -1, 8, 8, 8)
            up64 = torch.permute(up64, [0, 4, 1, 5, 2, 6, 3, 7,])
            #up8 = up8.view(B, 8, 8, 8, 8, 8, 8, 2)
            #up64 = torch.permute(up8, [0, 7, 1, 4, 2, 5, 3, 6,])
            up64 = up64.reshape(B, -1, 64, 64, 64)
            #up64 = torch.cat([up64s[i][:, :1, ...], up64, up64s[i][:, 2:, ...]], dim=1)
            #print(up64.shape)

            img_full = torch.zeros(B*8*8*8, 1, 8, 8, 8).to(up64.get_device())
            img_full.scatter_(0, img_idx, img)
            img = img_full
            img = img.view(B, 8, 8, 8, -1, 8, 8, 8)
            img = torch.permute(img, [0, 4, 1, 5, 2, 6, 3, 7,])
            img = img.reshape(B, -1, 64, 64, 64)
            imgs.append(img)
        #imgs = torch.cat(imgs, dim=0)

        return R, trans, up64, img

