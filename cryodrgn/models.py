'''Pytorch models'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib.pyplot as plt
import healpy as hp
from . import pose_encoder
from . import decoders
from . import fft
from . import lie_tools
from . import utils
from . import lattice
from . import mrc
from . import symm_groups
from . import healpix_sampler
from . import ctf

log = utils.log
ALIGN_CORNERS = utils.ALIGN_CORNERS

class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(self, lattice, # Lattice object
            qlayers, qdim,
            players, pdim,
            in_dim, zdim = 1,
            encode_mode = 'resid',
            enc_mask = None,
            enc_type = 'linear_lowf',
            enc_dim = None,
            domain = 'fourier',
            activation = nn.ReLU,
            ref_vol = None,
            Apix = 1.,
            ctf_grid = None,
            template_type = None,
            warp_type = None,
            num_struct = 1,
            deform_emb_size = 2,
            device = None,
            symm = None,
            render_size=140,
            downfrac=0.5,
            templateres=192,
            tmp_prefix="ref",
            window_r=0.85,
            masks_params=None,
            num_bodies=0,
            z_affine_dim=4,
            ctf_alpha=0.,
            ctf_beta=1.,
            normalize_ctf=False,
            rank=0):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        self.encode_mode = encode_mode
        self.num_struct = num_struct
        self.fixed_deform = False
        self.device = device
        self.render_size = (int((lattice.D - 1)*downfrac)//2)*2
        self.num_bodies = num_bodies
        self.z_affine_dim = z_affine_dim
        self.ctf_alpha = ctf_alpha
        self.ctf_beta = ctf_beta
        self.normalize_ctf = normalize_ctf
        if ref_vol is not None:
            in_vol_nonzeros = torch.nonzero(ref_vol)
            in_vol_mins, _ = in_vol_nonzeros.min(dim=0)
            in_vol_maxs, _ = in_vol_nonzeros.max(dim=0)
            if rank == 0:
                log("model: loading mask with nonzeros between {}, {}, {}".format(in_vol_mins, in_vol_maxs, ref_vol.shape))
            in_vol_maxs = ref_vol.shape[-1] - in_vol_maxs
            in_vol_min = min(in_vol_maxs.min(), in_vol_mins.min())
            mask_frac = (ref_vol.shape[-1] - in_vol_min*2 + 4) / ref_vol.shape[-1]
            self.window_r = min(mask_frac, 0.98)
            if rank == 0:
                log("model: cropping volume using fraction: {}".format(self.window_r))
            if templateres == 256:
                self.window_r = min(self.window_r, 0.9)
        else:
            self.window_r = window_r
        self.down_vol_size = int(self.render_size*self.window_r)//2*2
        self.encoder_crop_size = self.down_vol_size
        self.encoder_image_size = int(self.render_size*self.encoder_crop_size/self.down_vol_size)//2*2
        assert self.encoder_image_size == self.render_size
        log("model: image supplemented into encoder will be of size {}".format(self.encoder_image_size))
        self.ref_vol = ref_vol

        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, zdim*2)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(in_dim,
                            qlayers, # nlayers
                            qdim,  # hidden_dim
                            zdim*2, # out_dim
                            activation)
        elif encode_mode == 'mlp':
            self.encoder = MLP(in_dim,
                            qlayers,
                            qdim, # hidden_dim
                            zdim*2, # out_dim
                            activation) #in_dim -> hidden_dim
        elif encode_mode == 'fixed':
            self.encoder = FixedEncoder(self.num_struct, self.zdim+4)
            #self.pose_encoder = pose_encoder.PoseEncoder(image_size=128)
        elif encode_mode == 'deform':
            self.encoder = FixedEncoder(self.num_struct, self.zdim)
            self.fixed_deform = True
        elif encode_mode == 'grad':
            self.encoder = Encoder(self.zdim, lattice.D, crop_vol_size=self.encoder_crop_size,
                                   in_mask=ref_vol, window_r=self.window_r, render_size=self.encoder_image_size,
                                   masks_params=masks_params, z_affine_dim=self.z_affine_dim, rank=rank)
            self.fixed_deform = True
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        self.warp_type = warp_type

        self.encode_mode = encode_mode
        self.vanilla_dec = enc_type == "vanilla"
        self.template_type = template_type
        self.symm = symm
        self.deform_emb_size = deform_emb_size
        self.templateres = templateres
        self.masks_params = masks_params
        self.Apix = Apix
        self.decoder = get_decoder(3+zdim, lattice.D, players, pdim, domain, enc_type, enc_dim,
                                   activation, ref_vol=ref_vol, Apix=Apix,
                                   template_type=self.template_type, templateres=self.templateres,
                                   warp_type=self.warp_type,
                                   symm=self.symm, ctf_grid=ctf_grid,
                                   fixed_deform=self.fixed_deform, deform_emb_size=self.deform_emb_size,
                                   render_size=self.encoder_image_size, down_vol_size=self.down_vol_size, tmp_prefix=tmp_prefix,
                                   masks_params=self.masks_params, num_bodies=self.num_bodies, affine_dim=self.z_affine_dim,
                                   ctf_alpha=self.ctf_alpha, ctf_beta=self.ctf_beta, normalize_ctf=self.normalize_ctf, rank=rank)

    @classmethod
    def load(self, config, weights=None, device=None):
        '''Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        '''
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg['lattice_args']
        lat = lattice.Lattice(c['D'], extent=c['extent'])
        c = cfg['model_args']
        if c['enc_mask'] > 0:
            enc_mask = lat.get_circular_mask(c['enc_mask'])
            in_dim = int(enc_mask.sum())
        else:
            assert c['enc_mask'] == -1
            enc_mask = None
            in_dim = lat.D**2
        activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
        model = HetOnlyVAE(lat,
                          c['qlayers'], c['qdim'],
                          c['players'], c['pdim'],
                          in_dim, c['zdim'],
                          encode_mode=c['encode_mode'],
                          enc_mask=enc_mask,
                          enc_type=c['pe_type'],
                          enc_dim=c['pe_dim'],
                          domain=c['domain'],
                          activation=activation)
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt['model_state_dict'])
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, *img):
        img = (x.view(x.shape[0],-1) for x in img)
        if self.enc_mask is not None:
            img = (x[:,self.enc_mask] for x in img)
        z = self.encoder(*img)
        #if self.encode_mode == 'fixed':
        #    z = torch.tile(self.encoder, (x.shape[0], 1))
        return z[:,:self.zdim], z[:,self.zdim:]

    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1]*(coords.ndimension()-2)), self.zdim)
        z = torch.cat((coords,z.expand(*coords.shape[:-1],self.zdim)),dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords,z))

    def get_fixedcode(self):
        return self.encoder()

    def vanilla_encode(self, img, rots=None, trans=None, eulers=None, num_gpus=4, snr2=1., body_poses=[None, None], ctf_param=None):
        if self.encode_mode == 'fixed':
            z = self.encoder()
            z = z.repeat(num_gpus, 1)
            encout = {'encoding': None, 'z_mu': z}
        elif self.encode_mode == 'fixed_blur':
            #split encodings to template and blur kernel
            zs = self.encoder()
            z = zs[:1, :]
            #print(img.shape)
            encout = {"encoding": zs[1:, :]}
            #print(z.shape, encout['encoding'].shape)
        elif self.encode_mode == "grad":
            snr = np.sqrt(np.abs(snr2))
            ctf_embedding = None #ctf.compute_ctf_embedding(64, *torch.split(ctf_param, 1, 1))
            encout = self.encoder(img, rots, trans, losslist=["kldiv"], eulers=eulers, snr=snr,
                                  body_euler=body_poses[0], body_trans=body_poses[1], ctf_embedding=ctf_embedding)
            mu     = encout["z_mu"]
            logstd = encout["z_logstd"]
            z      = encout["z"]
            encout["encoding"] = z
            #x3d_center = encout["rotated_x"]
            #diff = (x3d_center.unsqueeze(1) - x3d_center.unsqueeze(0)).pow(2).sum(dim=(-1,-2))
            diff = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(dim=(-1))
            top = torch.topk(diff, k=3, dim=-1, largest=False, sorted=True)
            #print(top.values)
            #print(top.indices[:, 1:], mu)
            encout["z_knn"] = mu[top.indices[:, 1:],:]#self.zdim]
            #print(encout["z_knn"])
        return z, encout

    def vanilla_decode(self, rots, trans, z=None, save_mrc=False, eulers=None,
                       ref_fft=None, ctf_param=None, encout=None, mask=None,
                       body_poses=(None, None), ctf_grid=None, estpose=False,
                       ctf_filename=None, write_ctf=False, bfactor=4., snr2=1.):
        in_template = None
        if self.encode_mode != 'deform':
            #randomly perturb rotation
            #z = encout['encoding']
            #encout["z_mu"]     = mu
            #z, encout = self.vanilla_encode(img, rots, trans)
            pass
        else:
            #for deform embdding, the encoding will come from z
            encout = {'encoding': None}

        snr = np.sqrt(np.abs(snr2))
        decout = self.decoder(rots, trans, z=z, in_template=in_template, save_mrc=save_mrc,
                        euler=eulers, ref_fft=ref_fft, ctf_param=ctf_param,
                        body_euler=body_poses[0], body_trans=body_poses[1], ctf_grid=ctf_grid,
                        estimate_pose=estpose, ctf_filename=ctf_filename, bfactor=bfactor, snr=snr)
        #decout["y_recon_ori"] = y_recon_ori = decout["y_recon"]
        y_recon_ori = decout["y_recon_ori"]
        pad_size = (self.render_size - self.down_vol_size)//2

        #freqs = ctf_grid.freqs2d.unsqueeze(0)/self.Apix
        #c = ctf.compute_3dctf(ref_fft, ctf_grid.centered_freqs, freqs, *torch.split(ctf_param, 1, -1), plot=True)
        #c = decout["ctf"]
        #if len(c.shape) == 4:
        #    c = c.unsqueeze(1)
        ## downsample ctf to render_size
        #if c.shape[-2] != self.render_size:
        #    c = torch.fft.fftshift(c, dim=(-3,-2))
        #    c = utils.crop_fft3d(c, self.render_size)
        #    c = torch.fft.fftshift(c, dim=(-3,-2))
        #print(y_recon_ori.shape, c.shape)

        #retrieve fft of reconstruction from decout
        #y_recon_fft = torch.view_as_complex(decout["y_recon"])*c
        #decout["y_recon"] = fft.torch_irfft3_center(y_recon_fft)

        # crop reconstruction here
        #decout["y_recon"] = utils.crop_vol(decout["y_recon"], self.encoder_crop_size)
        #images = decout["y_recon"]
        #decout["y_recon_mean"] = images.detach().mean(dim=1, keepdim=True)
        # substract reconstruction to form diff image
        #rnd_factor = torch.rand([images.shape[0], 1, 1, 1], device=images.device)*0.2
        #decout["y_ref"] -= decout["y_recon_mean"]*rnd_factor
        #decout["y_ref"] /= (1. - rnd_factor)

        return decout

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, *args, **kwargs):
        if self.vanilla_dec:
            return self.vanilla_decode(*args, **kwargs)
        else:
            return self.decode(*args, **kwargs)

    def save_mrc(self, filename, enc=None, Apix=1., flip=False):
        if self.vanilla_dec:
            if enc is not None:
                self.decoder.save(filename, z=enc, Apix=Apix, flip=flip)

    def get_images(self, rots, trans):
        assert self.vanilla_dec
        return self.decoder.get_images(self.encoder(), rots, trans)

    def get_vol(self, z):
        if self.vanilla_dec:
            encoding = None
            if self.encode_mode == 'fixed':
                z = self.encoder()
            elif self.encode_mode == 'fixed_blur':
                z = self.encoder()
                #encout = self.affine_encoder(img)
                #encoding = encout['encoding']
                #z += encoding
            return self.decoder.get_vol(z=z)

def load_decoder(config, weights=None, device=None):
    '''
    Instantiate a decoder model from a config.pkl

    Inputs:
        config (str, dict): Path to config.pkl or loaded config.pkl
        weights (str): Path to weights.pkl
        device: torch.device object

    Returns a decoder model
    '''
    cfg = utils.load_pkl(config) if type(config) is str else config
    c = cfg['model_args']
    D = cfg['lattice_args']['D']
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
    model = get_decoder(3, D, c['layers'], c['dim'], c['domain'], c['pe_type'], c['pe_dim'], activation)
    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['model_state_dict'])
    if device is not None:
        model.to(device)
    return model

def get_decoder(in_dim, D, layers, dim, domain, enc_type, enc_dim=None, activation=nn.ReLU, templateres=128,
                ref_vol=None, Apix=1., template_type=None, warp_type=None,
                symm=None, ctf_grid=None, fixed_deform=False, deform_emb_size=2, render_size=140,
                down_vol_size=140, tmp_prefix="ref", masks_params=None, num_bodies=0, affine_dim=4,
                ctf_alpha=0, ctf_beta=1, normalize_ctf=False, rank=0):
    if enc_type == 'none':
        if domain == 'hartley':
            model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
            ResidLinearMLP.eval_volume = PositionalDecoder.eval_volume # EW FIXME
        else:
            model = FTSliceDecoder(in_dim, D, layers, dim, activation)
        return model
    elif enc_type == 'vanilla':
        return VanillaDecoder(D, ref_vol, Apix, template_type=template_type, templateres=templateres, warp_type=warp_type,
                              symm_group=symm, ctf_grid=ctf_grid,
                              fixed_deform=fixed_deform,
                              deform_emb_size=deform_emb_size,
                              zdim=in_dim - 3, render_size=render_size,
                              down_vol_size=down_vol_size, tmp_prefix=tmp_prefix, masks_params=masks_params, num_bodies=num_bodies,
                              affine_dim=affine_dim, ctf_alpha=ctf_alpha, ctf_beta=ctf_beta, normalize_ctf=normalize_ctf, rank=rank)
    else:
        model = PositionalDecoder if domain == 'hartley' else FTPositionalDecoder
        return model(in_dim, D, layers, dim, activation, enc_type=enc_type, enc_dim=enc_dim)

class FixedEncoder(nn.Module):
    def __init__(self, num_struct=1, in_dim=256):
        super(FixedEncoder, self).__init__()
        self.in_dim = in_dim
        self.num_struct=num_struct
        self.register_buffer('encoding1', torch.randn((self.num_struct, self.in_dim)))

    def forward(self,):
        return self.encoding1

class ConvTemplate(nn.Module):
    def __init__(self, in_dim=256, outchannels=1, templateres=128, affine=False, num_bodies=0, affine_dim=4):

        super(ConvTemplate, self).__init__()

        self.zdim = in_dim
        self.outchannels = outchannels
        self.templateres = templateres
        templateres = 256

        self.template1 = nn.Sequential(nn.Linear(self.zdim, 512), nn.LeakyReLU(0.2),
                                       nn.Linear(512, 2048), nn.LeakyReLU(0.2))
        # output affine parameter
        self.use_affine = False
        if affine:
            self.affine_hidden = 512
            self.num_bodies = num_bodies
            #self.affine_out = nn.Sequential(nn.Linear(2048, self.affine_hidden), nn.LeakyReLU(0.2), nn.Linear(self.affine_hidden, num_bodies*6))
            self.z_affine_dim = affine_dim
            self.affine_out = nn.Sequential(nn.Linear(self.z_affine_dim, self.affine_hidden), nn.LeakyReLU(0.2), nn.Linear(self.affine_hidden, (num_bodies+1)*6))
            torch.nn.init.normal_(self.affine_out[2].weight, std=0.5/self.affine_hidden)
            torch.nn.init.zeros_(self.affine_out[2].bias)
            self.use_affine = True
        template2 = []
        inchannels, outchannels = 2048, 1024
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))

        inchannels, outchannels = 1024, 512
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))
        if self.templateres != templateres:
            self.template2 = nn.Sequential(*template2)

            inchannels, outchannels = 512, 256
            template3 = []
            template4 = []
            for i in range(int(np.log2(templateres)) - 3):
                if i < 2: #2:
                    template3.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                    template3.append(nn.LeakyReLU(0.2))
                else:
                    template4.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                    template4.append(nn.LeakyReLU(0.2))
                inchannels = outchannels
                outchannels = inchannels//2 #max(inchannels // 2, 16)
            self.template3 = nn.Sequential(*template3)
            self.template4 = nn.Sequential(*template4)
            for m in [self.template1, self.template2, self.template3, self.template4]:
                utils.initseq(m)
        else:
            inchannels, outchannels = 512, 256
            for i in range(int(np.log2(templateres)) - 3):
                template2.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                template2.append(nn.LeakyReLU(0.2))
                inchannels = outchannels
                outchannels = inchannels//2 #max(inchannels // 2, 16)
            self.template2 = nn.Sequential(*template2)
            self.template3 = self.template4 = []
            for m in [self.template1, self.template2]:
                utils.initseq(m)

        self.conv_out = nn.ConvTranspose3d(inchannels, 1, 4, 2, 1)
        #self.conv_out = nn.Conv3d(inchannels, 1, 3, 1, 1)

        utils.initmod(self.conv_out)#, gain=1./np.sqrt(templateres))
        self.intermediate_size = int(16*self.templateres/256)
        log('convtemplate: the output volume is of size {}, resample intermediate activations of size 16 to {}'.format(self.templateres, self.intermediate_size))
        # output rigid grid transformations

    def forward(self, encoding,):
        #modules = [module for k, module in self.template2._modules.items()]
        #return checkpoint_sequential(modules, 2, self.template1(encoding).view(-1, 1024, 1, 1, 1))
        template1 = self.template1(encoding[..., :self.zdim])
        template2 = self.template2(template1.view(-1, 2048, 1, 1, 1))
        affine = None
        if self.use_affine and encoding.shape[-1] > self.zdim:
            #affine = self.affine_out(template1).view(-1, self.num_bodies, 6)
            affine = self.affine_out(encoding[..., self.zdim:]).view(-1, (self.num_bodies+1), 6)
            one = torch.ones_like(affine[..., :1])*16.
            quat = torch.cat([one, affine[..., :3]], dim=-1)
            #quat = lie_tools.exp_quaternion(affine[..., :3])
            trans = affine[..., 3:]
            affine = (quat, trans)
        if self.templateres != 256:
            template3 = self.template3(template2) #(B, 64, 32, 32, 32)
            #can revise this to achieve other resolutions, current output of size 24*2^3
            template3 = F.interpolate(template3, size=self.intermediate_size, mode="trilinear", align_corners=ALIGN_CORNERS)
            template4 = self.template4(template3)
        else:
            template4 = template2

        return self.conv_out(template4), affine

class AffineMixWeight(nn.Module):
    def __init__(self, in_dim=8, out_dim=3, out_size=32):
        super(AffineMixWeight, self).__init__()

        self.quat = utils.Quaternion()
        self.out_dim = out_dim

        inchannels = 8
        self.inchannels = inchannels
        self.warpf = nn.Sequential(
                nn.Linear(in_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, inchannels*2*2*2), nn.LeakyReLU(0.2)
                )
        outchannels = self.out_dim
        upsample = []
        n_layers = int(np.log2(out_size) - 1)
        for i in range(n_layers - 1):
            upsample.append(nn.ConvTranspose3d(inchannels, inchannels, 4, 2, 1))
            upsample.append(nn.LeakyReLU(0.2))
        upsample.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
        self.upsample = nn.Sequential(*upsample)

        utils.initseq(self.warpf)
        utils.initseq(self.upsample)

    def forward(self, encoding):
        init_vol = self.warpf(encoding).view(-1, self.inchannels, 2, 2, 2)
        out = self.upsample(init_vol)
        return out

class Encoder(nn.Module):
    def __init__(self, zdim, D, crop_vol_size, in_mask=None, window_r=None, render_size=None, masks_params=None, z_affine_dim=4, rank=0):
        super(Encoder, self).__init__()

        self.zdim = zdim
        self.inchannels = 1
        self.vol_size = D - 1
        self.render_size = render_size
        self.crop_vol_size = crop_vol_size #int(render_size*window)
        self.scale = self.render_size/(self.crop_vol_size-1)*2.
        self.crop_scale = (self.crop_vol_size-1)/(self.render_size-1)
        self.window_r = window_r #(the cropping fraction of input mask)
        #downsample volume
        self.transformer_e = SpatialTransformer(self.crop_vol_size, render_size=self.render_size)
        #self.out_dim = (self.crop_vol_size)//128 + 1
        self.out_dim = 1
        if rank == 0:
            log("encoder: the input after cropping is {}, render size is {}, original size is {}".format(
                                                    self.crop_vol_size, self.render_size, self.vol_size))
            log("encoder: the crop fraction is {}, the scale is {}".format(self.crop_scale, self.scale))
        # create 3d frequencies
        self.x_size = self.render_size//2 + 1
        #y_idx = torch.arange(-self.x_size+2, self.x_size)/float(self.render_size) #(-0.5, 0.5]
        y_idx = torch.arange(-self.x_size+1, self.x_size-1)/float(self.render_size) #[-0.5, 0.5)
        x_idx = torch.arange(self.x_size)/float(self.render_size) #(0, 0.5]
        grid  = torch.meshgrid(y_idx, y_idx, x_idx, indexing='ij')
        grid_x = grid[2] #change fast [[0,1,2,3]]
        grid_y = grid[1]
        grid_z = grid[0]
        #grid_y = torch.roll(grid[1], shifts=(self.x_size,), dims=(1,)) #fft shifted, center at the corner
        #grid_z = torch.roll(grid[0], shifts=(self.x_size,), dims=(0,))
        freqs3d = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        if rank == 0:
            log("encoder: created freqs3d {}".format(freqs3d[0,...].pow(2).sum(dim=-1)))
        x_idx = torch.linspace(-1., 1., self.render_size) #[-s, s)
        grid  = torch.meshgrid(x_idx, x_idx, indexing='ij')
        xgrid = grid[1] #change fast [[0,1,2,3]]
        ygrid = grid[0]
        grid2d = torch.stack((xgrid, ygrid), dim=-1).unsqueeze(0)
        self.register_buffer("freqs2d", grid2d)
        self.register_buffer("freqs3d", freqs3d)
        #print(self.freqs3d)

        #downsample mask
        if in_mask is not None:
            crop_mask_size = (int(in_mask.shape[-1]*self.window_r)//2)*2 #(self.crop_vol_size/128) previous default
            if rank == 0:
                log("encoder: cropping mask from {} to {} using window {},".format(in_mask.shape, crop_mask_size, self.window_r))
            in_mask = self.transformer_e.crop(in_mask, crop_mask_size).unsqueeze(0).unsqueeze(0)
            # downsample
            in_mask = self.transformer_e.sample(in_mask)
            self.register_buffer("in_mask", in_mask)
            if rank == 0:
                print(self.in_mask.shape)
            self.use_mask = True
        else:
            self.register_buffer("in_mask", (self.transformer_e.grid.pow(2).sum(dim=-1) < 1).float())
            self.use_mask = True

        downsample = []
        n_layers = int(np.log2(128//2))
        inchannels = 1
        outchannels = 32
        self.down2 = []
        downsample1 = []
        self.intermediate_size = 12
        for i in range(n_layers):
            if i < 3:
                downsample.append(nn.Conv3d(inchannels, outchannels, 4, 2, 1))
                #if i < 2:
                downsample.append(nn.LeakyReLU(0.2))
            else:
                #if i == n_layers - 1:
                #    downsample1.append(nn.Conv3d(inchannels, outchannels, 3,))
                #else:
                downsample1.append(nn.Conv3d(inchannels, outchannels, 4, 2, 1))
                downsample1.append(nn.LeakyReLU(0.2))
            inchannels = outchannels
            #if inchannels == outchannels:
            outchannels = min(inchannels * 2, 512)
            #else:
        self.out_channels = inchannels
        #downsample1.insert(0, nn.LeakyReLU(0.2))
        self.down1 = nn.Sequential(*downsample)
        self.down2 = nn.Sequential(*downsample1)
        #self.down1 = nn.Sequential(
        #        nn.Conv3d(self.inchannels, 16, 4, 2, 1),    nn.LeakyReLU(0.2),#40
        #        nn.Conv3d(16, 16, 4, 2, 1),   nn.LeakyReLU(0.2),#20
        #        nn.Conv3d(16, 32, 4, 2, 1),  nn.LeakyReLU(0.2),#10
        #        nn.Conv3d(32, 32, 4, 2, 1), nn.LeakyReLU(0.2),#5
        #        nn.Conv3d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),#2
        #        nn.Conv3d(64, self.out_channels, 4, 2, 1), nn.LeakyReLU(0.2))#1
        self.down3 = nn.Sequential(
                nn.Linear(self.out_channels * self.out_dim ** 3, 512), nn.LeakyReLU(0.2))

        self.z_affine_dim = z_affine_dim
        self.mu = nn.Linear(512, self.zdim+self.z_affine_dim)
        self.logstd = nn.Linear(512, self.zdim+self.z_affine_dim)

        utils.initseq(self.down1)
        utils.initseq(self.down2)
        utils.initseq(self.down3)
        utils.initmod(self.mu)
        utils.initmod(self.logstd)

    def sample_neighbor_euler(self, coords, hp_order=32):
        euler0 = coords[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = coords[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        euler_pixs = hp.ang2pix(hp_order//2, euler1, euler0, nest=True)

        neighbor_pix = np.random.randint(4, size=(coords.shape[0], 1)) + 4*euler_pixs[:, None]
        neighbor_pix = neighbor_pix.flatten()
        neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order, neighbor_pix, nest=True)

        neighbor_euler0 = torch.tensor(neighbor_euler0).float().to(coords.get_device())/np.pi*180 #(s)
        neighbor_euler1 = torch.tensor(neighbor_euler1).float().to(coords.get_device())/np.pi*180

        neighbor_eulers = torch.stack([neighbor_euler0, neighbor_euler1], dim=-1) #(s, neighbor, 2)
        #flatten eulers
        neighbor_eulers_flatten = neighbor_eulers.view(2) #(s*neighbor, 2)
        return neighbor_eulers_flatten

    def bicubic_translate(self, img, t, euler, scale=1.):
        assert img.shape[0] == 1
        assert len(t.shape) == 1
        W = img.shape[-1]
        img = img.view(1, W, W, W)
        rot_ref = lie_tools.zrot_2d(euler).unsqueeze(1) #(B, 1, 3, 3)
        t_grid = self.freqs2d@rot_ref + t[:2]
        img = F.grid_sample(img, t_grid, align_corners=ALIGN_CORNERS, mode='bicubic')
        H = img.shape[-1]
        img = img.reshape(1, 1, W, H, H)
        img = torch.permute(img, [0, 1, 4, 3, 2]) # z,y,x -> x,y,z
        zero = torch.zeros_like(t[2:])
        t_z = torch.cat((t[2:], zero))
        t_grid = self.freqs2d + t_z
        img = F.grid_sample(img.squeeze(0), t_grid, align_corners=ALIGN_CORNERS, mode='bicubic')
        img = img.reshape(1, 1, H, H, H)
        img = torch.permute(img, [0, 1, 4, 3, 2]) # x,y,z -> z,y,x
        return img

    def bfactor_blurring(self, img, bfactor):
        s2 = self.freqs3d.pow(2).sum(dim=-1)
        img = img*torch.exp(-s2*bfactor*np.pi**2)
        return img

    def translate_ft3d(self, img, t):
        '''
        Inputs:
            img: FT of image (B x img_dims x 3)
            t: shift in pixels (B x T x 3)
        Returns:
            Shifted images (B x T x img_dims x 2)
        '''
        # F'(k) = exp(-2*pi*k*x0)*F(k)
        coords = self.freqs3d #(D, H, W, 3)
        t = t.unsqueeze(-2).unsqueeze(-1) # BxCx1x3x1 to be able to do bmm
        #print(t.shape)
        tfilt = coords @ t * 2 * np.pi # BxCxHxWx1
        tfilt = tfilt.squeeze(-1) # BxCxHxW
        #print(coords.shape, t.shape, tfilt.shape)
        c = torch.cos(tfilt) # BxHxW
        s = torch.sin(tfilt) # BxHxN
        phase_shift = torch.view_as_complex(torch.stack([c, s], -1))#.unsqueeze(1)
        #phase_shift = phase_shift.to(img.get_device())
        #print(t.shape, img.shape, phase_shift.shape)
        return (img*phase_shift)

    def forward(self, x, rots, trans, losslist=[], eulers=None, snr=1., body_euler=None, body_trans=None, ctf_embedding=None):
        #3d to 3d suppose x is list of (N, 1, D, H, W)
        #print(len(x), eulers.get_device())
        #get x for corresponding device
        #x = torch.stack([torch.from_numpy(arr).float() for arr in x[eulers.get_device()]])
        #x = x.to(eulers.get_device(), non_blocking=True)
        x = x.unsqueeze(1)
        B = x.shape[0]
        #print(x.shape)
        if ctf_embedding is not None:
            ctf_embedding = None #self.ctf_embed_mlp(ctf_embedding).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #ctf_embedding = ctf_embedding.repeat(1, 1, self.intermediate_size, self.intermediate_size, self.intermediate_size)

        pixrad = hp.max_pixrad(128)
        encs = []
        x3d_downs = []
        x3d_center = []
        for i in range(B):
            euler_i = eulers[i,...] #(B, 3)
            euler2 = eulers[i:i+1, 2] #(B)
            euler01 = euler_i[:2] #self.sample_neighbor_euler(eulers[i:i+1, ...])
            #rot = lie_tools.euler_to_SO3(sample_euler_i)#euler_i[...,:2])#.unsqueeze(1).unsqueeze(1) #(B, 1, 1, 3, 3)
            # convert to hopf
            #hopf = lie_tools.euler_to_hopf(euler_i)
            o_rot = lie_tools.hopf_to_SO3(euler01)
            #o_rot = lie_tools.hopf_to_SO3(euler_i)
            # perturb z axis
            #rand_rot = lie_tools.random_biased_SO3(1, 3e3, o_rot.get_device())
            #rot = o_rot @ rand_rot.squeeze()

            #rot = o_rot
            #print(o_rot.shape, rand_rot.shape, o_rot @ rand_rot)
            rand_z = lie_tools.random_direction(1, pixrad*180/np.pi).to(o_rot.get_device())
            #print(lie_tools.hopf_to_SO3(lie_tools.direction_to_hopf(rand_z)), rand_rot)
            rand_z = torch.transpose(o_rot, -2, -1) @ rand_z.unsqueeze(-1)
            rand_z = rand_z.squeeze(-1)
            # to radian
            rand_e = lie_tools.direction_to_hopf(rand_z)
            #new rotation
            rot = lie_tools.hopf_to_SO3(rand_e.squeeze(0))
            #print(rand_e-euler01, torch.acos((torch.diag(o_rot.T @ rot).sum() - 1)/2)*180/np.pi)

            # minus z hopf angle == z euler angle
            euler2 = -euler2
            # if using hopf angles, use positive angle, if using euler, use negative
            #x_i = self.transformer_e.rotate_2d(x[i], -euler2, mode='bicubic') #(1, 1, H, W)

            x_i = x[i:i+1].float()
            # downsample volume if render_size < vol_size
            if self.render_size <= self.vol_size:
                x_fft = fft.torch_rfft3_center(x_i, center=True)
                # Uncomment if you want to check the FT of subtomogram
                #if i == 0:
                #    ref_fft_to_write = x_fft.abs()
                #    mrc.write("fft" + str(x_fft.get_device()) + ".mrc", ref_fft_to_write.squeeze().detach().cpu().numpy(), Apix=1., is_vol=True)
                x_fft = utils.crop_fft3d(x_fft, self.render_size)*(self.render_size/self.vol_size)**3
                x_fft = self.translate_ft3d(x_fft, -trans[i:i+1]*self.render_size/self.vol_size)
                # x_i_ori is the original image after shifting
                #x_i_ori = fft.torch_irfft3_center(x_fft, center=True)
                #x_i_ori = utils.crop_vol(x_i_ori, self.crop_vol_size)
                #x3d_center.append(x_i_ori.squeeze(1))
                b_factor = 0.5*np.random.rand() - 0.125
                x_fft_b = self.bfactor_blurring(x_fft, b_factor)
                # x_i is randomly blurred
                x_i = fft.torch_irfft3_center(x_fft_b, center=True)
            # translate volume in real space
            #trans_i_int = trans[i].round()
            #trans_i_frac = trans[i] - trans_i_int
            #t_grid = self.transformer_e.translate(-trans[i]*2./(self.crop_vol_size-1))*self.crop_scale
            #t_grid = self.transformer_e.translate(-trans[i]*self.scale/self.vol_size)*self.crop_scale
            #x_i = F.grid_sample(x[i:i+1], t_grid, align_corners=ALIGN_CORNERS)
            #print(x_i.shape, rot.T.shape)
            # translate volume in fourier space
            #x_i = fft.torch_rfft3_center(x[i:i+1])
            #x_i = self.translate_ft3d(x_i, -trans[i:i+1])
            #x_i = fft.torch_irfft3_center(x_i)
            # using bicubic sampling
            #print(trans[i])
            #x_i = self.bicubic_translate(x_i, -trans[i]*(self.render_size/self.vol_size)*2./(self.render_size - 1), -euler2, scale=1.)

            x_i = utils.crop_vol(x_i, self.crop_vol_size)

            x_i = self.transformer_e.rotate_2d(x_i.squeeze(0), -euler2, mode='bicubic').unsqueeze(0) #(1, 1, H, W)
            # standardize volume
            x_i = (x_i - x_i.mean())/x_i.std()


            # data augmentation
            if self.training:
                x_i = x_i *(1. + snr*0.3*(torch.rand(1).float().to(x.get_device()) - 0.5)) \
                    + 0.1*snr*torch.randn(1).to(x.get_device())

            # construct pseudo volume
            #x3d_i = x_i.unsqueeze(0).repeat(1, 1, self.crop_vol_size, 1, 1) #(N, C, D, H, W)
            x3d_i = x_i
            # rotate the pseudo volume
            pos = self.transformer_e.rotate(rot.T) #- trans_i_frac*self.scale/self.vol_size
            # - 2.*trans[i]/(self.render_size-1) #(B, 1, H, W, D, 3) x ( B, 1, 1, 3, 3) -> (B, 1, H, W, D, 3)

            x3d_down = F.grid_sample(x3d_i, pos, align_corners=ALIGN_CORNERS)
            x3d_downs.append(x3d_down)

            #pos = self.transformer_e.rotate(rots[i].T)
            #x3d_downp = F.grid_sample(x3d[i:i+1], pos, align_corners=ALIGN_CORNERS)
            #print((x3d_down - x3d_downp).abs().mean()/((x3d_down).abs()+(x3d_downp).abs()).mean())
            #pass through convolution nn
        x3d_downs = torch.cat(x3d_downs, dim=0)
        #x3d_center = torch.cat(x3d_center, dim=0)

        #print(x3d_downs.shape)
        # mask input
        if self.use_mask:
            x3d_downs *= self.in_mask
        # compute nearest neighbors in the same pose
        #x3d_center = x3d_downs[:, :, self.crop_vol_size//2, ...].squeeze(1).squeeze(1)

        enc1 = self.down1(x3d_downs)
        #print(enc1.shape, self.out_channels, self.out_dim)
        enc1 = F.interpolate(enc1, size=self.intermediate_size, mode="trilinear", align_corners=ALIGN_CORNERS) # 12^3
        #compute embeddings for coordinate
        #coordinate_embeddings = self.coordinate_feature.get_embedding(self.random_matrix)
        #coordinate_embeddings = self.fourier_mlp(coordinate_embeddings)
        #print(coordinate_embeddings.shape, enc1.shape)
        #enc2 = self.down2(torch.cat((enc1, ctf_embedding), dim=1))# + coordinate_embeddings)
        enc2 = self.down2(enc1)# + coordinate_embeddings)# + ctf_embedding)# )
        enc2 = enc2.view(B, self.out_dim ** 3 *self.out_channels)
        encs = self.down3(enc2)

        mu = self.mu(encs)
        if self.training:
            logstd = self.logstd(encs)
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
            #z = mu
        else:
            logstd = torch.zeros_like(mu).to(mu.device)
            z = mu

        losses = {}
        if "kldiv" in losslist:
            #losses["kldiv"] = torch.mean(mu**2, dim=-1)
            losses["mu2"] = torch.sum(mu**2, dim=-1)
            losses["std2"] = torch.sum(torch.exp(2*logstd), dim=-1)
            #losses["kldiv"] = torch.mean(- logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)
            losses["kldiv"] = torch.sum(-logstd, dim=-1) + 0.5*losses["std2"] + 0.5*losses["mu2"]

        return {"z":z, "z_mu": mu, "losses": losses, "z_logstd": logstd, "rotated_x": None, "skip": None}


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, normalize=True, use_fourier=False, mode='bilinear', render_size=180):
        super().__init__()

        self.mode = mode

        # create sampling grid
        #vectors = [torch.arange(0, s) for s in size]

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

        self.templateres = size
        self.normalize = normalize
        self.use_fourier = use_fourier
        self.render_size = render_size
        if self.normalize:
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres), indexing='ij')
        else:
            zgrid, ygrid, xgrid = np.meshgrid(np.arange(self.templateres),
                                  np.arange(self.templateres),
                                  np.arange(self.templateres), indexing='ij')
        #xgrid is the innermost dimension (-1, ..., 1)
        #self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))
        x_idx = torch.linspace(-1., 1., self.templateres) #[-s, s)
        grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        xgrid = grid[2]
        ygrid = grid[1]
        zgrid = grid[0]
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0)
        self.register_buffer("grid", grid)

        x_idx = torch.linspace(-1., 1., self.templateres//2) #[-s, s)
        grid = torch.meshgrid(x_idx, x_idx, x_idx, indexing='ij')
        xgrid = grid[2]
        ygrid = grid[1]
        zgrid = grid[0]
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0)
        self.register_buffer("grid_coarse", grid)

        self.scale = self.render_size/(self.templateres - 1)*2

        x_idx = torch.linspace(-1., 1., self.templateres) #[-s, s)
        grid  = torch.meshgrid(x_idx, x_idx, indexing='ij')
        xgrid = grid[1] #change fast [[0,1,2,3]]
        ygrid = grid[0]

        zgrid = torch.zeros_like(xgrid)
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("grid2d", grid)


    def rotate(self, rot):
        return self.grid @ rot #(1, 1, H, W, D, 3) @ (N, 1, 1, 1, 3, 3)

    def translate(self, trans):
        if trans.shape[-1] == 2:
            zero = torch.zeros_like(trans)[..., :1]
            trans = torch.cat([trans, zero], dim=-1)
        #trans = trans*self.scale #(n, 3)
        return self.grid + trans #(1, 1, H, W, D, 3) @ (N, 1, 1, 1, 3, 3)

    def translate_rotate(self, trans, rot):
        # must scale trans by yourself
        if trans.shape[-1] == 2:
            zero = torch.zeros_like(trans)[..., :1]
            trans = torch.cat([trans, zero], dim=-1)
        return (self.grid + trans) @ rot #(1, 1, H, W, D, 3) @ (N, 1, 1, 1, 3, 3)

    def multi_body_displace(self, rot_resi, coms, trans, tbody, radius=None):
        # tbody should be added to grid
        zero = torch.zeros_like(tbody)[..., :1]
        tbody = tbody*self.scale
        #tbody = torch.cat([tbody, zero], dim=-1)*self.scale #(n, 3)
        tcom = -coms.unsqueeze(1) #@ torch.transpose(rot_ori.squeeze(1).squeeze(1), -1, -2) #coms: (n, 1, 3) x rot_resi: (n, 3, 3)
        #tbody_r = (tbody.unsqueeze(1) @ rot_resi.detach()).squeeze(1)
        #print(tbody_r)
        tcom = tcom*self.scale
        rot = rot_resi.unsqueeze(1).unsqueeze(1) #(1, 1, 1, 3, 3) x (n, 1, 1, 3, 3)
        #map tbody to image frame tbody_img = tbody_r @ rot_resi.T @ rot_ori.T
        tbody_img = ((tbody).unsqueeze(1) @ torch.transpose(rot.squeeze(1).squeeze(1), -1, -2)).squeeze(1)
        #roted = (self.grid + (tcom + (trans*self.scale + tbody).unsqueeze(1)).unsqueeze(1).unsqueeze(1)) @ rot #+ (coms*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        roted = (self.grid + (tcom + (trans*self.scale).unsqueeze(1)).unsqueeze(1).unsqueeze(1)) @ rot + tbody.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        grid_size = self.templateres
        assert roted.shape == torch.Size([coms.shape[0], grid_size, grid_size, grid_size, 3])
        #resample weight to new frame
        radius = radius.unsqueeze(1).unsqueeze(1).unsqueeze(1) * self.scale
        #mask_weights = roted - (coms*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #mask_weights = torch.exp(-(roted.detach().pow(2).sum(dim=-1, keepdim=True))*0.5/radius**2) + 1e-5
        mask_weights = torch.exp(-1.5*(roted.detach().pow(2)/radius**2).sum(dim=-1, keepdim=True)) + 1e-5
        mask_weights /= mask_weights.sum(dim=0, keepdim=True)
        assert mask_weights.shape == torch.Size([coms.shape[0], grid_size, grid_size, grid_size, 1])
        body_grid = (mask_weights*(roted + (coms*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1))).sum(dim=0, keepdim=True)
        return body_grid, tbody_img

    def multi_body_grid(self, rot_ori, rot_resi, coms, trans, tbody, radius=None, axes=None, encode=False, save=None, Apix=1.):
        # tbody should be added to grid
        #zero = torch.zeros_like(tbody)[..., :1]
        #tbody = tbody*self.scale
        #tbody = torch.cat([tbody, zero], dim=-1)*self.scale #(n, 3)
        # t + com is the rotated com
        #print(tbody, coms)
        tcom = (coms).unsqueeze(1) * self.scale
        Rtcom = tcom @ torch.transpose(rot_ori.squeeze(1).squeeze(1), -1, -2) #coms: (n, 1, 3) x rot_resi: (n, 3, 3)
        #tbody_r = (tbody.unsqueeze(1) @ rot_resi.detach()).squeeze(1)
        #print(tbody_r)
        rot_axis = lie_tools.rot_to_axis(rot_resi)
        #tbody = tbody[...,:1]*rot_axis
        rot = rot_ori@rot_resi.unsqueeze(1).unsqueeze(1) #(1, 1, 1, 3, 3) x (n, 1, 1, 3, 3)
        #map tbody to image frame tbody_img = tbody_r @ rot_resi.T @ rot_ori.T
        tbody_img = ((tbody).unsqueeze(1) @ torch.transpose(rot.squeeze(1).squeeze(1), -1, -2)).squeeze(1)
        if not encode:
            #roted = (self.grid + (tcom + (trans*self.scale + tbody).unsqueeze(1)).unsqueeze(1).unsqueeze(1)) @ rot #+ (coms*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            roted = (self.grid_coarse + (-Rtcom + (trans).unsqueeze(1)).unsqueeze(1).unsqueeze(1)) @ rot + tbody.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        else:
            roted = (self.grid_coarse - (coms*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1))@torch.transpose(rot, -1, -2) \
                    - (tcom + (trans*self.scale + tbody).unsqueeze(1)).unsqueeze(1).unsqueeze(1)
        grid_size = self.templateres//2
        assert roted.shape == torch.Size([coms.shape[0], grid_size, grid_size, grid_size, 3])
        #resample weight to new frame
        #weight = F.grid_sample(weight.squeeze().unsqueeze(1), roted, align_corners=ALIGN_CORNERS).squeeze().unsqueeze(-1)
        #radius = (coms*self.scale).norm(dim=-1, keepdim=True) + 1e-5
        #raidus = radius * self.scale
        radius = radius.unsqueeze(1).unsqueeze(1).unsqueeze(1) * self.scale
        #mask_weights = roted - (coms*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #mask_weights = torch.exp(-(roted.detach().pow(2).sum(dim=-1, keepdim=True))*0.5/radius**2) + 1e-5
        if axes is not None:
            rroted = roted.detach() @ axes.unsqueeze(1).unsqueeze(1)
            mask_weights = torch.exp(-0.2*(rroted.pow(2)/radius**2).sum(dim=-1, keepdim=True)) + 1e-5
        else:
            mask_weights = torch.exp(-0.2*(roted.detach().pow(2)/radius**2).sum(dim=-1, keepdim=True)) + 1e-5
        mask_weights /= mask_weights.sum(dim=0, keepdim=True)
        assert mask_weights.shape == torch.Size([coms.shape[0], grid_size, grid_size, grid_size, 1])
        #sample_grid = self.grid_coarse @ rot_ori
        #sample_grid = torch.permute(sample_grid, dims=[0,4,1,2,3])
        #sample_grid = F.interpolate(sample_grid, size=self.templateres, mode="trilinear", align_corners=ALIGN_CORNERS)
        #sample_grid = torch.permute(sample_grid, dims=[0,2,3,4,1])
        #print((self.grid@rot_ori - sample_grid).norm())
        #body_grid = (mask_weights*(roted + (coms*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1))).sum(dim=0, keepdim=True)
        body_grid = (mask_weights*(roted + (tcom).unsqueeze(1).unsqueeze(1))).sum(dim=0, keepdim=True)
        # resample body_grid to target resolution
        body_grid = torch.permute(body_grid, dims=[0,4,1,2,3])
        body_grid = F.interpolate(body_grid, size=self.templateres, mode="trilinear", align_corners=ALIGN_CORNERS)
        body_grid = torch.permute(body_grid, dims=[0,2,3,4,1])
        #print(weight.min(), weight.max(), weight.mean(), 1./coms.shape[0])
        #valid = ((weight > 1./coms.shape[0]).float()*weight).sum(dim=0, keepdim=True).squeeze(-1).unsqueeze(1)
        #valid = (valid > 0.5).float()*valid
        #self.save_mrc(valid, "mas", Apix=Apix)
        valid = None
        if save is not None:
            tmp = ((mask_weights > 1./coms.shape[0]).float()*mask_weights).sum(dim=0)
            self.save_mrc(tmp, save, Apix=Apix)
        return body_grid, valid, tbody_img

    def save_mrc(self, template, filename, Apix=1., flip=False):
        with torch.no_grad():
            dev_id = template.get_device()
            template = template.squeeze()
            if flip:
                template = template.flip(0)
            mrc.write(filename + str(dev_id) + ".mrc", template.detach().cpu().numpy(), Apix=Apix, is_vol=True)

    def rotate_2d(self, ref, euler, out_size=None, mode='bicubic'):
        #euler (B,)
        rot_ref = lie_tools.zrot(euler).unsqueeze(1) #(B, 1, 3, 3)
        #print(ref.shape, rot_ref.shape)
        #grid (1, 1, H, W, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        #print(self.grid2d.shape, ref.shape)
        if out_size is not None:
            head = (self.render_size - out_size)//2
            tail = head + out_size
            grid_r = self.grid2d[..., head:tail, head:tail, :]
        else:
            grid_r = self.grid2d

        pos_ref = grid_r @ rot_ref
        rotated_ref = F.grid_sample(ref, pos_ref[..., :2].squeeze(0), align_corners=ALIGN_CORNERS, mode=mode)
        return rotated_ref

    def translate_2d(self, ref, trans, rot=None, out_size=None, mode='bicubic'):
        #grid (1, 1, H, W, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        if rot is not None:
            rot_ref = lie_tools.zrot(euler).unsqueeze(1)
            grid_r = self.grid2d @ rot_ref
            pos_ref = (grid_r[..., :2] + trans).squeeze(0)
        else:
            pos_ref = (self.grid2d[..., :2] + trans).squeeze(0)
        if ref.shape[0] != pos_ref.shape[0]:
            pos_ref = pos_ref.repeat(ref.shape[0], 1, 1, 1)
        rotated_ref = F.grid_sample(ref, pos_ref, align_corners=ALIGN_CORNERS, mode=mode)
        return rotated_ref

    def rotate_euler(self, ref, euler):
        # ref (1, 1, z, y, x), euler (B, 2)
        Ra = lie_tools.zrot(euler[..., 0]).unsqueeze(1) #(B, 1, 3, 3)
        Rb = lie_tools.yrot(euler[..., 1]).unsqueeze(1)
        #print(ref.shape, rot_ref.shape)
        #grid (1, 1, z, y, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        pos = self.gridz @ Ra
        # rotate around z, sample ref (1, z, y, x)
        rotated_ref = F.grid_sample(ref.squeeze(1), pos[..., :2].squeeze(0), align_corners=ALIGN_CORNERS, mode='bicubic')
        #print(pos.shape, ref.shape, rotated_ref.shape)
        # permute y axis to z
        #print(rotated_ref.shape, ref.shape)
        rotated_ref = rotated_ref.permute(dims=[0, 2, 1, 3]) # (1, y, z, x)
        # sample ref
        pos = self.gridy @ Rb
        pos = torch.stack([pos[...,0], pos[...,2]], dim=-1)
        rotated_ref = F.grid_sample(rotated_ref, pos.squeeze(0), align_corners=ALIGN_CORNERS, mode='bicubic')
        # permute again
        return rotated_ref#.permute(dims=[0, 2, 1, 3]) # (B, D, H, W)

    def sample(self, src):
        return F.grid_sample(src, self.grid, align_corners=ALIGN_CORNERS)

    def pad(self, src, out_size):
        #pad the 2d output
        src_size = src.shape[-1]
        pad_size = (out_size - src_size)//2
        if pad_size == 0:
            return src
        return F.pad(src, (pad_size, pad_size, pad_size, pad_size))

    def pad3d(self, src, out_size):
        #pad the 2d output
        src_size = src.shape[-1]
        pad_size = (out_size - src_size)//2
        if pad_size == 0:
            return src
        return F.pad(src, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size))

    def crop(self, src, out_size):
        #pad the 2d output
        src_size = src.shape[-1]
        assert out_size <= src_size
        head = (src_size - out_size)//2
        tail = head + out_size
        if head == 0:
            return src
        return src[...,head:tail, head:tail, head:tail]

    def pad_FT(self, src, out_size):
        ypad_size = (out_size - self.render_size)//2
        return F.pad(src, (0, ypad_size, ypad_size, ypad_size-1))

    def rotate_and_sample(self, src, rot):
        pos = self.rotate(rot)
        return F.grid_sample(src, pos, align_corners=ALIGN_CORNERS)


    def forward(self, src, flow):
        # new locations
        # flow (N, 3, H, W, D)
        shape = flow.shape[2:]
        flow = flow.permute(0, 2, 3, 4, 1)
        new_locs = self.grid + flow

        # need to normalize grid values to [-1, 1] for resampler
        new_locs = 2. * (new_locs/float(self.templateres - 1) - 0.5)
        #for i in range(len(shape)):
        #    new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)

        return F.grid_sample(src, new_locs, align_corners=ALIGN_CORNERS, mode=self.mode)

class VanillaDecoder(nn.Module):
    def __init__(self, D, in_vol=None, Apix=1., template_type=None, templateres=256, warp_type=None, symm_group=None,
                 ctf_grid=None, fixed_deform=False, deform_emb_size=2, zdim=8, render_size=140,
                 use_fourier=False, down_vol_size=140, tmp_prefix="ref", masks_params=None, num_bodies=0, affine_dim=4,
                 ctf_alpha=0., ctf_beta=1., normalize_ctf=False, rank=0):
        super(VanillaDecoder, self).__init__()
        self.D = D
        self.vol_size = (D - 1)
        self.Apix = Apix
        self.ctf_grid = ctf_grid
        self.template_type = template_type
        self.templateres = templateres
        self.use_conv_template = False
        self.fixed_deform = fixed_deform
        self.crop_vol_size = down_vol_size
        self.render_size = render_size
        self.use_fourier = use_fourier
        self.tmp_prefix = tmp_prefix
        self.ctf_alpha = ctf_alpha
        self.ctf_beta = ctf_beta
        self.normalize_ctf = normalize_ctf
        log("decoder: correct ctf using alpha {}, beta {}, and normalize ctf {}".format(self.ctf_alpha, self.ctf_beta, self.normalize_ctf))

        if symm_group is not None:
            self.symm_group = symm_groups.SymmGroup(symm_group)
            print(self.symm_group.symm_opsR[self.symm_group.SymsNo - 1])
            self.register_buffer("symm_ops_rot", torch.tensor([x.rotation_matrix for x in self.symm_group.symm_opsR]).float())
            self.register_buffer("symm_ops_trans", torch.tensor([x.translation_vector for x in self.symm_group.symm_opsR]).float())
            grid_size = self.templateres
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., grid_size),
                                np.linspace(-1., 1., grid_size),
                                np.linspace(-1., 1., grid_size), indexing='ij')
            #xgrid is the innermost dimension (-1, ..., 1)
            self.register_buffer("symm_grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

        else:
            self.symm_group = None

        if self.template_type == "conv":
            self.use_conv_template = True
            self.zdim = zdim
            if self.use_fourier:
                self.template = ConvTemplate(in_dim=self.zdim, outchannels=1, templateres=self.templateres)
            else:
                self.x_size = self.render_size//2 + 1
                y_idx = torch.arange(-self.x_size+1, self.x_size-1)/float(self.render_size) #[-0.5, 0.5)
                x_idx = torch.arange(self.x_size)/float(self.render_size) #(0, 0.5]
                grid  = torch.meshgrid(y_idx, y_idx, x_idx, indexing='ij')
                grid_x = grid[2] #change fast [[0,1,2,3]]
                grid_y = grid[1]
                grid_z = grid[0]
                freqs3d = torch.stack((grid_x, grid_y, grid_z), dim=-1)
                #print("freq3d decoder", freqs3d)
                #freqs3d = torch.fft.ifftshift(freqs3d, dim=(-2, -1)) # z, y, x, shift
                self.register_buffer("freqs3d", freqs3d)

                self.num_bodies = num_bodies
                self.scale = self.render_size/(self.crop_vol_size - 1)*2.
                if rank == 0:
                    log(f"decoder: the output size, aka, render_size {self.render_size}, crop_vol_size, aka, the size before padding {self.crop_vol_size}, scale {self.scale}")
                if masks_params is not None:
                    self.num_bodies = masks_params["com_bodies"].shape[0]
                    #remove com from relatives
                    self.register_buffer("in_relatives", (masks_params["in_relatives"] - masks_params["com_bodies"])/self.vol_size)
                    self.register_buffer("rotate_directions", masks_params["rotate_directions"]/self.vol_size)
                    self.register_buffer("com_bodies", masks_params["com_bodies"])
                    self.register_buffer("orient_bodies", masks_params["orient_bodies"])
                    self.register_buffer("orient_bodiesT", torch.transpose(masks_params["orient_bodies"], 1, 2).contiguous())
                    self.register_buffer("principal_axes", masks_params["principal_axes"])
                    self.register_buffer("principal_axesT", torch.transpose(masks_params["principal_axes"], 1, 2).contiguous())
                    self.register_buffer("A_rot90", lie_tools.yrot(torch.tensor(-90)))
                    #scale change the original scale in render_size to the volume size after cropping
                    self.register_buffer("radius", masks_params["radii_bodies"]/self.vol_size)
                    if rank == 0:
                        log(f"decoder: com of bodies are {self.com_bodies}, rg of bodies are {self.radius*self.scale}, scale is {self.scale}")
                        log(f"decoder: rotate_directions are {self.rotate_directions}, orient_bodies are {self.orient_bodies}, principal_axesT are {self.principal_axesT}")
                self.template = ConvTemplate(in_dim=self.zdim, templateres=self.templateres, affine=True, num_bodies=self.num_bodies, affine_dim=affine_dim)
        else:
            self.template = nn.Parameter(in_vol)

        if self.use_fourier:
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres), indexing='ij')
            mask = torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32))
            mask = mask.pow(2).sum(-1) < 0.85 ** 2
            self.register_buffer("mask_w", mask)
            ##xgrid is the innermost dimension (-1, ..., 1)
            #self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))
        else:
            # crop_vol_size is the size of volume with density of interest, which is obtained after cropping,
            # render_size is obtained after padding cropped volume
            # render_size also is equal to the size of downsampled experimental image
            self.transformer = SpatialTransformer(self.crop_vol_size, render_size=self.render_size)
            if in_vol is not None:
                if rank == 0:
                    log("decoder: loading mask {}, volume render size is {}, volume of interest is {}".format(in_vol.shape, self.render_size, self.crop_vol_size))
                #resample input mask to render_size
                mask_frac = (self.crop_vol_size - 2)/self.render_size
                crop_size = int(in_vol.shape[-1]*mask_frac)//2*2
                crop_vol = self.transformer.crop(in_vol, crop_size).unsqueeze(0).unsqueeze(0)
                #if self.num_bodies > 1:
                #    crop_vol = self.transformer.crop(masks_params["consensus_mask"], crop_size).unsqueeze(0).unsqueeze(0)

                in_vol_nonzeros = torch.nonzero(crop_vol.squeeze())
                in_vol_mins, _ = in_vol_nonzeros.min(dim=0)
                in_vol_maxs, _ = in_vol_nonzeros.max(dim=0)
                if rank == 0:
                    log("decoder: cropped mask with nonzeros between {}, {}, {}".format(in_vol_mins, in_vol_maxs, crop_vol.shape))
                in_vol_maxs = crop_vol.shape[-1] - in_vol_maxs
                self.vol_bound = torch.minimum(in_vol_maxs, in_vol_mins).float()
                self.vol_bound *= (self.templateres/crop_vol.shape[-1]) #(templateres is the size of output volume)
                self.vol_bound = self.vol_bound.int() + 1
                if rank == 0:
                    log("decoder: setting volume boundary {}".format(self.vol_bound))

                crop_vol = self.transformer.sample(crop_vol)
                if rank == 0:
                    log("decoder: cropping mask from {} to {}, cropping fraction is {}, downsample to {}".format(in_vol.shape[-1], crop_size, mask_frac, crop_vol.shape))
                self.register_buffer("ref_mask", crop_vol)
                self.register_buffer("mask_sum", (crop_vol > 0).sum().detach())

                if masks_params is not None:
                    # crop and downsample mask weights
                    if rank == 0:
                        log(f"decoder: radius of coms are {(self.com_bodies/self.vol_size*self.scale).norm(dim=-1)}")
                    #mask_weights = masks_params["weights"]
                    #assert mask_weights.shape[-1] == in_vol.shape[-1] and mask_weights.shape[-1] == self.vol_size, \
                    #        f"mask shape should agrees with consensus mask"
                    #mask_weights = self.transformer.crop(mask_weights, crop_size).unsqueeze(1)
                    #mask_weights = F.interpolate(mask_weights, size=self.crop_vol_size, mode="trilinear", align_corners=ALIGN_CORNERS)
                    #log(f"decoder: downsample masks to {mask_weights.shape}, {mask_weights.min()}, {mask_weights.max()}")
                    #self.register_buffer("mask_weights", mask_weights.squeeze().unsqueeze(-1))

                self.apix_ori = self.Apix
                self.Apix = self.vol_size/self.render_size*self.Apix
                if rank == 0:
                    log("decoder: changing apix from {} to {}".format(self.apix_ori, self.Apix))
                #self.ref_mask_com = (self.transformer.grid*self.ref_mask.unsqueeze(-1)).mean(dim=(0, 1, 2, 3, 4))
                #print(self.ref_mask_com)

            #self.register_buffer("sphere_mask", (torch.sum(self.transformer.grid ** 2, dim=-1) < 1.).float())
            #self.register_buffer("mask_sum", (torch.sum(self.sphere_mask, axis=-3) > 0).sum().detach())

        self.warp_type = warp_type

    def bfactor_blurring(self, img, bfactor):
        s2 = self.freqs3d.pow(2).sum(dim=-1)
        img = img*torch.exp(-s2*bfactor*np.pi**2)
        return img

    def tilt_blurring(self, img, bfactor):
        s2 = self.freqs3d.pow(2).sum(dim=-1)
        tilt_angle = torch.atan2(self.freqs3d[..., 2], self.freqs3d[..., 0])
        img = img*torch.exp(-s2*bfactor*np.pi**2*tilt_angle.abs())
        return img

    def translate_ft3d(self, img, t):
        '''
        Inputs:
            img: FT of image (B x img_dims x 3)
            t: shift in pixels (B x T x 3)
        Returns:
            Shifted images (B x T x img_dims x 2)
        '''
        # F'(k) = exp(-2*pi*k*x0)*F(k)
        coords = self.freqs3d #(D, H, W, 3)
        t = t.unsqueeze(-2).unsqueeze(-1) # BxCx1x3x1 to be able to do bmm
        #print(t.shape)
        tfilt = coords @ t * 2 * np.pi # BxCxHxWx1
        tfilt = tfilt.squeeze(-1) # BxCxHxW
        #print(coords.shape, t.shape, tfilt.shape)
        c = torch.cos(tfilt) # BxHxW
        s = torch.sin(tfilt) # BxHxN
        phase_shift = torch.view_as_complex(torch.stack([c, s], -1))#.unsqueeze(1)
        #phase_shift = phase_shift.to(img.get_device())
        #print(t.shape, img.shape, phase_shift.shape)
        return (img*phase_shift)

    def symmetrise_template(self, template, grid):
        B = template.shape[0]
        symm_template = template
        for i in range(self.symm_group.SymsNo - 1):
            pos = grid @ self.symm_ops_rot[i] + self.symm_ops_trans[i]
            pos = pos.repeat(B,1,1,1,1)
            symm_template = symm_template + F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
        return symm_template/float(self.symm_group.SymsNo + 1)

    def sample_symmetrised_ops(self, rots):
        B = rots.size(0)
        rand_choices = torch.randint(self.symm_group.SymsNo, (B,))
        symm_rots  = self.symm_ops_rot[rand_choices]
        #symm_trans = self.symm_ops_trans[rand_choices]
        symm_rots  = symm_rots @ rots
        #symm_trans = self.symm_trans @ rots
        return symm_rots

    def multi_body_grid(self, rot_ori, rot_resi, coms, trans, tbody, weight):
        #print(self.grid_affine_weight.shape, t.shape, rot.shape, weight_32.shape)
        zero = torch.zeros_like(tbody)[..., :1]
        tbody = torch.cat([tbody, zero], dim=-1)*self.scale #(n, 3)
        #print(rot_ori.shape, trans.shape, coms.shape)
        tcom = (trans.unsqueeze(1) - coms.unsqueeze(1) @ torch.transpose(rot_ori.squeeze(1).squeeze(1), -1, -2)) #+ coms.unsqueeze(1) #coms: (n, 1, 3) x rot_resi: (n, 3, 3)
        tcom = tcom/self.vol_size*self.scale
        rot = rot_ori@rot_resi.unsqueeze(1).unsqueeze(1) #(1, 1, 1, 3, 3) x (n, 1, 1, 3, 3)
        #print(rot_ori.shape, rot.shape, coms.shape, tcom, self.grid_affine_weight.shape, tbody)
        # tbody should be added to grid
        roted = (self.grid_affine_weight + (tcom + tbody.unsqueeze(1)).unsqueeze(1).unsqueeze(1)) @ rot \
                    + (coms/self.vol_size*self.scale).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        grid_size = self.crop_vol_size
        assert weight.shape == torch.Size([coms.shape[0], grid_size, grid_size, grid_size, 1]) \
            and roted.shape == torch.Size([coms.shape[0], grid_size, grid_size, grid_size, 3])
        grid = (weight*roted).sum(dim=0, keepdim=True)
        return grid

    def affine_mixture(self, rot, t, weight):
        #print(weight.shape) #(64, 16, 16, 16)
        weight = F.softmax(weight, dim=0)
        #weight_32 = F.grid_sample(weight.unsqueeze(0), self.grid_affine_weight, align_corners=True).squeeze(0)
        #(8, D, H, W, 3)
        #print(self.grid_affine_weight.shape, t.shape, rot.shape, weight_32.shape)
        roted = (self.grid_affine_weight) @ rot.unsqueeze(1).unsqueeze(1) \
                    + t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #print(weight_32.shape, roted.shape)
        #print(rot_mean, t_mean - roted_t_mean)
        assert weight.shape == torch.Size([64, 16, 16, 16]) and roted.shape == torch.Size([64, 16, 16, 16, 3])
        grid = (weight.unsqueeze(-1)*roted).sum(dim=0, keepdim=True)
        return grid

    def get_particle_hopfs(self, coords, hp_order=64, depth=2):
        euler0 = coords[:, 0].cpu().numpy()*np.pi/180 #(-180, 180)
        euler1 = coords[:, 1].cpu().numpy()*np.pi/180 #(0, 180)

        neighbor_pixs = hp.get_all_neighbours(hp_order//2, euler1, euler0, nest=True)

        neighbor_pixs = neighbor_pixs.flatten()
        neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
        neighbor_pixs = np.unique(neighbor_pixs)

        # sample again
        if depth == 1:
            neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
            neighbor_pixs = hp.get_all_neighbours(hp_order//2, neighbor_euler1, neighbor_euler0, nest=True)
            # keep unique indices
            neighbor_pixs = neighbor_pixs.flatten()
            neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
            neighbor_pixs = np.unique(neighbor_pixs)
            # sample again
        if depth == 2:
            neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
            neighbor_pixs = hp.get_all_neighbours(hp_order//2, neighbor_euler1, neighbor_euler0, nest=True)
            # keep unique indices
            neighbor_pixs = neighbor_pixs.flatten()
            neighbor_pixs = neighbor_pixs[neighbor_pixs != -1]
            neighbor_pixs = np.unique(neighbor_pixs)

        n_length = neighbor_pixs.shape[-1]

        neighbor_euler1, neighbor_euler0 = hp.pix2ang(hp_order//2, neighbor_pixs, nest=True)
        pixrad = hp.max_pixrad(hp_order//2)
        if depth == 2:
            n_sample = 50
        elif depth == 1:
            n_sample = 25
        else:
            n_sample = coords.shape[0]*8
        R = lie_tools.hopf_to_SO3(coords[:1, :].cpu())

        if n_length < n_sample:
            # convert to hopf
            rand_z = lie_tools.random_direction(n_sample - n_length, pixrad*4.*180/np.pi)
            rand_z = torch.transpose(R, -2, -1) @ rand_z.unsqueeze(-1)
            rand_z = rand_z.squeeze(-1)
            # to radian
            rand_e = lie_tools.direction_to_hopf(rand_z)*np.pi/180
            rand_angle0 = np.mod(rand_e[..., 0].cpu().numpy(), np.pi*2)
            rand_angle1 = rand_e[..., 1].cpu().numpy()
            #print(coords, rand_e*180/np.pi)

            #print(rand_angle0, rand_angle1, euler0, euler1)
            neighbor_euler0 = np.concatenate([neighbor_euler0, rand_angle0], axis=-1)
            neighbor_euler1 = np.concatenate([neighbor_euler1, rand_angle1], axis=-1)

        neighbor_euler0 = torch.tensor(neighbor_euler0).float().to(coords.get_device())/np.pi*180 #(s)
        neighbor_euler1 = torch.tensor(neighbor_euler1).float().to(coords.get_device())/np.pi*180

        neighbor_eulers = torch.stack([neighbor_euler0, neighbor_euler1], dim=-1) #(s, neighbor, 2)
        #flatten eulers
        neighbor_eulers_flatten = neighbor_eulers.view(-1, 2) #(s*neighbor, 2)
        return neighbor_eulers_flatten

    def lasso(self, template):
        head = self.vol_bound[2]
        tail  = template.shape[-1] - head + 1
        head_y = self.vol_bound[1]
        tail_y  = template.shape[-1] - head_y + 1
        head_z = self.vol_bound[0]
        tail_z  = template.shape[-1] - head_z + 1
        #print(head, head_y, head_z, tail, tail_y, tail_z)
        assert head >=1 and head_y >=1 and head_z >=1 and head < tail and head_y < tail_y and head_z < tail_z
        cropped_template = template[:, :, head_z:tail_z, head_y:tail_y, head:tail].abs()
        #return ((cropped_template) + 1e-1).log()
        return cropped_template

    def total_variation(self, template, vol_bound=None, sqrt=True):
        head = vol_bound[2]
        tail  = template.shape[-1] - head + 1
        head_y = vol_bound[1]
        tail_y  = template.shape[-1] - head_y + 1
        head_z = vol_bound[0]
        tail_z  = template.shape[-1] - head_z + 1
        assert head >=1 and head_y >=1 and head_z >=1 and head < tail and head_y < tail_y and head_z < tail_z
        out = (template[:, :, head_z:tail_z, head_y:tail_y, head:tail] - template[:, :, head_z:tail_z, head_y:tail_y, head-1:tail-1])**2 + \
                  (template[:, :, head_z:tail_z, head_y:tail_y, head:tail] - template[:, :, head_z:tail_z, head_y-1:tail_y-1, head:tail])**2 + \
                  (template[:, :, head_z:tail_z, head_y:tail_y, head:tail] - template[:, :, head_z-1:tail_z-1, head_y:tail_y, head:tail])**2
        if sqrt:
            #return (torch.sqrt(1e-8 + out) + 1e-2).log()
            return torch.sqrt(1e-8 + out)
        else:
            return out

    def forward(self, rots, trans, z=None, in_template=None, euler=None, ref_fft=None, ctf_param=None,
                save_mrc=False, refine_pose=True, body_euler=None, body_trans=None, ctf_grid=None, estimate_pose=False,
                ctf_filename=None, write_ctf=False, bfactor=2., snr=1.):

        #ref_fft = torch.stack([torch.from_numpy(arr).float() for arr in ref_fft[euler.get_device()]])
        #ref_fft = ref_fft.to(euler.get_device(), non_blocking=True)
        #ctf_param = torch.stack([torch.from_numpy(arr).float() for arr in ctf_param[euler.get_device()]])
        #ctf_param = ctf_param.to(euler.get_device(), non_blocking=True)

        #generate a projection
        if self.use_conv_template:
            #print((z[0] == z[1]).sum())
            template, affine = self.template(z)
        elif in_template is not None:
            template = in_template
        else:
            template = self.template.unsqueeze(0).unsqueeze(0)

        # downsample template
        #template = F.interpolate(template, size=200, mode="trilinear", align_corners=ALIGN_CORNERS)
        #template = F.interpolate(template, size=self.crop_vol_size, mode="trilinear", align_corners=ALIGN_CORNERS)
        #template_FT = fft.torch_rfft3_center(template)
        #left = (template.shape[-1] - self.crop_vol_size) // 2
        #right = self.crop_vol_size + left
        #scaled_xdim = self.crop_vol_size//2 + 1
        #template_FT = template_FT[..., left:right, left:right, :scaled_xdim] * (self.crop_vol_size/template.shape[-1]) ** 3
        #template = fft.torch_irfft3_center(template_FT)

        losses = {}
        if self.training:
            losses["l2"] = torch.mean(self.lasso(template)).unsqueeze(0)
            losses["tvl2"] = torch.mean(self.total_variation(template, vol_bound=self.vol_bound)).unsqueeze(0) #torch.tensor(0.).to(template.get_device())
        else:
            losses["l2"] = torch.tensor(0.).to(z.device)
            losses["tvl2"] = torch.tensor(0.).to(z.device)

        if self.symm_group is not None:
            #rots = self.sample_symmetrised_ops(rots)
            template = self.symmetrise_template(template, self.symm_grid)

        images = []
        refs = []
        masks = []
        mask_sums = []
        euler_samples = []
        B = rots.shape[0]
        #theta = np.zeros((3,4), dtype=np.float32)

        #valid = self.sphere_mask #(torch.sum(self.transformer.grid ** 2, dim=-1) < 1.).float()
        if not refine_pose:
            mask_i = self.ref_mask.repeat(ref_fft.shape[1], 1, 1, 1, 1)
            if len(euler.shape) == 2:
                euler = euler.unsqueeze(1)

        body_rots_pred = []
        body_rots = []
        body_trans_pred = []
        #rand_tilt = (torch.randn(1).to(euler.get_device()))/6.*10.
        rand_tilt = torch.tensor([0.]).to(euler.get_device())
        # precompute ctf here
        freqs = ctf_grid.freqs2d.unsqueeze(0)/self.Apix
        #ctf_param[:, :, 0] += rand_tilt

        if ctf_param.shape[-1] == 9:
            c = ctf.compute_3dctfaniso(ref_fft, ctf_grid.centered_freqs, freqs, *torch.split(ctf_param, 1, -1), Apix=self.Apix, plot=False, use_warp=self.normalize_ctf)
        else:
            c = ctf.compute_3dctf(ref_fft, ctf_grid.centered_freqs, freqs, *torch.split(ctf_param, 1, -1), Apix=self.Apix, plot=False, use_warp=self.normalize_ctf)

        for i in range(B):
            mask_sums.append(self.mask_sum)
            ref_i = ref_fft[i:i+1,...]
            #down sample and shift subtomogram
            x_fft = fft.torch_rfft3_center(ref_i.float(), center=True)
            x_fft = utils.crop_fft3d(x_fft, self.render_size)*(self.render_size/self.vol_size)**3
            ref_i_ft = self.translate_ft3d(x_fft, -trans[i:i+1]*self.render_size/self.vol_size)

            #ctf correction by phase flipping and sqrt of ctf, sqrt(ctf)*ctf.sign()
            #print(ref_i_ft.shape, c.shape)
            #ref_i_ft *= torch.sign(c[i:i+1])
            ref_i_ft *= c[i:i+1].abs().pow(self.ctf_alpha)*torch.sign(c[i:i+1]) # 1, Z, Y, X, * 1, Z, Y, X
            bfactor_shift = np.random.randn()/3.*bfactor*0.05
            ref_i_ft = self.bfactor_blurring(ref_i_ft, -(bfactor-bfactor_shift)/self.Apix**2)#/np.exp((bfactor-bfactor_shift)*np.pi**2/9.**2)
            ref_i = fft.torch_irfft3_center(ref_i_ft, center=True)
            ref_i = utils.crop_vol(ref_i, self.crop_vol_size)

            if refine_pose:
                euler_i = euler[i:i+1,...] #(B, 3)
                #R = lie_tools.euler_to_SO3(euler_i)
                #euler_i = lie_tools.euler_to_hopf(euler_i)

                #validate hopf
                #Rh = lie_tools.hopf_to_SO3(euler_i[..., :2])
                #Rz = lie_tools.hopf_to_SO3(euler_i[..., 2:])
                #Rz1 = lie_tools.zrot(-euler_i[..., 2])
                #print(R, Rz - Rz1, Rz @ Rh)

                rand_ang = (torch.rand(1, 1).to(euler.get_device()) - .5)*360
                #zero euler2
                #rand_ang = torch.zeros(1, 1).to(euler.get_device())
                # random ang is just eluer2, which means we rotate the 3d volume to target pose directly !
                #rand_ang = euler_i[:, 2:]

                euler2_sample_i = -euler_i[:, 2] #(1,) hopf angle == minus of euler angle
                euler2 = euler2_sample_i + rand_ang.squeeze(1) # use minus if using euler

                euler01 = euler_i[..., :2]
                neighbor_eulers = self.get_particle_hopfs(euler01, hp_order=32, depth=0) #hp_order=64, depth=0)

                len_euler = neighbor_eulers.shape[0]
                #n_eulers = torch.cat([neighbor_eulers, rand_ang.repeat(len_euler, 1)], dim=-1)
                #rot = lie_tools.hopf_to_SO3(n_eulers).unsqueeze(1).unsqueeze(1)

                #reset rot to local samples
                zero_eulers = torch.zeros_like(euler01)
                local_sample = self.get_particle_hopfs(zero_eulers, hp_order=64, depth=0) #hp_order=64, depth=0)
                rot = lie_tools.hopf_to_SO3(local_sample).unsqueeze(1).unsqueeze(1)

                i_euler = torch.cat([euler_i[..., :2], rand_ang], dim=-1)
                rot_i = lie_tools.hopf_to_SO3(i_euler).unsqueeze(1).unsqueeze(1)
                t_i = trans[i:i+1, ...]
                #zero the translation, since we have translated it in the encoder
                t_i -= t_i#.round()

                # get the residual rotation, and body trans
                if self.num_bodies > 1:
                    body_quat_i = affine[0][i, ...]
                    #body_trans_i = affine[1][i, ...]/self.vol_size
                    one = torch.ones_like(affine[1][i, :, :1])*16.
                    body_trans_i = torch.cat([one, affine[1][i, ...]], dim=-1)
                    body_trans_i = lie_tools.quaternions_to_SO3_wiki(body_trans_i)


                    if body_euler is not None:
                        body_euler_i = body_euler[i, ...] #(N_body, 3)
                        body_trans_i_exp = body_trans[i, ...] #(N_body, 2)
                        rot_resi_i_exp = lie_tools.euler_to_SO3(body_euler_i)
                        rot_resi_i_exp = self.orient_bodiesT @ self.A_rot90 @ rot_resi_i_exp @ self.orient_bodies
                        body_rots.append(rot_resi_i_exp)
                    #print(rot_resi_i)
                    rot_resi_i = lie_tools.quaternions_to_SO3_wiki(body_quat_i)

                    # global rotation corrected by mlp
                    rot_i = rot_i @ rot_resi_i[self.num_bodies:, ...].unsqueeze(1).unsqueeze(1)
                    # global translation corrected by mlp
                    global_trans_i = affine[1][i, self.num_bodies:, ...]
                    #print(rot_resi_i, body_trans_i)
                    rot_i_correction = lie_tools.so3_to_hopf(rot_resi_i[self.num_bodies:,...])
                    #save estimated global pose
                    body_rots_pred.append(rot_i_correction)
                    body_trans_pred.append(global_trans_i)

                    rot_resi_i = self.orient_bodiesT @ rot_resi_i[:self.num_bodies, ...] @ self.orient_bodies
                    #rot_resi_i = self.principal_axesT @ rot_resi_i[:self.num_bodies, ...] @ self.principal_axes
                    # rotate com according to rotate direction
                    body_trans_i = self.orient_bodiesT @ body_trans_i[:self.num_bodies, ...] @ self.orient_bodies
                    body_trans_i = (body_trans_i @ self.rotate_directions.unsqueeze(-1)) - self.rotate_directions.unsqueeze(-1)
                    body_trans_i = body_trans_i.squeeze(-1)
                    #zero_eulers = torch.zeros_like(euler01)
                    #local_sample = self.get_particle_hopfs(zero_eulers, hp_order=32, depth=0) #hp_order=64, depth=0)
                    #rot = lie_tools.hopf_to_SO3(local_sample).unsqueeze(1).unsqueeze(1)
                    #rot = torch.transpose(rot_i, -1, -2) @ rot
                    #print(lie_tools.so3_to_hopf(rot.squeeze()), lie_tools.so3_to_hopf(rot_resi_i))
                    #convert to multibody field
                    #t_i = trans[i:i+1, ...]
                    #t_i -= t_i.round()
                    #transform to the scale of cropped volume
                    t_i_3d = t_i/self.vol_size*self.scale
                    #zero = torch.zeros_like(t_i[..., :1])
                    #transform to the scale of cropped volume
                    #t_i_3d = torch.cat([t_i, zero], dim=-1)/self.vol_size*self.scale #(n, 3)
                    affine_grid_i, valid, trans_img = self.transformer.multi_body_grid(rot_i, rot_resi_i, self.com_bodies/self.vol_size,
                                                                     t_i_3d, body_trans_i, radius=self.radius, axes=self.principal_axesT)

                    #body_trans_pred.append(trans_img[..., :3]*self.vol_size/self.scale)
                    pos = self.transformer.rotate(rot_i)
                    valid = F.grid_sample(self.ref_mask, pos, align_corners=ALIGN_CORNERS)
                    #if save_mrc and i == 0:
                    #    self.save_mrc(valid, self.tmp_prefix+"mas")
                else:
                    assert self.num_bodies == 0
                    # rotation
                    body_quat_i = affine[0][i, ...]
                    rot_resi_i = lie_tools.quaternions_to_SO3_wiki(body_quat_i)
                    # rotation corrected by mlp
                    if estimate_pose:
                        rot_i = rot_i @ rot_resi_i[self.num_bodies:, ...].unsqueeze(1).unsqueeze(1)
                    # translation
                    global_trans_i = affine[1][i, self.num_bodies:, ...]
                    # transform the estimate global translation to experimental reference system
                    #R_global_trans_i = rot_i @ global_trans_i.unsqueeze(-1)
                    #print(R_global_trans_i.shape, rot_i.shape, global_trans_i.shape)

                    #apply estimated global rotation and translation in this step
                    #pos = self.transformer.translate_rotate(t_i/self.vol_size*self.scale, rot_i)
                    #position for resample mask
                    if estimate_pose:
                        #print(global_trans_i)
                        pos = self.transformer.rotate(rot_i) + global_trans_i/self.vol_size*self.scale
                    else:
                        pos = self.transformer.rotate(rot_i)
                    #valid = F.grid_sample(self.sphere_mask.unsqueeze(1), pos, align_corners=ALIGN_CORNERS)
                    valid = F.grid_sample(self.ref_mask, pos, align_corners=ALIGN_CORNERS)
                    # convert body_rots_pred to hopf_angles
                    rot_i_correction = lie_tools.so3_to_hopf(rot_resi_i[self.num_bodies:,...])
                    body_rots_pred.append(rot_i_correction)
                    body_trans_pred.append(global_trans_i)
                    #body_trans_pred.append(R_global_trans_i.squeeze() + trans[i:i+1, ...])

                #print(euler2.shape, neighbor_eulers.shape, rot.shape)
                #ref_i = ref_fft[i:i+1,...].repeat(euler2.shape[0], 1, 1, 1)
                # convert neighbor_eulers to a list
                euler_sample_i = neighbor_eulers#.repeat(2, 1)
                euler2_sample_i = euler2_sample_i.unsqueeze(0).repeat(len_euler, 1).view(len_euler, -1)
                euler_sample_i = torch.cat([euler_sample_i, -euler2_sample_i], dim=1)
                #euler_samples.append(euler_sample_i)

                template_i = template[i:i+1,...].repeat(rot.shape[0], 1, 1, 1, 1)
                if self.num_bodies > 1:
                    pos = affine_grid_i @ rot + global_trans_i/self.vol_size*self.scale
                else:
                    # if you want to translate the 3D reconstruction
                    # rot represents the local perturbation
                    #pos = self.transformer.translate_rotate(t_i/self.vol_size*self.scale, rot_i @ rot)
                    # create the 3d grid for reconstruction
                    #print(global_trans_delta)
                    if estimate_pose:
                        pos = self.transformer.rotate(rot_i @ rot) + global_trans_i/self.vol_size*self.scale
                    else:
                        pos = self.transformer.rotate(rot_i @ rot)
                    #pos = self.transformer.translate_rotate(t_i/self.vol_size*self.scale, rot_i @ rot)
            else:
                raise RuntimeError("Not implemented")

            # rotate reference
            if self.num_bodies >= 0:
                #ref: (1, D, H, W)
                ###rotate tilt around y, permute ref_i from [z, y, x] to [y, z, x], then convert back
                #ref_i = torch.permute(ref_i, [0, 2, 1, 3,])
                ##rotate_2d using lie_tools.zrot, which is the inverse of lie_tools.rot_2d
                ##but in compute3dctf, we use -tilt, so rotate_2d is identical to compute3dctf
                ##so if the ref is rotated by rand_tilt using rotate_2d, the 3dctf should be rotated by tilt + rand_tilt
                #ref_i = self.transformer.rotate_2d(ref_i, rand_tilt, mode='bicubic')
                #ref_i = torch.permute(ref_i, [0, 2, 1, 3])

                ##rotate the images around z axis by -(-euler2) == euler2 counter-clockwise
                ref_i = self.transformer.rotate_2d(ref_i, -euler2, mode='bicubic') #.unsqueeze(0)
                #applying 3d mask to reference here
                ref = ref_i*(valid>0)
                ref = ref.squeeze(1)
                #randomly change contrast
                ref *= (1. + (np.random.rand() - 0.5)*snr)
                # if you want to translate the 2d experimental image
                #ref = self.transformer.translate_2d(ref_i, -t_i/self.vol_size*2.).squeeze(1)
            else:
                raise RuntimeError("num_bodies should not be negative")
                #ref = self.transformer.rotate_2d(ref_i, -euler2).squeeze(1)
            refs.append(ref.sum(dim=-3))
            #pos = self.transformer.rotate(rot) # + 1)/2*(self.crop_vol_size - 1) #(B, 1, H, W, D, 3) x ( B, 1, 1, 3, 3) -> (B, 1, H, W, D, 3)
            #mask_i = (torch.sum(valid, axis=-3) > 0).detach().squeeze(1)
            #masks.append(mask_i)

            # sample reference image from template
            vol = F.grid_sample(template_i, pos, align_corners=ALIGN_CORNERS)
            #vol = self.transformer.rotate_euler(template_i, euler_i)
            #vol = vol*valid
            #image = torch.sum(vol, axis=-3)#.squeeze(1)
            image = self.transformer.pad3d(vol, self.render_size)
            #print('image shape: ', image.shape)
            # if you want to translate the 2d reconstruction
            #image = self.transformer.translate_2d(image, (t_i)/self.vol_size*2.)
            image = image.squeeze(1)
            # append sampled angles
            euler_samples.append(euler_sample_i)

            image = self.transformer.pad3d(image, self.render_size)
            image_fft = fft.torch_rfft3_center(image, center=True)
            #print(image_fft.shape, c.shape, ref.shape)

            #image_fft: 8, D, H, W; c: 1, D, H, W; ref: 1, crop_D, crop_H, crop_W
            ctf_beta_rand = (np.random.randn()/4.)*0.02
            image_fft = image_fft*c[i:i+1].abs().pow(self.ctf_beta + ctf_beta_rand)
            image_fft = self.bfactor_blurring(image_fft, (bfactor+bfactor_shift)/2./self.Apix**2)#*max(np.sqrt((bfactor+bfactor_shift)/2./self.Apix**2)**3, 0.6)/np.power(0.5, ctf_beta_rand)
            #*ctf_beta*2.
            image_fft = self.tilt_blurring(image_fft, (bfactor+bfactor_shift)/2./self.Apix**2)
            image_fft = fft.torch_irfft3_center(image_fft, center=True)
            image_fft = utils.crop_vol(image_fft, self.crop_vol_size)
            image = utils.crop_vol(image, self.crop_vol_size)

            if "y_recon2" not in losses:
                losses["y_recon2"] = [(image_fft**2).sum(dim=(-1,-2,-3)).view(-1)]
                losses["ycorr"] = [(-2.*image_fft*ref).sum(dim=(-1,-2,-3)).view(-1)]
                losses["y2"] = [(ref**2).sum(dim=(-1,-2,-3)).view(-1)]
            else:
                losses["y_recon2"].append((image_fft**2).sum(dim=(-1,-2,-3)).view(-1))
                losses["ycorr"].append((-2.*image_fft*ref).sum(dim=(-1,-2,-3)).view(-1))
                losses["y2"].append((ref**2).sum(dim=(-1,-2,-3)).view(-1))

            images.append(image.sum(dim=-3).detach())

        #B, 8
        losses["y_recon2"] = torch.stack(losses["y_recon2"], 0)
        losses["ycorr"] = torch.stack(losses["ycorr"], 0) #B, 1
        losses["y2"] = torch.stack(losses["y2"], 0)
        #print(losses["y_recon2"].shape, losses["ycorr"].shape, losses["y2"].shape)
        images = torch.stack(images, 0)
        refs   = torch.stack(refs, 0)
        #print("images and refs shape: ", images.shape, refs.shape)
        #masks  = torch.stack(masks, 0)
        mask_sums = torch.stack(mask_sums, 0)
        if len(euler_samples):
            euler_samples = torch.stack(euler_samples, 0)
        if len(body_rots_pred) and (estimate_pose or self.num_bodies > 0):
            if len(body_rots):
                body_rots = torch.stack(body_rots, 0)
            body_rots_pred = torch.stack(body_rots_pred, 0)
            body_trans_pred = torch.stack(body_trans_pred, 0)
            #print(body_rots_pred.shape, body_trans_pred.shape,)
            body_poses_pred = [body_rots_pred, body_trans_pred]
            #body_poses_pred = [body_rots_pred, body_rots, body_trans_pred, body_trans]
        else:
            body_poses_pred = None
        # pad to original size
        #if not self.use_fourier:
        #    images = self.transformer.pad3d(images, self.render_size)
        #    images_fft = fft.torch_rfft3_center(images)
        #    # compute ctf here
        #    freqs = ctf_grid.freqs2d.unsqueeze(0)/self.Apix
        #    c = ctf.compute_3dctf(images_fft, ctf_grid.centered_freqs, freqs, *torch.split(ctf_param, 1, -1), Apix=self.Apix, plot=False)
        #    # Uncomment if you want to check the 3DCTF
        #    #c_to_write = torch.fft.fftshift(c[:1,...], dim=(-3, -2,))
        #    #mrc.write("ctf" + str(c.get_device()) + ".mrc", c_to_write.squeeze().detach().cpu().numpy(), Apix=self.Apix/self.apix_ori, is_vol=True)
        #    images_fft = images_fft*c.unsqueeze(1)
        #    images_fft = fft.torch_irfft3_center(images_fft)
        #    images_fft = utils.crop_vol(images_fft, self.crop_vol_size)
        #    images = utils.crop_vol(images, self.crop_vol_size)
        #    # compute reconstruction loss locally
        #    losses["y_recon2"] = (images_fft**2).sum(dim=(-1,-2,-3)).view(B, -1)
        #    losses["ycorr"] = (-2.*images_fft*refs).sum(dim=(-1,-2,-3)).view(B, -1)
        #    losses["y2"] = (refs**2).sum(dim=(-1,-2,-3)).view(B, -1)

        if write_ctf:
            for f_i in range(len(ctf_filename)):
                if not os.path.exists(ctf_filename[f_i]):
                    c_to_write = c[f_i]
                    mrc.write(ctf_filename[f_i], c_to_write.squeeze().detach().cpu().numpy(), Apix=self.Apix, is_vol=True)
        if save_mrc:
            self.save_mrc(template[0:1, ...], self.tmp_prefix, flip=False)
        return {"y_recon_ori": None, "y_recon": images ,"losses": losses, "y_ref": refs, "mask_sum": mask_sums,
                "affine": body_poses_pred, "ctf_3d": None}

    def save_mrc(self, template, filename, flip=False):
        with torch.no_grad():
            dev_id = template.get_device()
            if self.use_fourier:
                #template_FT = fft.torch_rfft3_center(template)
                #the origin is at self.templateres//2 - 1
                start = (self.templateres - self.vol_size)//2 - 1
                template_FT = template[..., start:start+self.vol_size, start:start+self.vol_size, \
                                      self.templateres//2-1:self.templateres//2+self.vol_size//2]
                template_FT = template_FT*(self.vol_size/self.templateres)**3
                #print(template_FT.shape)
                template = fft.torch_irfft3_center(template_FT)
            elif self.transformer.templateres != self.vol_size:
                template = self.transformer.sample(template)
            template = template.squeeze(0).squeeze(0)
            if flip:
                template = template.flip(0)
            mrc.write(filename + str(dev_id) + ".mrc", template.detach().cpu().numpy(), Apix=self.Apix, is_vol=True)

    @torch.no_grad()
    def save(self, filename, z=None, encoding=None, flip=False, Apix=1.):
        if self.template_type == "conv":
            template, affine = self.template(z)
            print(template.shape)
            if affine is not None:
                body_quat_i = affine[0][0, ...]
                one = torch.ones_like(affine[1][0, :, :1])*16.
                body_trans_i = torch.cat([one, affine[1][0, ...]], dim=-1)
                body_trans_i = lie_tools.quaternions_to_SO3_wiki(body_trans_i)

                i_euler = torch.zeros(1, 2).to(body_quat_i.get_device())
                rot_i = lie_tools.hopf_to_SO3(i_euler).unsqueeze(1).unsqueeze(1)

                rot_resi_i = lie_tools.quaternions_to_SO3_wiki(body_quat_i)
                # multiply rot_i by an estimated global rot
                #global_trans_i = affine[1][0, self.num_bodies:, ...]

                rot_resi_i = self.orient_bodiesT @ rot_resi_i[:self.num_bodies, ...] @ self.orient_bodies
                # rotate com according to rotate direction
                body_trans_i = self.orient_bodiesT @ body_trans_i[:self.num_bodies, ...] @ self.orient_bodies
                body_trans_i = (body_trans_i @ self.rotate_directions.unsqueeze(-1)) - self.rotate_directions.unsqueeze(-1) #+ self.in_relatives.unsqueeze(-1)
                body_trans_i = body_trans_i.squeeze(-1)
                zero_3d = torch.zeros(1, 3).to(body_quat_i.get_device())
                affine_grid_i, valid, trans_img = self.transformer.multi_body_grid(rot_i, rot_resi_i, self.com_bodies/self.vol_size,
                                                                                zero_3d, body_trans_i, radius=self.radius,)
                template = F.grid_sample(template, affine_grid_i, align_corners=ALIGN_CORNERS)

            elif self.transformer.templateres != self.templateres:
                #resample
                template = self.transformer.sample(template)
                if self.ref_mask is not None:
                    mask = self.transformer.sample(self.ref_mask)
                    template *= mask # apply mask
            template = template.squeeze(0).squeeze(0)
        else:
            template = self.template
        if flip:
            template = template.flip(0)
        mrc.write(filename + ".mrc", template.detach().cpu().numpy(), Apix=Apix, is_vol=True)

    def get_vol(self, z=None):
        if self.template_type == "conv":
            template, _ = self.template(z)
            if self.transformer.templateres != self.vol_size:
                #resample
                template = self.transformer.sample(template)
        else:
            template = self.template
        return template

    def get_images(self, z, rots, trans):
        if self.template_type == "conv":
            template, _ = self.template(z)
        else:
            template = self.template
        B = rots.shape[0]
        images = []
        for i in range(B):
            pos = self.transformer.rotate(rots[i])
            valid = (torch.sum(pos ** 2, dim=-1) < 1.).float()

            vol = F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
            vol *= valid
            image = torch.sum(vol, axis=-3)
            image = image.squeeze(0)
            images.append(image)
        images = torch.stack(images, 0)
        if self.transformer.templateres != self.vol_size:
            images = self.transformer.pad(images, self.vol_size)
        images = self.translate(images, trans)
        return images


class PositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            freqs = self.DD*np.pi*(2./self.DD)**(freqs/(self.enc_dim-1)) # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD*np.pi*(1./self.DD/np.pi)**(freqs/(self.enc_dim-1)) # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2*(1./self.D2)**(freqs/(self.enc_dim-1)) # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2*(2.*np.pi/self.D2)**(freqs/(self.enc_dim-1)) # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float)
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, coords):
        '''Input should be coordinates from [-.5,.5]'''
        assert (coords[...,0:3].abs() - 0.5 < 1e-4).all()
        return self.decoder(self.positional_encoding_geom(coords))

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.forward(x)
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol = fft.ihtn_center(vol_f[0:-1,0:-1,0:-1]) # remove last +k freq for inverse FFT
        return vol

class FTPositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            freqs = self.DD*np.pi*(2./self.DD)**(freqs/(self.enc_dim-1)) # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD*np.pi*(1./self.DD/np.pi)**(freqs/(self.enc_dim-1)) # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2*(1./self.D2)**(freqs/(self.enc_dim-1)) # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2*(2.*np.pi/self.D2)**(freqs/(self.enc_dim-1)) # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float)
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        # if ignore_DC = False, then the size of the lattice will be odd (since it
        # includes the origin), so we need to evaluate one additional pixel
        c = lattice.shape[-2]//2 # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c # include the origin
        assert abs(lattice[...,0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,0:cc,:])
        image[..., 0:cc] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,cc:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        assert (lattice[...,0:3].abs() - 0.5 < 1e-4).all()
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
        result = self.decoder(self.positional_encoding_geom(lattice))
        result[...,1][w] *= -1 # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        assert extent <= 0.5
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            keep = x.pow(2).sum(dim=1) <= extent**2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x,z.expand(x.shape[0],zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[...,0] - y[...,1]
                slice_ = torch.zeros(D**2, device='cpu')
                slice_[keep] = y.cpu()
                slice_ = slice_.view(D,D).numpy()
            vol_f[i] = slice_
        vol_f = vol_f*norm[1]+norm[0]
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class FTSliceDecoder(nn.Module):
    '''
    Evaluate a central slice out of a 3D FT of a model, returns representation in
    Hartley reciprocal space

    Exploits the symmetry of the FT where F*(x,y) = F(-x,-y) and only
    evaluates half of the lattice. The decoder is f(x,y,z) => real, imag
    '''
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation):
        '''D: image width or height'''
        super(FTSliceDecoder, self).__init__()
        self.decoder = ResidLinearMLP(in_dim, nlayers, hidden_dim, 2, activation)
        D2 = int(D/2)

        ### various pixel indices to keep track of for forward_even
        self.center = D2*D + D2
        self.extra = np.arange((D2+1)*D, D**2, D) # bottom-left column without conjugate pair
        # evalute the top half of the image up through the center pixel
        # and extra bottom-left column (todo: just evaluate a D-1 x D-1 image so
        # we don't have to worry about this)
        self.all_eval = np.concatenate((np.arange(self.center+1), self.extra))

        # pixel indices for the top half of the image up to (but not incl)
        # the center pixel and excluding the top row and left-most column
        i, j = np.meshgrid(np.arange(1,D),np.arange(1,D2+1))
        self.top = (j*D+i).ravel()[:-D2]

        # pixel indices for bottom half of the image after the center pixel
        # excluding left-most column and given in reverse order
        i, j =np.meshgrid(np.arange(1,D),np.arange(D2,D))
        self.bottom_rev = (j*D+i).ravel()[D2:][::-1].copy()

        self.D = D
        self.D2 = D2

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        assert lattice.shape[-2] % 2 == 1
        c = lattice.shape[-2]//2 # center pixel
        assert lattice[...,c,0:3].sum() == 0.0, '{} != 0.0'.format(lattice[...,c,0:3].sum())
        assert abs(lattice[...,0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,0:c+1,:])
        image[..., 0:c+1] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,c+1:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def forward_even(self, lattice):
        '''Extra bookkeeping with extra row/column for an even sized DFT'''
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,self.all_eval,:])
        image[..., self.all_eval] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,self.bottom_rev] = top_half[...,self.top,0] + top_half[...,self.top,1]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
        result = self.decoder(lattice)
        result[...,1][w] *= -1 # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.decode(x)
                y = y[...,0] - y[...,1]
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol_f = utils.zero_sphere(vol_f)
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class VAE(nn.Module):
    def __init__(self,
            lattice,
            qlayers, qdim,
            players, pdim,
            encode_mode = 'mlp',
            no_trans = False,
            enc_mask = None
            ):
        super(VAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D*lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 2
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, qdim)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(self.in_dim,
                            qlayers-2, # -2 bc we add 2 more layers in the homeomorphic encoer
                            qdim,  # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        elif encode_mode == 'mlp':
            self.encoder = MLP(self.in_dim,
                            qlayers-2,
                            qdim, # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        # predict rotation and translation in two completely separate NNs
        #self.so3_encoder = SO3reparameterize(qdim) # hidden_dim -> SO(3) latent variable
        #self.trans_encoder = ResidLinearMLP(nx*ny, 5, qdim, 4, nn.ReLU)

        # or predict rotation/translations from intermediate encoding
        self.so3_encoder = SO3reparameterize(qdim, 1, qdim) # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(qdim, 1, qdim, 4, nn.ReLU)

        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, img):
        '''img: BxDxD'''
        img = img.view(img.size(0),-1)
        if self.enc_mask is not None:
            img = img[:,self.enc_mask]
        enc = nn.ReLU()(self.encoder(img))
        z_mu, z_std = self.so3_encoder(enc)
        if self.no_trans:
            tmu, tlogvar = (None, None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:,:2], z[:,2:]
        return z_mu, z_std, tmu, tlogvar

    def eval_volume(self, norm):
        return self.decoder.eval_volume(self.lattice.coords, self.D, self.lattice.extent, norm)

    def decode(self, rot):
        # transform lattice by rot.T
        x = self.lattice.coords @ rot # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)
        return y_hat

    def forward(self, img):
        z_mu, z_std, tmu, tlogvar = self.encode(img)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        # transform lattice by rot and predict image
        y_hat = self.decode(rot)
        if not self.no_trans:
            # translate image by t
            B = img.size(0)
            t = self.reparameterize(tmu, tlogvar)
            t = t.unsqueeze(1) # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B,-1), t)
            img = img.view(B,self.D, self.D)
        return y_hat, img, z_mu, z_std, w_eps, tmu, tlogvar

class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(ResidLinearMLP, self).__init__()
        layers = [ResidLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(ResidLinear(hidden_dim, out_dim) if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout)
        #self.linear = nn.utils.weight_norm(nn.Linear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z

class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class SO3reparameterize(nn.Module):
    '''Reparameterize R^N encoder output to SO(3) latent variable'''
    def __init__(self, input_dims, nlayers=None, hidden_dim=None):
        super().__init__()
        if nlayers is not None:
            self.main = ResidLinearMLP(input_dims, nlayers, hidden_dim, 9, nn.ReLU)
        else:
            self.main = nn.Linear(input_dims, 9)

        # start with big outputs
        #self.s2s2map.weight.data.uniform_(-5,5)
        #self.s2s2map.bias.data.uniform_(-5,5)

    def sampleSO3(self, z_mu, z_std):
        '''
        Reparameterize SO(3) latent variable
        # z represents mean on S2xS2 and variance on so3, which enocdes a Gaussian distribution on SO3
        # See section 2.5 of http://ethaneade.com/lie.pdf
        '''
        # resampling trick
        if not self.training:
            return z_mu, z_std
        eps = torch.randn_like(z_std)
        w_eps = eps*z_std
        rot_eps = lie_tools.expmap(w_eps)
        #z_mu = lie_tools.quaternions_to_SO3(z_mu)
        rot_sampled = z_mu @ rot_eps
        return rot_sampled, w_eps

    def forward(self, x):
        z = self.main(x)
        z1 = z[:,:3].double()
        z2 = z[:,3:6].double()
        z_mu = lie_tools.s2s2_to_SO3(z1,z2).float()
        logvar = z[:,6:]
        z_std = torch.exp(.5*logvar) # or could do softplus
        return z_mu, z_std



