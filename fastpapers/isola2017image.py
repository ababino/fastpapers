# AUTOGENERATED! DO NOT EDIT! File to edit: 01_isola2017image.ipynb (unless otherwise specified).

__all__ = ['gen_bce_l1_loss', 'gen_bce_loss', 'crit_bce_loss', 'crit_real_bce', 'crit_fake_bce', 'Patch70',
           'UnetUpsample', 'CGenerator']

# Cell
from fastai.data.external import untar_data
from fastai.data.transforms import get_image_files
from fastai.data import *
from fastai.basics import *
from fastai.vision.data import *
from fastai.vision.core import *
from fastcore.all import *
from fastai.vision.augment import *
from fastai.vision.gan import *
from fastai.vision.models import *
from fastai.vision.models.unet import *
from fastai.callback.hook import *
from fastai.vision.widgets import *
from fastprogress import progress_bar, master_bar
from .core import *
import seaborn as sns

# Cell
def gen_bce_l1_loss(fake_pred, output, target, l1_weight=100, bce_weight=1):
    l1_loss = nn.L1Loss()(output[-1],target[-1])
    ones = fake_pred.new_ones(*fake_pred.shape)
    bce_loss= nn.BCEWithLogitsLoss()(fake_pred, ones)
    return bce_weight*bce_loss + l1_weight*l1_loss

def gen_bce_loss(learn, output, target):
    fake_pred = learn.model.critic(output)
    ones = fake_pred.new_ones(*fake_pred.shape)
    bce = nn.BCEWithLogitsLoss()(fake_pred, ones)
    return bce

# Cell
def crit_bce_loss(real_pred, fake_pred):
    ones  = real_pred.new_ones(*real_pred.shape)
    zeros = fake_pred.new_zeros(*fake_pred.shape)
    loss_neg = nn.BCEWithLogitsLoss()(fake_pred, zeros)
    loss_pos = nn.BCEWithLogitsLoss()(real_pred, ones)
    return (loss_neg + loss_pos)/2

def crit_real_bce(learn, real_pred, input):
    ones  = real_pred.new_ones(*real_pred.shape)
    rbce = nn.BCEWithLogitsLoss()(real_pred, ones)
    return rbce

def crit_fake_bce(learn, real_pred, input):
    fake = learn.model.generator(input).requires_grad_(False)
    fake_pred = learn.model.critic(fake)
    zeros = fake_pred.new_zeros(*fake_pred.shape)
    fbce = nn.BCEWithLogitsLoss()(fake_pred, zeros)
    return fbce

# Cell
def Patch70(n_channels):
    layers = []
    layers.append(ConvLayer(n_channels, 64, ks=4, stride=2, norm_type=None, bias=False,
                           act_cls=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
    layers.append(ConvLayer(64, 128, ks=4, stride=2, norm_type=NormType.Batch, bias=False,
                           act_cls=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
    layers.append(ConvLayer(128, 256, ks=4, stride=2, norm_type=NormType.Batch, bias=False,
                           act_cls=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
    layers.append(ConvLayer(256, 512, ks=4, stride=1, norm_type=NormType.Batch, bias=False,
                           act_cls=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
    layers.append(nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)))
    return nn.Sequential(*layers)

# Cell
class UnetUpsample(Module):
    def __init__(self, ni, nout, hook, ks, padding, dropout=False):
        self.hook = hook
        self.upsample = ConvLayer(ni, nout, ks=ks, stride=2, norm_type=NormType.Batch,
                                  transpose=True, padding=padding, bias=False)
        if dropout:
            layers = list(self.upsample.children())
            layers.append(nn.Dropout(0.5))
            self.upsample = nn.Sequential(*layers)
    def forward(self, x):
        return torch.cat([self.upsample(x), self.hook.stored], dim=1)

class CGenerator(SequentialEx):
    def __init__(self, n_channels, out_channels, enc_l=5):
        encoder = []
        encoder.append(ConvLayer(n_channels, 64, ks=4, stride=2, norm_type=None, bias=False,
                                 act_cls=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
        ni = 64
        for i in range(enc_l):
            nout = min(ni*2, 512)
            encoder.append(ConvLayer(ni, nout, ks=4, stride=2, norm_type=NormType.Batch, bias=False,
                                     act_cls=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)))
            ni = nout
        nout = min(ni*2, 512)
        encoder.append(ConvLayer(ni, nout, ks=4, stride=2, norm_type=NormType.Batch, bias=False, padding=1))# act_cls=None
        ni = nout
        hooks = hook_outputs(encoder[:-1])
        decoder = []
        for i, (l, h) in enumerate(zip(encoder[-2::-1], hooks[::-1])):
            nout = first(l.children()).out_channels
            ks = 4
            padding = 1
            dropout = i<enc_l-3
            decoder.append(UnetUpsample(ni, nout, h, ks, padding, dropout=dropout))
            ni = 2*nout
        nout = out_channels
        decoder.append(ConvLayer(ni, nout, ks=4, stride=2, norm_type=NormType.Batch, transpose=True, padding=1, bias=True, act_cls=nn.Tanh))
        layers = encoder + decoder
        super().__init__(*layers)
    def forward(self, x):
        return super().forward(x)
