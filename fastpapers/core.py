# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['explode_types', 'explode_lens', 'explode_shapes', 'explode_ranges', 'pexpt', 'pexpl', 'pexps',
           'receptive_fields', 'ImageNTuple', 'ImageTupleBlock', 'ConditionalGenerator', 'SiameseCritic', 'GenMetric',
           'CriticMetric', 'l1', 'l1', 'ProgressImage', 'download_file_from_google_drive', 'save_response_content',
           'FID_WEIGHTS_URL']

# Cell
import requests
from fastcore.all import *
from fastai.data.all import *
from fastai.vision.core import *
from fastai.vision.data import *
from fastai.basics import *
from fastai.vision.gan import *
from fastai.vision.models.all import *
from fastai.vision.augment import *
from fastai.callback.hook import *
from fastai.vision.widgets import *

# Cell
def explode_types(o):
    '''Like fastcore explode_types, but only shows __name__ of type.'''
    if not is_listy(o): return type(o).__name__
    return {type(o).__name__: [explode_types(o_) for o_ in o]}

# Cell
def explode_lens(o):
    if is_listy(o):
        if all(is_listy(o_) for o_ in o):
            return [explode_lens(o_) for o_ in o]
        else: return len(o)

# Cell
def explode_shapes(o):
    if not is_listy(o): return tuple(bind(getattr, arg0, 'shape')(o))
    return [explode_shapes(o_) for o_ in o]

# Cell
def explode_ranges(o):
    if not is_listy(o): return (float(o.min()), float(o.max()))
    return [explode_ranges(o_) for o_ in o]

# Cell
def pexpt(o): print(explode_types(o))

# Cell
def pexpl(o): print(explode_lens(o))

# Cell
def pexps(o): print(explode_shapes(o))

# Cell
def receptive_fields(model, nf, imsize, bs=64):
    '''returns the size of the receptive field for each feature output.'''
    x = dummy_eval(model, imsize).detach()
    outsz = x.shape[-2:]
    for p in model.named_parameters():
        if 'weight' in p[0]:
            n = p[1].shape[1] if len(p[1].shape)==4 else 1
            nn.init.constant_(p[1], 1./n)
        elif 'bias' in p[0]:
            nn.init.constant_(p[1], 0)

    with torch.no_grad():
        rfs = []
        model.eval()
        model = model.cuda()
        t = torch.eye(imsize[0]**2).reshape(imsize[0]*imsize[1], 1, imsize[0], imsize[1])
        for i,batch in enumerate(chunked(t, bs)):
            new = torch.cat(batch, dim=0).unsqueeze(1)
            new = torch.cat([new for _ in range(nf)], dim=1).cuda()
            rfs.append(model(new).cpu())
    rfs = torch.cat(rfs, dim=0).squeeze()
    rfs = rfs.reshape(imsize[0], imsize[1], outsz[0], outsz[1])
    rfs = (rfs>0.99).sum(axis=(0,1)).float().sqrt()
    return rfs

# Cell
class ImageNTuple(fastuple):
    @classmethod
    def create(cls, fns): return cls(tuple(PILImage.create(f) for f in fns))

    def show(self, ctx=None, **kwargs):
        all_tensors = all([isinstance(t, Tensor) for t in self])
        same_shape = all([self[0].shape==t.shape for t in self[1:]])
        if not all_tensors or not same_shape: return ctx
        line = self[0].new_zeros(self[0].shape[0], self[0].shape[1], 10)
        imgs = sum(L(zip(self, [line]*len(self))).map(list),[])[:-1]
        return show_image(torch.cat(imgs, dim=2), ctx=ctx, **kwargs)

    def requires_grad_(self, value):
        for item in self: item.requires_grad_(value)
        return self

    def detach(self):
        for item in self: item.detach()
        return self


# Cell
def ImageTupleBlock():
    '''Like fastai tutoria siemese transform, but uses ImageNTuple.'''
    return TransformBlock(type_tfms=ImageNTuple.create, batch_tfms=[IntToFloatTensor])

# Cell
class ConditionalGenerator(nn.Module):
    '''Wraper around a GAN generator that returns the generated image and the input.'''
    def __init__(self, gen):
        super().__init__()
        self.gen = gen
    def forward(self, x):
        if is_listy(x):
            input = torch.cat(x, axis=1)
        else:
            input = x
        return ImageNTuple(x, TensorImage(self.gen(input)))

# Cell
class SiameseCritic(Module):
    def __init__(self, critic): self.critic = critic
    def forward(self, x): return self.critic(torch.cat(x, dim=1))

# Cell
class GenMetric(AvgMetric):
    def accumulate(self, learn):
        if learn.model.gen_mode:
            bs = find_bs(learn.yb)
            self.total += to_detach(self.func(learn, learn.pred, *learn.yb))*bs
            self.count += bs

# Cell
class CriticMetric(AvgMetric):
    def accumulate(self, learn):
        if not learn.model.gen_mode:
            bs = find_bs(learn.yb)
            self.total += to_detach(self.func(learn, learn.pred, *learn.yb))*bs
            self.count += bs

# Cell
def l1(learn, output, target): return nn.L1Loss()(output[-1], target[-1])
l1 = GenMetric(l1)

# Cell
class ProgressImage(Callback):
    @delegates(show_image)
    def __init__(self, out_widget, save_img=False, folder='pred_imgs', conditional=False, **kwargs):
        self.out_widget = out_widget
        self.kwargs = kwargs
        self.save_img = save_img
        self.folder = folder
        self.conditional = conditional
        if self.conditional:
            self.title = 'input/real/fake'
        else:
            self.title = 'generated'
        Path(self.folder).mkdir(exist_ok=True)
    def before_fit(self):
        with self.out_widget:
            self.fig, self.ax = plt.subplots()
            self.ax.axis('off')
    def before_batch(self):
        if self.gan_trainer.gen_mode and self.training: self.last_gen_target = self.learn.yb[0][-1]
    def after_epoch(self):
        "Show a sample image."
        if not hasattr(self.learn.gan_trainer, 'last_gen'): return
        b = self.learn.gan_trainer.last_gen
        gt = self.last_gen_target
        gt, b = self.learn.dls.decode((gt, b))
        gt, imt = batch_to_samples((gt, b), max_n=1)[0]
        if self.conditional:
            imt = ToTensor()(ImageNTuple.create((imt[0], gt, imt[-1])))
        self.out_widget.clear_output(wait=True)
        with self.out_widget:
            self.ax.clear()
            imt.show(ax=self.ax, title=self.title, **self.kwargs)
            display(self.fig)
        if self.save_img: self.fig.savefig(self.path / f'{self.folder}/pred_epoch_{self.epoch}.png')

# Cell
@typedispatch
def show_results(x:TensorImage, y:ImageNTuple, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    max_n = min(x.shape[0], max_n)
    if max_n<ncols: ncols = max_n
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs):
        title = 'Input-Real-Fake'
        ImageNTuple(x[i], y[1][i], outs[i][0][1]).show(ctx=ctx, title=title)

# Cell
@patch
def show_results(self:GANLearner, ds_idx=1, dl=None, max_n=9, shuffle=True, **kwargs):
    if dl is None: dl = self.dls[ds_idx].new(shuffle=shuffle)
    b = dl.one_batch()
    _,_,preds = self.get_preds(dl=[b], with_decoded=True)
    preds = (preds,)
    self.dls.show_results(b, preds, max_n=max_n, **kwargs)

# Cell
URLs.FACADES = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz'
URLs.FACADES_BASE = 'http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip'
URLs.FACADES_EXTENDED = 'http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip'
URLs.CELEBA = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'

# Cell
def download_file_from_google_drive(file_id, destination, folder_name=None):
    if folder_name:
        dst = Config()['data'] / folder_name
        if dst.exists():
            return dst
    else:
        dst = Config()['data']
    arch_dst = Config()['archive'] / destination
    if not arch_dst.exists():
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params = { 'id' : file_id }, stream = True)
        token = first([(k,v) for k,v in response.cookies.items() if k.startswith('download_warning')])[1]
        if token:
            params = { 'id' : file_id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        save_response_content(response, Config()['archive'] / destination)
    file_extract(Config()['archive'] / destination, Config()['data'])
    return dst

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

# Cell
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'