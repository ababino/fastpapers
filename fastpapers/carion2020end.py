# AUTOGENERATED! DO NOT EDIT! File to edit: 04_carion2020end.ipynb (unless otherwise specified).

__all__ = ['coco_vocab', 'ParentSplitter', 'box_cxcywh_to_xyxy', 'box_xyxy_to_cxcywh', 'TensorBBoxWH', 'TensorBBoxTL',
           'ToWH', 'ToXYXY', 'ToTL', 'box_area', 'all_op', 'generalized_box_iou', 'DETRLoss', 'DETR', 'CocoEval',
           'sorted_detr_trainable_params', 'GetAnnotatedImageFiles', 'GetBboxAnnotation', 'GetClassAnnotation',
           'CocoDataLoaders', 'detr_learner']

# Cell
import os
import torch
import numpy as np
import seaborn as sns
import io
from contextlib import redirect_stdout
from IPython.core.debugger import set_trace
from torch import functional as F
from scipy.optimize import linear_sum_assignment
from fastprogress.fastprogress import master_bar, progress_bar
from fastai.data.all import *
from fastai.vision.all import *
from .core import *
from itertools import chain
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from .core import _parent_idxs

# Cell
coco_vocab = [
    'N/A0', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A1',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A2', 'backpack',
    'umbrella', 'N/A3', 'N/A4', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A5', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A6', 'dining table', 'N/A7',
    'N/A8', 'toilet', 'N/A9', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A10',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Cell
def ParentSplitter(train_name='train', valid_name='valid'):
    "Split `items` from the grand parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        tindex = _parent_idxs(o, train_name)
        vindex = _parent_idxs(o, valid_name)
        return tindex, vindex
    return _inner

# Cell
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# Cell
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# Cell
class TensorBBoxWH(TensorPoint): pass
class TensorBBoxTL(TensorPoint): pass

# Cell
@Transform
def ToWH(x:TensorBBox): return TensorBBoxWH(box_xyxy_to_cxcywh(x*0.5+0.5), img_size=x.img_size)

# Cell
@Transform
def ToXYXY(x:TensorBBoxWH)->None:
    return TensorBBox(box_cxcywh_to_xyxy(x)*2-1, img_size=x.img_size)

# Cell
class ToTL(Transform):
    def encodes(self, x:TensorBBoxWH)->None: return TensorBBoxTL(box_cxcywh_to_xyxy(x), img_size=x.img_size)
    def encodes(self, x:TensorBBox)->None: return TensorBBoxTL((x+1)/2, img_size=x.img_size)

# Cell
def box_area(boxes): return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

# Cell
def all_op(cmp):
    "Compares all the elements of `a` and `b` using cmp."
    def _inner(a, b):
        if not is_iter(b): return False
        return all(cmp(a_,b_) for a_,b_ in itertools.zip_longest(a,b))
    return _inner

# Cell
def generalized_box_iou(boxes1, boxes2, pairwise=False):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2).
    This implemenation expects bs as first dim.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    #pexpt((boxes1, boxes2))
    #set_trace()
    boxes1, boxes2 = ToTL()((boxes1, boxes2))
    #pexpt((boxes1, boxes2))
    assert (boxes1[..., 2:] >= boxes1[..., :2]).all(), 'boxes1 are not in [left_x, top_y, right_x, bottom_y] coords'
    assert (boxes2[..., 2:] >= boxes2[..., :2]).all(), 'boxes2 are not in [left_x, top_y, right_x, bottom_y] coords'
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    if pairwise:
        boxes1 = boxes1[:, :, None, :]
        boxes2 = boxes2[:, None, :, :]
        area1 = area1[:, :, None]
        area2 = area2[:, None, :]
    lt = torch.max(boxes1[..., :2], boxes2[..., :2])  # [N,M,2]
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [N,M]
    union = (area1 + area2) - inter
    iou = inter / union

    lt = torch.min(boxes1[..., :2], boxes2[..., :2])  # [N,M,2]
    rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[..., 0] * wh[..., 1]

    #set_trace()
    return iou - (area - union) / area

# Cell
class DETRLoss(nn.Module):
    def __init__(self, classw=1, boxw=1, giouw=1, n_queries=100, th=0.7, eos_coef=0.1, n_classes=92):
        super().__init__()
        store_attr()
        self.emptyw = torch.ones(n_classes)
        self.emptyw[-1] = eos_coef
        self.entropy = nn.CrossEntropyLoss(weight=self.emptyw)

    def class_loss(self, output_classes, target_id, indices):
        bs, nq, nc = output_classes.shape
        target_id_full = torch.full((bs, nq), nc-1, dtype=torch.int64, device=target_id.device)
        for i, ind in enumerate(indices): target_id_full[i, ind[0]] = target_id[i, ind[1]]
        return self.entropy(output_classes.transpose(1,2), target_id_full)

    def box_loss(self, output_boxes, target_boxes, indices):
        output_boxes, target_boxes = ToWH((output_boxes, target_boxes))

        output_boxes_ind = []
        target_boxes_ind = []
        for i, (src, dst) in enumerate(indices):
            output_boxes_ind.append(output_boxes[i, src, :])
            target_boxes_ind.append(target_boxes[i, dst, :])
        output_boxes_ind = torch.cat(output_boxes_ind)
        target_boxes_ind = torch.cat(target_boxes_ind)
        l1_loss = nn.L1Loss()(output_boxes_ind, target_boxes_ind)
        giou = 1 - generalized_box_iou(output_boxes_ind, target_boxes_ind)
        return self.boxw * l1_loss + self.giouw * giou.mean()

    def box_cost(self, output_boxes, target_boxes):
        output_boxes, target_boxes = ToWH((output_boxes, target_boxes))
        return torch.cdist(output_boxes, target_boxes, p=1)

    def class_cost(self, output_class, target_ids):
        bs, nq, _ = output_class.shape
        _, mc = target_ids.shape
        p = output_class.flatten(0,1).softmax(-1) # [bs*nq, num_classes]
        ids = target_ids.flatten() # [bs*nq]
        loss = -p[:, ids].reshape(bs, nq, -1) # [bs, nq, bs*mc]
        return torch.cat([loss[i, :, i*mc:(i+1)*mc][None, ...] for i in range(bs)], 0) # [bs, nq, mc]


    @torch.no_grad()
    def matcher(self, output, target):
        output_boxes, output_class = output # [bs, nq, 4], [bs, nq, num_classes]
        target_boxes, target_ids = target # [bs, max(n in batch), 4], [bs, max(n in batch)]

        l_iou = -generalized_box_iou(output_boxes, target_boxes, pairwise=True)
        l_box = self.box_cost(output_boxes, target_boxes)
        l_class = self.class_cost(output_class, target_ids)

        C = self.classw*l_class + self.boxw*l_box + self.giouw*l_iou
        C = C.cpu()
        sizes = [(v<self.n_classes-1).type(torch.int).sum() for v in target[1]]
        Cs = [C[i, :, :s] for i, s in enumerate(sizes)]

        indices = [linear_sum_assignment(C[i, :, :s]) for i, s in enumerate(sizes)]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def forward(self, output, target_boxes, target_ids):
        output_boxes, output_class, aux_outputs = output
        indices = self.matcher((output_boxes, output_class), (target_boxes, target_ids))
        l_class = self.class_loss(output_class, target_ids, indices)
        l_box = self.box_loss(output_boxes, target_boxes, indices)
        loss = l_class * self.classw  + l_box
        if aux_outputs:
            for output in aux_outputs:
                output_boxes, output_class = output['pred_boxes'], output['pred_logits']
                indices = self.matcher((output_boxes, output_class), (target_boxes, target_ids))
                l_class = self.class_loss(output_class, target_ids, indices)
                l_box = self.box_loss(output_boxes, target_boxes, indices)
                loss += l_class * self.classw  + l_box

        return loss

    def activation(self, x): return (ToXYXY(x[0]), F.softmax(x[1], dim=-1))

    def decodes(self, x, pad=True):
        pred_boxes, probs = x
        max_probs, pred_ids = probs.max(axis=-1)
        ind = (max_probs>self.th) & (pred_ids<probs.shape[-1]-1) & (box_area(pred_boxes)>0)

        max_probs = [max_probs[i, ind[i]] for i in range(ind.shape[0])]
        pred_ids = [pred_ids[i, ind[i]] for i in range(ind.shape[0])]
        #pred_boxes = L([pred_boxes[i, ind[i], :] for i in range(ind.shape[0])]).map(TensorBBox)
        pred_boxes = L(pred_boxes[i, ind[i], :] for i in range(ind.shape[0]))
        if pad:
            imgs = [None for i in range_of(pred_ids)]
            z_inp = zip(imgs ,pred_boxes, pred_ids)
            out = bb_pad(list(z_inp), pad_idx=self.n_classes-1)
            pred_boxes = torch.cat([x[1].unsqueeze(0) for x in out])
            pred_ids = torch.cat([x[2].unsqueeze(0) for x in out])
            pred_boxes, pred_ids = TensorBBox(pred_boxes), TensorMultiCategory(pred_ids)
        self.scores = max_probs
        return pred_boxes, pred_ids

# Cell
class DETR(nn.Module):
    def __init__(self, pretrained=True, n_classes=92, aux_loss=False):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=pretrained)
        if self.model.class_embed.out_features!=n_classes:
            self.model.class_embed = nn.Linear(256, n_classes)
        self.model.aux_loss = aux_loss
    def forward(self, x):
        img_sz = x.shape[2:]
        x = self.model(x)

        pred_boxes, pred_logits = x['pred_boxes'], x['pred_logits']
        aux_outputs = x.get('aux_outputs', None)
        if aux_outputs:
            for o in aux_outputs: o['pred_boxes'] = TensorBBoxWH(o['pred_boxes'], img_size=img_sz)
        return TensorBBoxWH(pred_boxes, img_size=img_sz), pred_logits, aux_outputs

# Cell
class CocoEval(Callback):
    run_before=Recorder
    run_train = False
    def __init__(self):
        metrics = 'AP AP50 AP75 AP_small AP_medium AP_large AR1 AR10 AR100 AR_small AR_medium AR_large'.split()
        self.metrics = L(metrics).map(partial(getattr, self)).map(ValueMetric)

    def before_validate(self):
        vocab = self.dls.vocab
        bs = self.learn.dls.bs
        self.gt_ds = {'annotations': [], 'images': [], 'categories': []}
        self.dt_ds = {'annotations': [], 'images': [], 'categories': []}
        self.gt_ds['categories'] = [{'id': i+1,'name':o} for i,o in enumerate(vocab)]
        self.dt_ds['categories'] = [{'id': i+1,'name':o} for i,o in enumerate(vocab)]

        self.reset_counters()
        self.bs = bs
        self.dec_bbox = compose(ToXYXY, to_cpu, self.learn.dls.after_item.decode)#
        self.dec_cls = compose(to_cpu, lambda x: x[x>0])

        self.batch_to_samples = compose(partial(batch_to_samples, max_n=self.bs), L)
    def reset_counters(self):
        self.img_id = Inf.count
        self.gtann = Inf.count
        self.dtann = Inf.count
    def after_batch(self):
        pred_boxes, pred_ids = self.learn.loss_func.decodes(self.loss_func.activation(self.pred), pad=False)
        max_probs = self.learn.loss_func.scores
        _, _, w, h = self.xb[0].shape
        gt_cls = self.batch_to_samples(self.yb[1]).map(to_cpu)
        dt_cls = L(pred_ids).map(to_cpu)
        gt_boxes = self.batch_to_samples(self.yb[0]).map(self.dec_bbox)
        dt_boxes = L(pred_boxes).map(self.dec_bbox)
        for gtb, gtc, dtb, dtc, i, socres in zip(gt_boxes, gt_cls, dt_boxes, dt_cls, self.img_id, max_probs):
            self.gt_ds['images'].append({'id': i, 'height': h, 'width': w})
            self.gt_ds['annotations'].extend([{'iscrowd': 0, 'bbox': o.tolist(), 'area': box_area(o), 'category_id': int(c), 'image_id': i, 'id': j} for o, c, j in zip(gtb, gtc, self.gtann)])
            self.dt_ds['images'].append({'id': i, 'height': h, 'width': w})
            self.dt_ds['annotations'].extend([{'iscrowd': 0, 'score': s, 'bbox': o.tolist(), 'area': box_area(o), 'category_id': int(c), 'image_id': i, 'id': j} for o, c, j, s in  zip(dtb, dtc, self.dtann, socres)])

    def after_validate(self):
        with redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = self.gt_ds
            gt.createIndex()
            dt = COCO()
            dt.dataset = self.dt_ds
            dt.createIndex()
            coco_eval = COCOeval(gt, dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            self.stats = coco_eval.stats
            self.reset_counters()
    def AP(self): return self.stats[0]
    def AP50(self): return self.stats[1]
    def AP75(self): return self.stats[2]
    def AP_small(self): return self.stats[3]
    def AP_medium(self): return self.stats[4]
    def AP_large(self): return self.stats[5]
    def AR1(self): return self.stats[6]
    def AR10(self): return self.stats[7]
    def AR100(self): return self.stats[8]
    def AR_small(self): return self.stats[9]
    def AR_medium(self): return self.stats[10]
    def AR_large(self): return self.stats[11]

# Cell
@typedispatch
def show_results(x:TensorImage, y:tuple, samples, outs, ctxs=None, max_n=6,
                 nrows=None, ncols=1, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(2*len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize, double=True,
                                     title='Target/Prediction')
    for i in [0, 2]:
        ctxs[::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs[::2],range(2*max_n))]
    ctxs[1::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(samples.itemgot(0),ctxs[1::2],range(2*max_n))]
    ctxs[1::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(outs.itemgot(1),ctxs[1::2],range(2*max_n))]

    return ctxs

# Cell
def sorted_detr_trainable_params(m):
    named_params = (L(m.named_parameters())).map(L).sorted(itemgetter(0))

    backbone_mask = named_params.map(itemgetter(0)).map(Self.startswith('model.backbone'))
    input_proj_mask = named_params.map(itemgetter(0)).map(Self.startswith('model.input_proj'))
    transformer_enc_mask = named_params.map(itemgetter(0)).map(Self.startswith('model.transformer.encoder'))
    transformer_dec_mask = named_params.map(itemgetter(0)).map(Self.startswith('model.transformer.decoder'))

    query_embed_mask = named_params.map(itemgetter(0)).map(Self.startswith('model.query_embed'))

    bbox_head_mask = named_params.map(itemgetter(0)).map(Self.startswith('model.bbox_embed'))
    class_head_mask = named_params.map(itemgetter(0)).map(Self.startswith('model.class_embed'))

    transformer_enc = named_params[transformer_enc_mask].itemgot(1)
    transformer_dec = named_params[transformer_dec_mask].itemgot(1)
    query_embed = named_params[query_embed_mask].itemgot(1)
    input_proj = named_params[input_proj_mask].itemgot(1)
    backbone = named_params[backbone_mask].itemgot(1)
    bbox_head = named_params[bbox_head_mask].itemgot(1)
    class_head = named_params[class_head_mask].itemgot(1)

    return L(backbone + input_proj, transformer_enc + transformer_dec + query_embed, bbox_head + class_head)

# Cell
class GetAnnotatedImageFiles:
    def __init__(self, img2bbox): self.img2bbox = img2bbox
    def __call__(self, x): return compose(get_image_files, partial(filter, compose(attrgetter('name'), self.img2bbox.__contains__)), L)(x)
class GetBboxAnnotation:
    def __init__(self, img2bbox): self.img2bbox = img2bbox
    def __call__(self, x): return compose(attrgetter('name'), self.img2bbox.__getitem__, itemgetter(0))(x)
class GetClassAnnotation:
    def __init__(self, img2bbox): self.img2bbox = img2bbox
    def __call__(self, x): return compose(attrgetter('name'), self.img2bbox.__getitem__, itemgetter(1))(x)

# Cell
class CocoDataLoaders(DataLoaders):
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_path(cls, path, train='train', valid='val', vocab=None, bs=16, item_tfms=Resize(800), batch_tfms=None, **kwargs):
        source = Path(path)
        ann_files = source.ls(file_exts='.json')
        train_ann = ann_files.filter(lambda x: x.name.startswith(train))
        assert len(train_ann)==1, 'More than one (or none) training annotation file'
        val_ann = ann_files.filter(lambda x: x.name.startswith(valid))
        assert len(val_ann)<2, 'More than one validation annotation file'
        ann_files = [train_ann[0]]
        if val_ann: ann_files.append(val_ann[0])

        img2bbox = {}
        for ann_file in ann_files: img2bbox = merge(img2bbox, dict(zip(*get_annotations(ann_file))))

        if not vocab: vocab = L(chain(*L(img2bbox.values()).itemgot(1))).unique()
        if not '#na#' in vocab:
            vocab = L(vocab) + '#na#'
        elif '#na#'!=vocab[-1]:
            warn('Empty category #na# should be the last element of the vocab.')
            warn('Moving category #na# at the end of vocab.')
            vocab.pop(vocab.index('#na#'))
            vocab = L(vocab) + '#na#'

        img_folders = source.ls().filter(Self.is_dir())
        train_name = img_folders.filter(Self.name.startswith(train))
        val_name = img_folders.filter(Self.name.startswith(valid))
        assert len(train_name)==1
        train_name = train_name[0].name
        if len(ann_files)==2:
            assert len(val_name)==1
            val_name = val_name[0].name
            splitter = ParentSplitter(train_name=train_name, valid_name=val_name)
        else:
            splitter = RandomSplitter()

        BBoxBlock.dls_kwargs = {'before_batch': partial(bb_pad, pad_idx=len(vocab)-1)}

        dblock = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock(vocab=list(vocab), add_na=False)),
                 get_items=GetAnnotatedImageFiles(img2bbox),
                 splitter=splitter,
                 get_y=[GetBboxAnnotation(img2bbox), GetClassAnnotation(img2bbox)],
                 item_tfms=item_tfms,
                 batch_tfms=batch_tfms,
                 n_inp=1)
        return cls.from_dblock(dblock, source, bs=bs, **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_sources(cls, sources, vocab=None, bs=16, item_tfms=Resize(800), batch_tfms=None, **kwargs):
        ann_files = [sources['train_ann'], sources['val_ann']]
        img2bbox = {}
        for ann_file in ann_files: img2bbox = merge(img2bbox, dict(zip(*get_annotations(ann_file))))

        if not vocab: vocab = L(chain(*L(img2bbox.values()).itemgot(1))).unique()
        if not '#na#' in vocab:
            vocab = L(vocab) + '#na#'
        elif '#na#'!=vocab[-1]:
            warn('Empty category #na# should be the last element of the vocab.')
            warn('Moving category #na# at the end of vocab.')
            vocab.pop(vocab.index('#na#'))
            vocab = L(vocab) + '#na#'

        splitter = ParentSplitter(train_name=sources['train'].name, valid_name=sources['val'].name)

        BBoxBlock.dls_kwargs = {'before_batch': partial(bb_pad, pad_idx=len(vocab)-1)}

        dblock = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock(vocab=list(vocab), add_na=False)),
                 get_items=GetAnnotatedImageFiles(img2bbox),
                 splitter=splitter,
                 get_y=[GetBboxAnnotation(img2bbox), GetClassAnnotation(img2bbox)],
                 item_tfms=item_tfms,
                 batch_tfms=batch_tfms,
                 n_inp=1)
        return cls.from_dblock(dblock, sources['base'], bs=bs, **kwargs)

# Cell
def detr_learner(dls, pretrained=True, bs=16):
    model = DETR(pretrained=pretrained, n_classes=len(dls.vocab), aux_loss=True)
    loss = DETRLoss(classw=1, boxw=5, giouw=2).cuda()
    ce = CocoEval()
    learn = Learner(dls, model, loss, splitter=sorted_detr_trainable_params,
                    cbs=[ce], metrics=ce.metrics,
                    opt_func=partial(Adam, decouple_wd=True))
    learn.coco_eval = ce
    return learn