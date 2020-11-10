# Welcome to fastpapers
> Play LEGO with papers.


`fastpapers` is a python library where I reproduce papers on [Jupyter Notebooks](https://jupyter.org/). I use [nbdev](https://nbdev.fast.ai/) to turn these notebooks into modules. The implementations are done using [fastai](https://docs.fast.ai/). This a personal project that I use to learn and it is not a tested library.

## Install

`pip install fastpapers`

## How to use

The name of each module is the [bibtexkey](https://en.wikipedia.org/wiki/BibTeX#Field_types) of the corresponing paper.
For example, if you want to use the FID metric from [Heusel, Martin, et al. 2017](http://papers.nips.cc/paper/7240-gans-trained-by-a-two-t), you can import it like so:

```
from fastpapers.heusel2017gans import FIDMetric
```

If you want to train a pix2pix model from Isola, [Phillip, et al](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) you can import a pix2pix_learner

```
from fastpapers.isola2017image import pix2pix_learner
```

The `core` module contains functions and classes that are useful for several papers.
For example, you have a `ImageNTuple` to work with an arbitrary amount of images as input.

```
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
```

```
it = ImageNTuple.create((files[0], files[1], files[2]))
it = Resize(224)(it)
it = ToTensor()(it)
it.show();
```


![png](docs/images/output_10_0.png)


Or useful functions for debuging like `explode_shapes` or `explode_ranges`

```
explode_shapes(it)
```




    [(3, 224, 224), (3, 224, 224), (3, 224, 224)]


