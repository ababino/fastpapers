# Welcome to fastpapers
> Play lego with papers


`fastpapers` is a python library to reproduce papers in [Jupyter Notebooks](https://jupyter.org/). Thanks to [nbdev](https://nbdev.fast.ai/), we turn these notebooks into modules. With this library you can import your favorite feature from your favorite paper. The implementations are done using [fastai](https://docs.fast.ai/).

## Install

`pip install your_project_name`

## How to use

The name of each module is the [Bibtexkey](https://en.wikipedia.org/wiki/BibTeX#Field_types) of the corresponing paper.
For example, if you want to use the FID metric from [Heusel, Martin, et al. 2017](http://papers.nips.cc/paper/7240-gans-trained-by-a-two-t), you can import it like so:

```python
from fastpapers.heusel2017gans import FIDMetric
```

The `core` module contains functions and classes that are useful for several papers.
For example, you have a `ImageNTuple` to work with an arbitrary amount of images as input.

```python
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
```

```python
it = ImageNTuple.create((files[0], files[1], files[2]))
it = Resize(224)(it)
it = ToTensor()(it)
it.show();
```


![png](docs/images/output_8_0.png)


Or useful debugging functions like `explode_shapes` or `explode_ranges`

```python
explode_shapes(it)
```




    [(3, 224, 224), (3, 224, 224), (3, 224, 224)]


