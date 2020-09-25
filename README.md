# FastPapers
> Implementation of papers algorithms compliant with fastai library.


This file will become your README and also the index of your documentation.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

```python
1+1  
```




    2



```python
from fastai.data.external import untar_data,URLs
from fastai.data.transforms import get_image_files
from fastai.vision.augment import Resize, ToTensor
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
```

```python
it = ImageNTuple.create((files[0], files[1], files[2]))
it = Resize(224)(it)
it = ToTensor()(it)
it.show();
```


![png](docs/images/output_7_0.png)

