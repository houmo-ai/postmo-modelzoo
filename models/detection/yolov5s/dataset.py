from hmatc.datasets.coco import COCO2017Val


class Dataset(COCO2017Val):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
