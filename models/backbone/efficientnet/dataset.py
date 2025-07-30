from hmtool.datasets.imagenet import ILSVRC2012


class Dataset(ILSVRC2012):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)