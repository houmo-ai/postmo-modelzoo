from hmatc.datasets.widerface import WiderFace


class Dataset(WiderFace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
