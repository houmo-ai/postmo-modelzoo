from .xh1_infer import Xh1Infer


class Xh2Infer(Xh1Infer):
    def __init__(self):
        super().__init__()
        self.backend = "xh2"
