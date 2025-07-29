import abc

class BaseInfer(object, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.time_span = 0
        self.total = 0
        self.engine = None
        self.backend = "onnx"  # onnx/hmquant/xh1/xh2
        self.device = "cpu"

    @abc.abstractmethod
    def load(self, model_path):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, in_datas: dict, to_file=False):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    @property
    def ave_latency_ms(self):
        if self.total == 0:
            return 0
        return self.time_span / self.total