class NpuOptimizerManager:
    NpuOptimizerDict = {}

    def __init__(self, npu_platform):
        self.npu_platform = npu_platform

    def __call__(self, npu_optimizer):
        NpuOptimizerManager.NpuOptimizerDict[self.npu_platform] = npu_optimizer

    @classmethod
    def get_npu_optimizer(cls, npu_platform):
        return cls.NpuOptimizerDict[npu_platform]