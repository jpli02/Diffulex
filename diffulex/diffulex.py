from diffulex.config import Config
from diffulex.engine.dp_worker import DiffulexDPWorker
from diffulex.engine.tp_worker import DiffulexTPWorker

class Diffulex:
    def __new__(cls, model, **kwargs):
        cfg = Config(model, **{k: v for k, v in kwargs.items() if k in Config.__dataclass_fields__.keys()})
        if cfg.data_parallel_size > 1:
            return DiffulexDPWorker(model, **kwargs)
        return DiffulexTPWorker(model, **kwargs)