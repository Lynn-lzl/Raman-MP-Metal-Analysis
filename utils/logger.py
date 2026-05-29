import time
from dataclasses import dataclass

@dataclass
class Timer:
    t0: float = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0

class Logger:
    @staticmethod
    def log(msg: str):
        print(msg, flush=True)

    @staticmethod
    def log_step(title: str):
        bar = "-" * 100
        Logger.log(f"\n{bar}\n{title}\n{bar}")