# coding:utf-8
# Author:mylady
# Datetime:2023/4/23 4:09
import time
import numpy as np


class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

    def final_time(self) -> str:
        """返回最后记录的时间, 总计耗时"""
        return "%8.4f" % self.times[-1]

    def interval_consume(self) -> str:
        """区间时间段计算"""
        diffs = ['%.4f' % (y - x) for x, y in zip(self.times, self.times[1:])]
        return str(diffs)

    def __call__(self) -> str:
        return str(['%.4f' % item for item in self.times])

    def __str__(self) -> str:
        return str(['%.4f' % item for item in self.times])


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in `sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    pass


if __name__ == '__main__':
    main()
