import time

class TAITime(object):
    def __init__(self, t=None):
        if t is None:
            self.t = time.time()
        elif isinstance(t, TAITime):
            self.t = t.t
        else:
            self.t = float(t)

    def __sub__(self, other):
        if isinstance(other, TAITime):
            return self.t - other.t
        return self.t - float(other)

    def __rsub__(self, other):
        if isinstance(other, TAITime):
            return other.t - self.t
        return float(other) - self.t

    def __add__(self, other):
        if isinstance(other, TAITime):
            return self.t + other.t
        return self.t + float(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __float__(self):
        return float(self.t)

    def __str__(self):
        return str(self.t)

    def copy(self):
        return TAITime(self.t)
