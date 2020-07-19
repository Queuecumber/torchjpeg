import torch


class Stats(object):
    def __init__(self, root, type='ms'):
        self.stats = torch.load(root)
        self.type = type

    def forward(self, *args, **kwargs):
        if self.type == 'ms':
            return self._mean_variance_f(*args, **kwargs)
        elif self.type == '01':
            return self._zero_one_f(*args, **kwargs)
        elif self.type == '-11':
            return self._one_one_f(*args, **kwargs)
        else:
            raise NotImplementedError()

    def backward(self, *args, **kwargs):
        if self.type == 'ms':
            return self._mean_variance_r(*args, **kwargs)
        elif self.type == '01':
            return self._zero_one_r(*args, **kwargs)
        elif self.type == '-11':
            return self._one_one_r(*args, **kwargs)
        else:
            raise NotImplementedError()

    def _mean_variance_f(self, *args, **kwargs):
        raise NotImplementedError()

    def _zero_one_f(self, *args, **kwargs):
        raise NotImplementedError()

    def _one_one_f(self, *args, **kwargs):
        return self._zero_one_f(*args, **kwargs) * 2. - 1.

    def _mean_variance_r(self, *args, **kwargs):
        raise NotImplementedError()

    def _zero_one_r(self, *args, **kwargs):
        raise NotImplementedError()

    def _one_one_r(self, input, *args, **kwargs):
        return self._zero_one_r((input + 1.) / 2., *args, **kwargs)
