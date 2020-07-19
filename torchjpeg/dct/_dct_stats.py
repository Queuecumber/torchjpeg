import torch
from torch import Tensor
from ._stats import Stats
from pathlib import Path


class DCTStats(Stats):
    def __init__(self, root: Path, type: str = 'ms') -> None:
        super(DCTStats, self).__init__(root, type)
        stats = torch.load(root.open('rb'))

        if 'y' in stats:
            self.mean = {x: stats[x]['mean'].view(1, 1, 8, 8) for x in stats.keys()}
            self.variance = {x: stats[x]['variance'].view(1, 1, 8, 8) for x in stats.keys()}
            self.std = {x: torch.sqrt(self.variance[x]) for x in stats.keys()}

            self.min = {x: stats[x]['min'].view(1, 1, 8, 8) for x in stats.keys()}
            self.max = {x: stats[x]['max'].view(1, 1, 8, 8) for x in stats.keys()}
        else:
            self.mean = {'y': stats['mean'].view(1, 1, 8, 8)}
            self.variance = {'y': stats['variance'].view(1, 1, 8, 8)}
            self.std = {'y': torch.sqrt(self.variance['y'])}

            self.min = {'y': stats['min'].view(1, 1, 8, 8)}
            self.max = {'y': stats['max'].view(1, 1, 8, 8)}

    def _mean_variance_f(self, blocks: Tensor, type: str = 'y') -> Tensor:
        m = self.mean[type]
        s = self.std[type]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()            

        return (blocks - m) / s

    def _zero_one_f(self, blocks: Tensor, type: str = 'y') -> Tensor:
        m = -self.min[type]
        s = self.max[type] - self.min[type]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()            

        return (blocks + m) / s

    def _mean_variance_r(self, blocks: Tensor, type: str = 'y') -> Tensor:
        s = self.std[type]
        m = self.mean[type]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()            

        return blocks * s + m

    def _zero_one_r(self, blocks: Tensor, type: str = 'y') -> Tensor:
        s = self.max[type] - self.min[type]
        m = -self.min[type]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()

        return blocks * s - m
