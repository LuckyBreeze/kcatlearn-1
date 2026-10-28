"""Approved CataPro-only training batch sampler: merge terminal 8+1."""
from torch.utils.data import BatchSampler


class MergeSingletonBatchSampler(BatchSampler):
    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is None:
                pending = batch
            elif len(batch) == 1:
                yield pending + batch
                pending = None
            else:
                yield pending
                pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        count = super().__len__()
        return count - int(len(self.sampler) % self.batch_size == 1 and count > 1)
