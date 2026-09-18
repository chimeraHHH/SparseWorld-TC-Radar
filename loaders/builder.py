from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from mmdet.datasets.builder import worker_init_fn
from mmdet.datasets.samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from mmcv.parallel import DataContainer
from torch.utils.data._utils.pin_memory import pin_memory as pin_tensor_tree


class PinnableDataContainer(DataContainer):
    def pin_memory(self):
        if self.cpu_only:
            return self
        return PinnableDataContainer(pin_tensor_tree(self.data), self.stack,
                                     self.padding_value, self.cpu_only, self.pad_dims)


def pinnable_collate(batch, samples_per_gpu):
    def convert(value):
        if isinstance(value, DataContainer):
            return PinnableDataContainer(value.data, value.stack, value.padding_value,
                                         value.cpu_only, value.pad_dims)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(convert(v) for v in value)
        return value
    return convert(collate(batch, samples_per_gpu=samples_per_gpu))


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     pin_memory=False,
                     **kwargs):

    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(
                dataset, samples_per_gpu, world_size, rank, seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False, seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(pinnable_collate if pin_memory else collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader
