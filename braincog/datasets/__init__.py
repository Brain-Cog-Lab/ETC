__all__ = ['datasets', 'cut_mix', 'rand_aug', 'NOmniglot', 'is_dvs_data']

from . import (
    datasets,
    cut_mix,
    rand_aug,
    event_drop,
    NOmniglot,
    hmdb_dvs,
    ucf101_dvs,
    ncaltech101
)

dvs_data = [
    'dvsg',
    'dvsc10',
    'NCALTECH101',
    'NCARS',
    'DVSG',
    'UCF101DVS',
    'HMDBDVS',
]


def is_dvs_data(dataset):
    if dataset in dvs_data:
        return True
    else:
        return False


