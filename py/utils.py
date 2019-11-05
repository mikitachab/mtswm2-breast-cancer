import itertools
import functools


def map_key_to_every_value(key, values):
    return [{key: value} for value in values]


def merge_dicts(dicts):
    return functools.reduce(lambda a, b: {**a, **b}, dicts)


def product_from_dict(param_grid):
    kek = [map_key_to_every_value(key, value) for key, value in param_grid.items()]
    return [merge_dicts(args) for args in itertools.product(*kek)]
