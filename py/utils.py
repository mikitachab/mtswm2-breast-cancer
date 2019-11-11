import itertools
import functools


def map_key_to_every_value(key, values):
    return [{key: value} for value in values]


def merge_dicts(dicts):
    return functools.reduce(lambda a, b: {**a, **b}, dicts)


def product_from_dict(grid):
    """
    return list of dict with combinations of items from dict iterators values
    e.g.
        grid = {
            'even': [2,4],
            'odd': [1,3]
        }
        returns:
        [
            {'even': 2, 'odd': 1},
            {'even': 2, 'odd': 3},
            {'even': 4, 'odd': 1},
            {'even': 4, 'odd': 3}
        ]
    """
    buff = [map_key_to_every_value(key, value) for key, value in grid.items()]
    return [merge_dicts(args) for args in itertools.product(*buff)]
