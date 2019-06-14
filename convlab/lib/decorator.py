# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import time
from functools import wraps

from convlab.lib import logger

logger = logger.get_logger(__name__)


def lab_api(fn):
    '''
    Function decorator to label and check Lab API methods
    @example

    from convlab.lib.decorator import lab_api
    @lab_api
    def foo():
        print('foo')
    '''
    return fn


def timeit(fn):
    '''
    Function decorator to measure execution time
    @example

    from convlab.lib.decorator import timeit
    @timeit
    def foo(sec):
        time.sleep(sec)
        print('foo')

    foo(1)
    # => foo
    # => Timed: foo 1000.9971ms
    '''
    @wraps(fn)
    def time_fn(*args, **kwargs):
        start = time.time()
        output = fn(*args, **kwargs)
        end = time.time()
        logger.debug(f'Timed: {fn.__name__} {round((end - start) * 1000, 4)}ms')
        return output
    return time_fn
