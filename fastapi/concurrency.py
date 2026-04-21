import functools
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractContextManager
from contextlib import asynccontextmanager as asynccontextmanager
from typing import ParamSpec, TypeVar

import anyio.to_thread
from anyio import CapacityLimiter
from starlette.concurrency import iterate_in_threadpool as iterate_in_threadpool  # noqa
from starlette.concurrency import run_in_threadpool as run_in_threadpool  # noqa
from starlette.concurrency import (  # noqa
    run_until_first_complete as run_until_first_complete,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Blocking __exit__ and other teardown operations from running can create race
# conditions/deadlocks if the context manager itself has its own internal pool
# (e.g. a database connection pool).
# To avoid this maintain a separate limiter for teardown operations, and attempt
# to use it if the main threadpool is full.
_deadlock_overflow_limiter = CapacityLimiter(1)


@asynccontextmanager
async def contextmanager_in_threadpool(
    cm: AbstractContextManager[_T],
) -> AsyncGenerator[_T, None]:
    try:
        yield await run_in_threadpool(cm.__enter__)
    except Exception as e:
        ok = bool(
            await run_in_threadpool_with_overflow(
                cm.__exit__, type(e), e, e.__traceback__
            )
        )
        if not ok:
            raise e
    else:
        await run_in_threadpool_with_overflow(cm.__exit__, None, None, None)


async def run_in_threadpool_with_overflow(
    func: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """Run a function in the threadpool, or if is full use the overflow pool.

    This will run the function a separate threadpool if the main one is full in order
    to avoid being blocked by other operations waiting to acquire resources.

    Unless you know what you are doing, you probably don't want this function,
    use run_in_threadpool instead.
    """
    func = functools.partial(func, *args, **kwargs)

    if anyio.to_thread.current_default_thread_limiter().available_tokens > 0:
        return await run_in_threadpool(func)

    return await anyio.to_thread.run_sync(func, limiter=_deadlock_overflow_limiter)
