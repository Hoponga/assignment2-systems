import time


def benchmark(fn, args=(), kwargs=None, warmup_iters=5, iters=100):
    if kwargs is None:
        kwargs = {}

    for _ in range(warmup_iters):
        fn(*args, **kwargs)

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000
    print(f"{fn.__name__}: {avg_ms:.4f} ms avg over {iters} iters ({warmup_iters} warmup)")
    return avg_ms


def sweep(fn, args_list, kwargs=None, warmup_iters=5, iters=100):
    results = []
    for args in args_list:
        avg_ms = benchmark(fn, args=args, kwargs=kwargs, warmup_iters=warmup_iters, iters=iters)
        results.append((args, avg_ms))
    return results
