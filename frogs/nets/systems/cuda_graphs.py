"""Bounded CUDA-graph replay for no-grad prior operations.

Capture preserves the RNG state, replay copies new inputs, reads the live parameter storage and
clones the output before the next replay can overwrite it.
"""
from collections import OrderedDict

import torch

MAX_GRAPHS = 4


def replay(model, name, operation, *args):
    device = model.head.weight.device
    if (device.type != "cuda" or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or torch.cuda.is_current_stream_capturing()):
        return operation(*args)
    cache = model.__dict__.setdefault("_graph_cache", OrderedDict())
    storage = tuple((v.data_ptr(), v.shape, v.stride(), v.dtype, v.device)
                    for v in (*model.parameters(), *model.buffers()))
    signature = tuple((a.shape, a.dtype, a.device) if torch.is_tensor(a) else (type(a), a)
                      for a in args)
    key = (name, signature, storage, torch.cuda.current_stream(device).cuda_stream)
    entry = cache.get(key)
    if entry is None:
        static = tuple(a.detach().clone().contiguous() if torch.is_tensor(a) else a for a in args)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.random.fork_rng(devices=[device]):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    operation(*static)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = operation(*static)
        entry = (graph, static, output)
        cache[key] = entry
        while len(cache) > MAX_GRAPHS:
            cache.popitem(last=False)
    cache.move_to_end(key)
    graph, static, output = entry
    for dst, src in zip(static, args):
        if torch.is_tensor(dst):
            dst.copy_(src)
    graph.replay()
    return output.clone()
