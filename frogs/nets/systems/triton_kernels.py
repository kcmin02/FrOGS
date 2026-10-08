"""Fused FP32 Triton kernels for the AR prior (systems backend).

`delta_log_prob_all_sites` packs, for every candidate flip, only the reachable nodes of its causal
dependency cone and propagates activation differences through them; `sample_triton` evaluates one
site's whole causal network per kernel launch. Matrix products use tf32x3, so results agree with the
reference to FP32 round-off rather than bitwise. Requires an Ampere-or-newer GPU.
"""
from functools import lru_cache

import torch
import triton
import triton.language as tl


def causal_taps(kernel_size, mask_type):
    """Kernel offsets kept by the causal mask, in raster order."""
    centre = tuple(k // 2 for k in kernel_size)
    taps = []
    for a in range(kernel_size[0]):
        for b in range(kernel_size[1]):
            for d in range(kernel_size[2]):
                off = (a - centre[0], b - centre[1], d - centre[2])
                if off < (0, 0, 0) or (mask_type == "B" and off == (0, 0, 0)):
                    taps.append(off)
    return tuple(taps)


def film_parameters(film, c):
    gamma_raw, beta = film.net(c).chunk(2, dim=-1)
    return 1.0 + gamma_raw, beta


@lru_cache(maxsize=8)
def _topology(shape, kernel, layers):
    """Per layer: reachable in-bounds nodes of every flip's cone, with links into the layer below."""
    lx, ly, lz = shape
    xyz = [(i, j, k) for i in range(lx) for j in range(ly) for k in range(lz)]
    flat = {p: n for n, p in enumerate(xyz)}
    previous = [{p: 0} for p in xyz]
    result = []
    for layer in range(layers):
        taps = causal_taps(kernel, "A" if layer == 0 else "B")
        sites, flips, links, first_taps, offsets = [], [], [], [], [0]
        current = []
        for flip, prev in enumerate(previous):
            nodes = sorted({
                (p[0]-t[0], p[1]-t[1], p[2]-t[2])
                for p in prev for t in taps
                if (p[0]-t[0], p[1]-t[1], p[2]-t[2]) in flat
            })
            mapping = {}
            for q in nodes:
                mapping[q] = len(sites)
                sites.append(flat[q])
                flips.append(flip)
                if layer == 0:
                    p = xyz[flip]
                    first_taps.append(taps.index(tuple(p[d]-q[d] for d in range(3))))
                else:
                    links.append([prev.get(tuple(q[d]+t[d] for d in range(3)), -1)
                                  for t in taps])
            current.append(mapping)
            offsets.append(len(sites))
        result.append((torch.tensor(sites, dtype=torch.int32),
                       torch.tensor(flips, dtype=torch.int32),
                       torch.tensor(first_taps if layer == 0 else links,
                                    dtype=torch.int32),
                       torch.tensor(offsets, dtype=torch.int32),
                       max(b-a for a, b in zip(offsets, offsets[1:]))))
        previous = current
    return tuple(result)


@lru_cache(maxsize=8)
def _device_topology(shape, kernel, layers, device):
    return tuple(tuple(x.to(device) if torch.is_tensor(x) else x for x in row)
                 for row in _topology(shape, kernel, layers))


@triton.jit
def _first(S, W, Z, H, G, Beta, Sites, Flips, Taps, Out,
           N: tl.constexpr, Q: tl.constexpr, C: tl.constexpr,
           BC: tl.constexpr, R: tl.constexpr):
    q = tl.program_id(0) * R + tl.arange(0, R)
    b = tl.program_id(1)
    ch = tl.arange(0, BC)
    site = tl.load(Sites+q, q<Q, other=0)
    flip = tl.load(Flips+q, q<Q, other=0)
    tap = tl.load(Taps+q, q<Q, other=0)
    ds = -2.0 * tl.load(S+b*N+flip, q<Q, other=0)
    w = tl.load(W+tap[:, None]*C+ch[None, :], (q[:, None]<Q)&(ch[None, :]<C), other=0)
    ix = (b*N+site[:, None])*C+ch[None, :]
    mask = (q[:, None]<Q)&(ch[None, :]<C)
    z = tl.load(Z+ix, mask, other=0) + ds[:, None]*w
    g = tl.load(G+b*C+ch, ch<C, other=0)
    beta = tl.load(Beta+b*C+ch, ch<C, other=0)
    x = g[None, :]*z+beta[None, :]
    dh = x*tl.sigmoid(x)-tl.load(H+ix, mask, other=0)
    tl.store(Out+(b*Q+q[:, None])*C+ch[None, :], dh, mask)


@triton.jit
def _deeper(D, W, Z, H, G, Beta, Sites, Links, Out,
            N: tl.constexpr, Q: tl.constexpr, P: tl.constexpr,
            C: tl.constexpr, BC: tl.constexpr, T: tl.constexpr,
            R: tl.constexpr):
    q = tl.program_id(0)*R+tl.arange(0, R)
    b = tl.program_id(1)
    ch = tl.arange(0, BC)
    acc = tl.full((R, BC), 0, tl.float32)
    for tap in range(T):
        prev = tl.load(Links+q*T+tap, q<Q, other=-1)
        a = tl.load(D+(b*P+prev[:, None])*C+ch[None, :],
                    (q[:, None]<Q)&(prev[:, None]>=0)&(ch[None, :]<C), other=0)
        w = tl.load(W+tap*C*C+ch[:, None]*C+ch[None, :],
                    (ch[:, None]<C)&(ch[None, :]<C), other=0)
        acc = tl.dot(a, w, acc, input_precision="tf32x3")
    site = tl.load(Sites+q, q<Q, other=0)
    ix = (b*N+site[:, None])*C+ch[None, :]
    mask = (q[:, None]<Q)&(ch[None, :]<C)
    z = tl.load(Z+ix, mask, other=0)+acc
    g = tl.load(G+b*C+ch, ch<C, other=0)
    beta = tl.load(Beta+b*C+ch, ch<C, other=0)
    x = g[None, :]*z+beta[None, :]
    dh = x*tl.sigmoid(x)-tl.load(H+ix, mask, other=0)
    tl.store(Out+(b*Q+q[:, None])*C+ch[None, :], dh, mask)


@triton.jit
def _softplus(x):
    return tl.maximum(x, 0.)+tl.log(1.+tl.exp(-tl.abs(x)))


@triton.jit
def _head(D, W, S, Logits, Sites, Out, N: tl.constexpr, Q: tl.constexpr,
          C: tl.constexpr, BC: tl.constexpr, R: tl.constexpr):
    q = tl.program_id(0)*R+tl.arange(0, R)
    b = tl.program_id(1)
    ch = tl.arange(0, BC)
    dh = tl.load(D+(b*Q+q[:, None])*C+ch[None, :],
                 (q[:, None]<Q)&(ch[None, :]<C), other=0)
    w = tl.load(W+ch, ch<C, other=0)
    dl = tl.sum(dh*w[None, :], 1)
    site = tl.load(Sites+q, q<Q, other=0)
    s = tl.load(S+b*N+site, q<Q, other=0)
    logit = tl.load(Logits+b*N+site, q<Q, other=0)
    value = _softplus(-s*logit)-_softplus(-s*(logit+dl))
    tl.store(Out+b*Q+q, value, q<Q)


@triton.jit
def _reduce(Changes, Offsets, S, Logits, Out, N: tl.constexpr,
            Q: tl.constexpr, BLOCK: tl.constexpr):
    flip, b = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(Offsets+flip), tl.load(Offsets+flip+1)
    ix = start+tl.arange(0, BLOCK)
    changes = tl.load(Changes+b*Q+ix, ix<end, other=0)
    s, logit = tl.load(S+b*N+flip), tl.load(Logits+b*N+flip)
    own = _softplus(-s*logit)-_softplus(s*logit)
    tl.store(Out+b*N+flip, own+tl.sum(changes, 0))


@triton.jit
def _cache_layer(Input, W, Bias, Gamma, Beta, Neighbors, Pre, Hidden,
                 N: tl.constexpr, CI: tl.constexpr, C: tl.constexpr,
                 BC: tl.constexpr, T: tl.constexpr, R: tl.constexpr):
    site = tl.program_id(0)*R+tl.arange(0, R)
    b = tl.program_id(1)
    ch = tl.arange(0, BC)
    acc = tl.full((R, BC), 0., tl.float32)
    for tap in range(T):
        p = tl.load(Neighbors+site*T+tap, site<N, other=-1)
        a = tl.load(Input+(b*N+p[:, None])*CI+ch[None, :],
                    (site[:, None]<N)&(p[:, None]>=0)&(ch[None, :]<CI), other=0)
        w = tl.load(W+tap*CI*C+ch[:, None]*C+ch[None, :],
                    (ch[:, None]<CI)&(ch[None, :]<C), other=0)
        acc = tl.dot(a, w, acc, input_precision="tf32x3")
    bias = tl.load(Bias+ch, ch<C, other=0)
    pre = acc+bias[None, :]
    g = tl.load(Gamma+b*C+ch, ch<C, other=0)
    beta = tl.load(Beta+b*C+ch, ch<C, other=0)
    x = g[None, :]*pre+beta[None, :]
    ix = (b*N+site[:, None])*C+ch[None, :]
    mask = (site[:, None]<N)&(ch[None, :]<C)
    tl.store(Pre+ix, pre, mask)
    tl.store(Hidden+ix, x*tl.sigmoid(x), mask)


@triton.jit
def _cache_head(H, W, Bias, Out, N: tl.constexpr, C: tl.constexpr,
                BC: tl.constexpr, R: tl.constexpr):
    site = tl.program_id(0)*R+tl.arange(0, R)
    b = tl.program_id(1)
    ch = tl.arange(0, BC)
    h = tl.load(H+(b*N+site[:, None])*C+ch[None, :],
                (site[:, None]<N)&(ch[None, :]<C), other=0)
    w = tl.load(W+ch, ch<C, other=0)
    logit = tl.sum(h*w[None, :], 1)+tl.load(Bias)
    tl.store(Out+b*N+site, logit, site<N)


def _cache(model, sigma, c):
    """Base activations (channels-last) computed with the same arithmetic as the delta passes."""
    batch, n, channels = sigma.shape[0], model.N, model.num_channels
    pre, hidden, affine, weights = [], [], [], []
    h = sigma.contiguous()
    for layer, (conv, film) in enumerate(zip(
            (model.input_conv, *model.convs), (model.film_initial, *model.films))):
        kind = "A" if layer == 0 else "B"
        taps = len(causal_taps(conv.kernel_size, kind))
        w = (conv.weight*conv.mask).flatten(2)[:, :, :taps].permute(2, 1, 0).contiguous()
        gamma, beta = (a.contiguous() for a in film_parameters(film, c))
        neighbors = _neighbors_for(model, sigma.device, conv.kernel_size, kind)
        z = torch.empty((batch, n, channels), device=sigma.device, dtype=sigma.dtype)
        out = torch.empty_like(z)
        _cache_layer[(triton.cdiv(n, 32), batch)](
            h, w, conv.bias, gamma, beta, neighbors, z, out,
            n, conv.in_channels, channels, max(32, triton.next_power_of_2(channels)),
            taps, 32, enable_fp_fusion=False)
        pre.append(z)
        hidden.append(out)
        affine.append((gamma, beta))
        weights.append(w)
        h = out
    logits = torch.empty((batch, *model.lattice_shape), device=sigma.device, dtype=sigma.dtype)
    _cache_head[(triton.cdiv(n, 32), batch)](
        h, model.head.weight.contiguous(), model.head.bias, logits,
        n, channels, triton.next_power_of_2(channels), 32, enable_fp_fusion=False)
    return pre, hidden, affine, logits, weights


def supported(model, sigma):
    """True when the fused kernels apply; otherwise callers use the reference scorer."""
    return (not torch.is_grad_enabled() and sigma.is_cuda
            and sigma.dtype == torch.float32
            and model.head.weight.dtype == torch.float32
            and not torch.is_autocast_enabled()
            and torch.cuda.get_device_capability(sigma.device)[0] >= 8
            and model.num_channels <= 64
            and all(c.kernel_size == model.input_conv.kernel_size
                    and c.stride == (1, 1, 1) and c.dilation == (1, 1, 1)
                    and c.groups == 1
                    and c.padding == tuple(k//2 for k in c.kernel_size)
                    for c in (model.input_conv, *model.convs))
            and all(k % 2 == 1 for k in model.input_conv.kernel_size))


def delta_log_prob_all_sites(model, sigma, c):
    """log rho_0(flip_s sigma) - log rho_0(sigma) for all sites s, shape (B, N)."""
    shape = tuple(model.lattice_shape)
    if tuple(sigma.shape[1:]) != shape:
        raise ValueError("sigma shape does not match AR lattice_shape")
    convs = (model.input_conv, *model.convs)
    key = ("cone", shape, model.input_conv.kernel_size, len(convs), sigma.device)
    metadata = model.__dict__.setdefault("_triton_metadata", {})
    if key not in metadata:
        metadata[key] = _device_topology(shape, model.input_conv.kernel_size,
                                          len(convs), sigma.device)
    meta = metadata[key]
    pre, hidden, affine, logits, packed_weights = _cache(model, sigma, c)
    batch, channels, n = sigma.shape[0], model.num_channels, model.N
    sigma, logits = sigma.contiguous(), logits.contiguous()
    delta = None
    for layer, (conv, row) in enumerate(zip(convs, meta)):
        sites, flips, links, offsets, max_nodes = row
        q = sites.numel()
        if q == 0:
            return -sigma.flatten(1)*logits.flatten(1)
        taps = causal_taps(conv.kernel_size, "A" if layer == 0 else "B")
        weights = packed_weights[layer]
        z, h = pre[layer], hidden[layer]
        gamma, beta = (v.contiguous() for v in affine[layer])
        out = torch.empty((batch, q, channels), device=sigma.device, dtype=sigma.dtype)
        grid = (triton.cdiv(q, 32), batch)
        if layer == 0:
            _first[grid](sigma, weights, z, h, gamma, beta, sites, flips, links,
                         out, n, q, channels, triton.next_power_of_2(channels), 32,
                         enable_fp_fusion=False)
        else:
            _deeper[grid](delta, weights, z, h, gamma, beta, sites, links,
                          out, n, q, delta.shape[1], channels,
                          max(32, triton.next_power_of_2(channels)), len(taps), 32,
                          enable_fp_fusion=False)
        delta = out
    changes = torch.empty((batch, q), device=sigma.device, dtype=sigma.dtype)
    _head[grid](delta, model.head.weight.contiguous(), sigma, logits, sites,
                changes, n, q, channels, triton.next_power_of_2(channels), 32,
                enable_fp_fusion=False)
    result = torch.empty((batch, n), device=sigma.device, dtype=sigma.dtype)
    _reduce[(n, batch)](changes, offsets, sigma, logits, result, n, q,
                        triton.next_power_of_2(max_nodes), enable_fp_fusion=False)
    return result


@triton.jit
def _sample_site(S, Cache, Neighbors, W0, W, Bias, Gamma, Beta, Head, HeadBias,
                 Logits, site, N: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                 LAYERS: tl.constexpr, BC: tl.constexpr, R: tl.constexpr):
    b = tl.program_id(0)*R+tl.arange(0, R)
    ch = tl.arange(0, BC)
    taps = tl.arange(0, 16)
    prev = tl.load(Neighbors+site*14+taps, taps<13, other=-1)
    a0 = tl.load(S+b[:, None]*N+prev[None, :],
                (b[:, None]<B)&(prev[None, :]>=0), other=0)
    w0 = tl.load(W0+taps[:, None]*C+ch[None, :],
                (taps[:, None]<13)&(ch[None, :]<C), other=0)
    h = tl.dot(a0, w0, input_precision="tf32x3")
    mask = (b[:, None]<B)&(ch[None, :]<C)
    bc = b[:, None]*C+ch[None, :]
    bias = tl.load(Bias+ch, ch<C, other=0)
    g = tl.load(Gamma+bc, mask, other=0)
    beta = tl.load(Beta+bc, mask, other=0)
    x = g*(h+bias[None, :])+beta
    h = x*tl.sigmoid(x)
    tl.store(Cache+(b[:, None]*N+site)*C+ch[None, :], h, mask)
    for layer in range(1, LAYERS):
        acc = tl.full((R, BC), 0., tl.float32)
        for tap in range(14):
            p = tl.load(Neighbors+site*14+tap)
            old = tl.load(Cache+((layer-1)*B*N+b[:, None]*N+p)*C+ch[None, :],
                          mask&(p>=0)&(p!=site), other=0)
            a = tl.where(p==site, h, old)
            w = tl.load(W+((layer-1)*14+tap)*C*C+ch[:, None]*C+ch[None, :],
                        (ch[:, None]<C)&(ch[None, :]<C), other=0)
            acc = tl.dot(a, w, acc, input_precision="tf32x3")
        bias = tl.load(Bias+layer*C+ch, ch<C, other=0)
        g = tl.load(Gamma+layer*B*C+bc, mask, other=0)
        beta = tl.load(Beta+layer*B*C+bc, mask, other=0)
        x = g*(acc+bias[None, :])+beta
        h = x*tl.sigmoid(x)
        tl.store(Cache+(layer*B*N+b[:, None]*N+site)*C+ch[None, :], h, mask)
    head = tl.load(Head+ch, ch<C, other=0)
    logit = tl.sum(h*head[None, :], 1)+tl.load(HeadBias)
    tl.store(Logits+b, logit, b<B)


@lru_cache(maxsize=8)
def _sampling_neighbors(shape, device, kernel=(3, 3, 3), mask_type="B"):
    lx, ly, lz = shape
    taps = causal_taps(kernel, mask_type)
    rows = []
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                rows.append([((i+a)*ly+j+b)*lz+k+d
                             if 0<=i+a<lx and 0<=j+b<ly and 0<=k+d<lz else -1
                             for a, b, d in taps])
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _neighbors_for(model, device, kernel=(3, 3, 3), mask_type="B"):
    # Owned by the model: captured CUDA graphs hold these addresses.
    key = ("neighbors", tuple(model.lattice_shape), device, kernel, mask_type)
    metadata = model.__dict__.setdefault("_triton_metadata", {})
    if key not in metadata:
        metadata[key] = _sampling_neighbors(tuple(model.lattice_shape), device,
                                              kernel, mask_type)
    return metadata[key]


@torch.no_grad()
def sample_triton(model, c, n_samples):
    """One fused kernel per site; the Bernoulli draws stay in PyTorch, in raster order."""
    device, dtype = model.head.weight.device, model.head.weight.dtype
    sigma = torch.zeros((n_samples, *model.lattice_shape), device=device, dtype=dtype)
    if not supported(model, sigma) or model.input_conv.kernel_size != (3, 3, 3):
        from frogs.nets.systems.sampling import sample_cached
        return sample_cached(model, c, n_samples)
    if c is not None:
        c = torch.as_tensor(c, device=device, dtype=dtype)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        c = c.expand(n_samples, -1).contiguous() if c.shape[0] == 1 else c
        if c.shape[0] != n_samples:
            raise ValueError("c batch must be 1 or n_samples")
    convs = (model.input_conv, *model.convs)
    films = (model.film_initial, *model.films)
    channels = model.num_channels
    affines = [film_parameters(f, c) for f in films]
    gamma = torch.stack([a[0] for a in affines]).contiguous()
    beta = torch.stack([a[1] for a in affines]).contiguous()
    weights = [(conv.weight*conv.mask).flatten(2)[:, :, :14].permute(2, 1, 0).contiguous()
               for conv in convs]
    deeper = torch.stack(weights[1:]) if len(convs)>1 else weights[0]
    bias = torch.stack([conv.bias for conv in convs])
    hidden = torch.empty((len(convs), n_samples, model.N, channels),
                         device=device, dtype=dtype)
    logits = torch.empty((n_samples,), device=device, dtype=dtype)
    neighbors = _neighbors_for(model, device)
    flat = sigma.flatten(1)
    for site in range(model.N):
        _sample_site[(triton.cdiv(n_samples, 16),)](
            sigma, hidden, neighbors, weights[0], deeper, bias, gamma, beta,
            model.head.weight.contiguous(), model.head.bias, logits, site,
            model.N, n_samples, channels, len(convs),
            max(32, triton.next_power_of_2(channels)), 16, enable_fp_fusion=False)
        flat[:, site] = 2.*torch.bernoulli(torch.sigmoid(logits))-1.
    return sigma
