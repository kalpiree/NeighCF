from __future__ import annotations

def backward_cached(microbatches, forward, objective, *, device, check_replay=True):
    import torch
    device = torch.device(device)
    snapshots, cached = ([], [])

    def capture():
        return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)

    def restore(state):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
    for batch in microbatches:
        snapshots.append(capture())
        with torch.no_grad():
            outputs = forward(batch)
            if not isinstance(outputs, tuple) or not outputs:
                raise ValueError('cached forward must return a nonempty tensor tuple')
            if any((not torch.isfinite(output).all() for output in outputs)):
                raise RuntimeError('nonfinite representation in gradient-cache first pass')
            cached.append(tuple((output.detach() for output in outputs)))
    after_first_pass = capture()
    full = tuple((torch.cat([outputs[index] for outputs in cached]).float().detach().requires_grad_(True) for index in range(len(cached[0]))))
    loss, diagnostics = objective(full)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite full-batch objective')
    gradients = torch.autograd.grad(loss, full)
    if any((not torch.isfinite(gradient).all() for gradient in gradients)):
        raise RuntimeError('nonfinite full-batch representation gradients')
    offsets = [0] * len(full)
    try:
        for batch, state, expected in zip(microbatches, snapshots, cached):
            restore(state)
            outputs = forward(batch)
            per_micro = []
            for index, (output, reference) in enumerate(zip(outputs, expected)):
                if check_replay and (not torch.allclose(output.detach(), reference, rtol=0.001, atol=0.0001)):
                    raise RuntimeError('gradient-cache replay changed representations beyond tolerance')
                stop = offsets[index] + output.shape[0]
                per_micro.append(gradients[index][offsets[index]:stop].to(output.dtype))
                offsets[index] = stop
            torch.autograd.backward(outputs, tuple(per_micro))
    finally:
        restore(after_first_pass)
    return (float(loss.detach()), diagnostics)
