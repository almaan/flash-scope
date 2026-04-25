from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad

from flash_scope._utils import resolve_device, to_dense_array
from flash_scope.model._deconv import FlashScopeModel


def fit(
    model: FlashScopeModel,
    adata: ad.AnnData,
    epochs: int = 5000,
    batch_size: int = 1024,
    lr: float = 0.01,
    device: str = "auto",
    use_compile: bool = True,
    verbose: bool = False,
    grad_clip: float | None = None,
    tol: float = 1e-4,
    patience: int = 50,
    l1_w: float = 0.0,
) -> FlashScopeModel:
    """Train a FlashScopeModel on spatial expression data.

    Uses Adam with optional early stopping. Returns the model with best
    validation loss on CPU.

    Parameters
    ----------
    model : FlashScopeModel
        Initialized model (may have custom ``init_w``).
    adata : AnnData
        Spatial expression data (dense or sparse ``.X``).
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for Adam.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    use_compile : bool
        Apply ``torch.compile`` on CUDA.
    verbose : bool
        Print progress every ~10% of epochs.
    grad_clip : float or None
        Max gradient norm. ``None`` disables.
    tol : float
        Relative improvement threshold for early stopping.
    patience : int
        Epochs without improvement before stopping. 0 disables.
    l1_w : float
        L1 penalty on mixing weights. 0 disables.

    Returns
    -------
    FlashScopeModel
        Trained model on CPU.
    """
    dev = resolve_device(device)
    use_cuda = dev.type == "cuda"

    X = torch.tensor(to_dense_array(adata.X), dtype=torch.float32)
    lgamma_X1 = torch.lgamma(X + 1)
    indices = torch.arange(X.shape[0], dtype=torch.long)
    dataset = TensorDataset(X, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = model.to(dev)
    lgamma_X1 = lgamma_X1.to(dev)

    amp_dtype = torch.float16 if use_cuda else torch.bfloat16

    def _train_step(x_batch, idx_batch, lgamma_x1_batch):
        with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_cuda):
            r_sp = model(x_batch, idx_batch)
        r_sp = r_sp.float()
        return model.loss(x_batch, r_sp, lgamma_x1_batch)

    train_step = _train_step
    if use_compile and use_cuda:
        train_step = torch.compile(_train_step)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"[flash-scope] fitting on {dev} ({epochs} epochs, batch_size={batch_size})")
        log_every = max(1, epochs // 10)

    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, idx_batch in loader:
            x_batch = x_batch.to(dev, non_blocking=use_cuda)
            idx_batch = idx_batch.to(dev, non_blocking=use_cuda)
            lgamma_x1_batch = lgamma_X1[idx_batch]

            loss = train_step(x_batch, idx_batch, lgamma_x1_batch)
            if l1_w > 0:
                loss = loss + l1_w * model._softplus(model.w[idx_batch]).abs().sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"[flash-scope]   epoch {epoch + 1:>{len(str(epochs))}}/{epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss * (1 - tol):
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if patience > 0 and stale >= patience:
            if verbose:
                print(f"[flash-scope]   early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.cpu()
    return model
