from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad

from flash_scope._utils import resolve_device, to_dense_array
from flash_scope.model._deconv import FlashScopeModel


def fit(
    model: FlashScopeModel,
    adata: ad.AnnData,
    epochs: int = 500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "auto",
    use_compile: bool = True,
) -> FlashScopeModel:
    dev = resolve_device(device)
    use_cuda = dev.type == "cuda"

    X = torch.tensor(to_dense_array(adata.X), dtype=torch.float32)
    indices = torch.arange(X.shape[0], dtype=torch.long)
    dataset = TensorDataset(X, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = model.to(dev)

    train_model = model
    if use_compile and use_cuda:
        train_model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16

    for _epoch in range(epochs):
        for x_batch, idx_batch in loader:
            x_batch = x_batch.to(dev, non_blocking=use_cuda)
            idx_batch = idx_batch.to(dev, non_blocking=use_cuda)

            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_cuda):
                r_sp, p_sp = train_model(x_batch, idx_batch)
                loss = model.loss(x_batch, r_sp, p_sp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model = model.cpu()
    return model
