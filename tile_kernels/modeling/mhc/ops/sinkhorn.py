import torch

from tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_bwd, _mhc_sinkhorn_fwd


class _SinkhornNormalize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: '_SinkhornNormalize',
        x: torch.Tensor,
        repeat: int,
        eps: float,
        kernel_config: dict[str, dict[str, int]] | None = None,
    ) -> torch.Tensor:
        hidden_size = x.shape[1]
        output = torch.empty_like(x)
        kernel_config = kernel_config or {}
        fwd_kernel = _mhc_sinkhorn_fwd(hidden_size, repeat=repeat, eps=eps, **{'token_block_size': 1, **kernel_config.get('fwd', {})})
        bwd_kernel = _mhc_sinkhorn_bwd(hidden_size, repeat=repeat, eps=eps, **{'token_block_size': 32, **kernel_config.get('bwd', {})})
        ctx.save_for_backward(x)
        ctx.bwd_kernel = bwd_kernel
        fwd_kernel(x, output)
        return output

    @staticmethod
    def backward(ctx: '_SinkhornNormalize', grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        x = ctx.saved_tensors[0]
        grad_input = torch.empty_like(x)
        ctx.bwd_kernel(grad_output, x, grad_input)
        return grad_input, None, None, None


def sinkhorn_normalize(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
    *,
    kernel_config: dict[str, dict[str, int]] | None = None,
) -> torch.Tensor:
    return _SinkhornNormalize.apply(x.contiguous().view(-1, *x.shape[-2:]), repeat, eps, kernel_config).view_as(x)
