import torch
from torch_sparse_solve_cpp import solve_forward
from torch_sparse_solve_cpp import solve_backward

class Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        if not A.dtype == torch.float64:
            A = A.double()
        if not b.dtype == torch.float64:
            b = b.double()
        x = solve_forward(A, b)
        ctx.save_for_backward(A, b, x)
        return x

    @staticmethod
    def backward(ctx, grad):
        A, b, x = ctx.saved_tensors
        gradA, gradb = solve_backward(grad, A, b, x)
        return gradA, gradb