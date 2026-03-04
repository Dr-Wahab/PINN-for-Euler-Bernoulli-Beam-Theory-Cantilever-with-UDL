# PINN-Cantilever: Physics-Informed Neural Network for Beam Deflection

This project implements a Physics-Informed Neural Network (PINN) to solve the 4th-order Euler-Bernoulli beam equation for a cantilever beam under a Uniform Distributed Load (UDL).

## Technical Overview
The model approximates the deflection $w(x)$ by minimizing a loss function composed of:
1. **The PDE Residual**: $EI \frac{d^4w}{dx^4} - q = 0$
2. **Kinematic BCs (x=0)**: Zero deflection and zero slope.
3. **Static BCs (x=L)**: Zero bending moment and zero shear force.



## Implementation Details
- **Framework**: PyTorch
- **Activation**: Tanh (to ensure non-zero 4th-order gradients)
- **Optimizer**: Adam (can be fine-tuned with L-BFGS)
- **Differentiation**: Exact automatic differentiation via `torch.autograd`
