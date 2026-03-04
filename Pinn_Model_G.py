# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:18:39 2026

@author: Wahab
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# 0) SETTINGS (stability for 4th derivatives)
# ============================================================
torch.manual_seed(0)
np.random.seed(0)

torch.set_default_dtype(torch.float64)  # IMPORTANT for w'''' stability
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# 1) BEAM INPUTS  (EDIT THESE)
# ============================================================
L = 2.0        # m
E = 200e9      # Pa
I = 8e-6       # m^4

P = 10e3       # N (magnitude you input)
P_load = -abs(P)  # DOWNWARD tip load = negative (OUR CONVENTION)

EI = E * I
w_ref = abs(P_load) * L**3 / (3.0 * EI)  # positive scaling magnitude
print(f"EI = {EI:.3e} N·m^2")
print(f"P_load = {P_load:.3e} N (downward)")
print(f"w_ref = {w_ref:.3e} m (tip deflection magnitude scale)")

# ============================================================
# 2) PINN: wbar(xi) with hard clamp BC
#    xi = x/L in [0,1]
#    w = sign(P_load) * w_ref * wbar(xi)
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xi):
        # Trial function to enforce clamp BCs exactly:
        # wbar(0)=0 and wbar'(0)=0
        N = self.net(xi)
        return (xi**2) * N

model = MLP(hidden=64, depth=4).to(device)

# ============================================================
# 3) AUTODIFF HELPERS
# ============================================================
def d(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

def derivs_wbar(model, xi):
    xi = xi.clone().detach().requires_grad_(True)
    w  = model(xi)
    w1 = d(w, xi)
    w2 = d(w1, xi)
    w3 = d(w2, xi)
    w4 = d(w3, xi)
    return w, w1, w2, w3, w4

# ============================================================
# 4) TRAINING POINTS
# ============================================================
N_f = 4000
xi_f = torch.rand(N_f, 1, device=device).clamp(1e-4, 1-1e-4)
xi1 = torch.tensor([[1.0]], device=device)

# ============================================================
# 5) LOSS (nondimensional PDE and tip BCs)
#
# Interior: wbar''''(xi) = 0
# Tip BCs:
#   wbar''(1) = 0
#   wbar'''(1) = s
#
# where s = +1 means the solution bends "positive" in your plotted sign.
# Since we want downward deflection NEGATIVE, we apply the sign later in w.
# Therefore we target wbar'''(1) = +1 and multiply w by sign(P_load)=-1.
# This keeps the training stable and consistent.
# ============================================================
lam_pde = 1.0
lam_tip = 50.0

def compute_loss():
    # PDE residual
    _, _, _, _, w4 = derivs_wbar(model, xi_f)
    loss_pde = torch.mean(w4**2)

    # Tip BCs: wbar''(1)=0, wbar'''(1)=+1
    _, _, w2_1, w3_1, _ = derivs_wbar(model, xi1)
    loss_tip = torch.mean(w2_1**2) + torch.mean((w3_1 - 1.0)**2)

    loss = lam_pde * loss_pde + lam_tip * loss_tip
    return loss, loss_pde, loss_tip

# ============================================================
# 6) STAGE 1: ADAM
# ============================================================
adam = optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(1, 4001):
    adam.zero_grad()
    loss, lpde, ltip = compute_loss()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    adam.step()

    if epoch % 500 == 0:
        print(f"[Adam] Epoch {epoch:4d} | Total {loss.item():.3e} | PDE {lpde.item():.3e} | Tip {ltip.item():.3e}")

# ============================================================
# 7) STAGE 2: LBFGS POLISH
# ============================================================
lbfgs = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=500,
    history_size=50,
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()
    loss, _, _ = compute_loss()
    loss.backward()
    return loss

final_loss = lbfgs.step(closure)
print("[LBFGS] Final loss:", float(final_loss))

# ============================================================
# 8) POST-PROCESS: convert to physical units
# ============================================================
xi = torch.linspace(0, 1, 401, device=device).view(-1, 1)

wbar, wbar1, wbar2, wbar3, _ = derivs_wbar(model, xi)

# physical coordinate
x = (L * xi).detach().cpu().numpy().ravel()

# Apply load sign consistently:
# downward load => sign = -1 => downward deflection negative
sgn = np.sign(P_load)  # -1 for downward

# w(x)
w = (sgn * w_ref * wbar).detach().cpu().numpy().ravel()

# theta = dw/dx
theta = (sgn * (w_ref / L) * wbar1).detach().cpu().numpy().ravel()

# M = -EI * d2w/dx2
M = (-EI * (sgn * w_ref / L**2) * wbar2).detach().cpu().numpy().ravel()

# V = -EI * d3w/dx3
V = (-EI * (sgn * w_ref / L**3) * wbar3).detach().cpu().numpy().ravel()

# ============================================================
# 9) ANALYTICAL (same sign convention)
# For downward load P_load < 0:
# w_true(x) = P_load * x^2 (3L - x) / (6EI)  (negative)
# M_true(x) = -P_load (L - x) ??? careful:
# Standard: M(x) = -EI w'' = P_load*(L-x)  (with P_load negative -> M negative)
# V(x) = -EI w''' = P_load (constant)
# ============================================================
x_t = torch.tensor(x, device=device).view(-1, 1)
w_true = (P_load * x_t**2 * (3.0 * L - x_t)) / (6.0 * EI)
w_true = w_true.detach().cpu().numpy().ravel()

M_true = (P_load * (L - x_t))  # equals -EI w''; with P_load negative => M negative
M_true = M_true.detach().cpu().numpy().ravel()

V_true = (P_load * torch.ones_like(x_t))  # constant shear
V_true = V_true.detach().cpu().numpy().ravel()

mse_w = np.mean((w - w_true)**2)
print("MSE(w) vs analytical:", mse_w)

print(f"Tip deflection PINN = {w[-1]:.6e} m")
print(f"Tip deflection TRUE = {P_load*L**3/(3*EI):.6e} m")
print(f"Shear (PINN) at tip ≈ {V[-1]:.3e} N, expected {P_load:.3e} N")
print(f"Moment (PINN) at fixed ≈ {M[0]:.3e} N·m, expected {P_load*L:.3e} N·m")

# ============================================================
# 10) PLOTS
# ============================================================
plt.figure()
plt.plot(x, w, label="PINN w(x)")
plt.plot(x, w_true, "--", label="Analytical w(x)")
plt.xlabel("x (m)")
plt.ylabel("w (m)")
plt.title("Cantilever beam (tip load): deflection (downward negative)")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(x, theta, label="PINN θ(x)=w'(x)")
plt.xlabel("x (m)")
plt.ylabel("θ (rad)")
plt.title("Rotation (downward negative convention)")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(x, M, label="PINN M(x)=-EI w''(x)")
plt.plot(x, M_true, "--", label="Analytical M(x)")
plt.xlabel("x (m)")
plt.ylabel("M (N·m)")
plt.title("Bending moment (should be linear, negative for downward load)")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(x, V, label="PINN V(x)=-EI w'''(x)")
plt.plot(x, V_true, "--", label="Analytical V(x)")
plt.xlabel("x (m)")
plt.ylabel("V (N)")
plt.title("Shear force (should be constant = P_load)")
plt.grid(True)
plt.legend()

plt.show()