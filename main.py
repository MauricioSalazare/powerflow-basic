from tensorpowerflow import GridTensor
import numpy as np
import matplotlib
from scipy.stats import multivariate_normal
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#%%
def tensor_solve_two_nodes(*, r_ohm, x_ohm, s_kw: np.ndarray, s_kvar:np.ndarray, s_base=1_000, v_base=11):

    assert (s_kw.ndim == 1) and (s_kvar.ndim == 1), "Power should be 1-D"
    assert (s_kw.shape == s_kvar.shape), "Power must have the same shape"

    v1_sols = []
    v2_sols = []
    for active, reactive in zip(s_kw, s_kvar):
        v1_sol, v2_sol = solve_two_nodes(r_ohm, x_ohm, active, reactive, s_base, v_base)
        v1_sols.append(v1_sol)
        v2_sols.append(v2_sol)

    return np.array(v1_sols), np.array(v2_sols)

def solve_two_nodes(r_ohm, x_ohm, s_kw, s_kvar, s_base=1_000, v_base=11):
    z_base = (v_base ** 2 * 1000) / s_base
    z_pu = (r_ohm + 1j * x_ohm) / z_base

    # % 2-bus system solution
    s = (s_kw + 1j * s_kvar) / s_base
    v0 = 1 + 1j * 0.0

    A = 1
    B = 2 * (z_pu.real * s.real + z_pu.imag * s.imag) - np.abs(v0) ** 2
    C = (z_pu.real * s.real + z_pu.imag * s.imag) ** 2 + (z_pu.real * s.imag - z_pu.imag * s.real) ** 2
    DISCRIMINANT = B ** 2 - 4 * A * C

    v1 = np.sqrt((-B + np.sqrt(DISCRIMINANT)) / (2 * A))
    v2 = np.sqrt((-B - np.sqrt(DISCRIMINANT)) / (2 * A))

    theta_v1 = np.arcsin((z_pu.real * s.imag - z_pu.imag * s.real) / (v1 * np.abs(v0)))
    theta_v2 = np.arcsin((z_pu.real * s.imag - z_pu.imag * s.real) / (v2 * np.abs(v0)))

    v1_sol = v1 * (np.cos(theta_v1) + 1j * np.sin(theta_v1))
    v2_sol = v2 * (np.cos(theta_v2) + 1j * np.sin(theta_v2))

    return v1_sol, v2_sol


#%%
network = GridTensor(node_file_path="data/Nodes_2.csv",
                     lines_file_path="data/Lines_2.csv",
                     gpu_mode=False)
active_power = 50.0
reactive_power = 10.0
solutions = network.run_pf(active_power=np.array([[active_power]]), reactive_power=np.array([[reactive_power]]))
v1_sol, v2_sol = solve_two_nodes(r_ohm=network.branch_info['R'].values[0],
                                 x_ohm=network.branch_info['X'].values[0],
                                 s_kw=active_power,
                                 s_kvar=reactive_power)
print("\n\n\n\n")
print(f"Analytical:       {v1_sol}")
print(f"Solution tensor:  {solutions['v'][0][0]}")
print(f"Analytical (abs):      {np.abs(v1_sol)}")
print(f"Solution tensor (abs): {np.abs(solutions['v'][0][0])}")

print(f"Analytical v2:       {v2_sol}")


#%%
mu_X = np.array([1000, 500])
sigma_X = np.array([[1_000.0,   500.0],
                    [  500.0, 1_000.0]])
mvn = multivariate_normal(mean=mu_X, cov=sigma_X)
mnv_samples = mvn.rvs(size=1_000)


solutions = network.run_pf(active_power=np.atleast_2d(mnv_samples[:,0]).T,
                           reactive_power=np.atleast_2d(mnv_samples[:,1]).T)
solutions_tensor = solutions['v'].flatten()

v1_anal, v2_anal = tensor_solve_two_nodes(r_ohm=network.branch_info['R'].values[0],
                                         x_ohm=network.branch_info['X'].values[0],
                                         s_kw=mnv_samples[:,0],
                                         s_kvar=mnv_samples[:,1])


fig, ax = plt.subplots(1,2, figsize=(7, 3))
plt.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.15)
ax[0].scatter(mnv_samples[:,0]/1000, mnv_samples[:, 1]/1000, facecolors='none', edgecolors='C0', s=10, zorder=2)
ax[1].scatter(solutions_tensor.real, solutions_tensor.imag, facecolors='none', edgecolors='C1', s=10, zorder=2)
ax[1].scatter(v2_anal.real, v2_anal.imag, facecolors='none', edgecolors='C3', s=10, zorder=2)

ax[0].set_xlim([0, 4])
ax[0].set_ylim([0, 2])
ax[0].grid(zorder=0)

ax[1].set_xlim([0, 1.1])
ax[1].set_ylim([0, 0.2])
ax[1].grid(zorder=0)

ax[0].set_xlabel("Active [p.u.]", fontsize='x-small')
ax[0].set_ylabel("Reactive [p.u]", fontsize='x-small')

ax[1].set_xlabel("V(real) [p.u.]", fontsize='x-small')
ax[1].set_ylabel("V(imag) [p.u]", fontsize='x-small')