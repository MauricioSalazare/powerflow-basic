from tensorpowerflow import GridTensor
import numpy as np
from time import perf_counter
import matplotlib
matplotlib.use("TkAgg")


#%%
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



