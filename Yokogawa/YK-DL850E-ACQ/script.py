import matplotlib.pyplot as plt
import pandas as pd

# # Provided dataset
# data = {
#     "CH1_Force[N]": [
#         -1.493868366, -0.800542805, 1.222806882, 3.953236429,
#         7.394296859, 9.465135571, 11.86882358, 13.39386994,
#         15.45322701, 8.635853147, 6.319638689, 5.030972152,
#         3.991232384, 2.97453876, 1.630239516, 0.142171156,
#         -0.147260957
#     ],
#     "CH2_Displacement[mm]": [
#         -0.06763084, -0.070670027, 0.154609715, 0.997794178,
#         1.737456331, 2.202451953, 2.750075474, 2.984662726,
#         3.405400186, 2.988651659, 2.596026679, 2.120203953,
#         1.567641754, 1.053259342, 0.270288773, -0.068580586,
#         -0.069340383
#     ]
# }

# df = pd.DataFrame(data)

# # Plot Force vs Displacement
# plt.figure(figsize=(7,5))
# plt.plot(df["CH2_Displacement[mm]"], df["CH1_Force[N]"], marker="o", linestyle="-")
# plt.xlabel("Displacement [mm]")
# plt.ylabel("Force [N]")
# plt.title("mbuckle manual1 Force vs. Displacement")
# plt.grid(True)
# plt.savefig("250911_mbuckle_manual1_fvsd.png")


import os
import numpy as np

# ----------------------- user inputs -----------------------
# Path to your 2-column data file: displacement [m], force [N]
# Works with CSV or whitespace-separated TXT; header is OK.

base_dir = r"Z:\Users\Soyeon\QZS_COMSOL_Theory_Curves"
file_name = "Non-rigid-center.txt"
data_path = os.path.join(base_dir,file_name)   # <- change to your file

# Masses to evaluate (kg)
masses = [0.0807, 0.9588, 1.107, 0.5, 1.26]

g = 9.80665             # m/s^2
weight_sign = +1        # use +1 if equilibrium solves F(x)=+m*g; use -1 if F(x)=-m*g
fit_window_pts = 7      # number of nearest points for local poly fit (odd number >=5)
poly_deg = 3            # degree of local polynomial used to estimate slope
# -----------------------------------------------------------


def load_fx(path):
    """Load displacement x (m) and force F (N) from a 2-column file."""
    # try comma, then whitespace
    try:
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
    except Exception:
        arr = np.loadtxt(path, ndmin=2)
    # If there is a header row with text, try genfromtxt
    if arr.size == 0 or np.isnan(arr).any():
        arr = np.genfromtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    x, F = arr[:, 0].astype(float), arr[:, 1].astype(float)
    # sort by x
    idx = np.argsort(x)
    return x[idx], F[idx]


def interp_x_for_force(x, F, F_target):
    """
    Find x such that F(x) ~= F_target using linear interpolation.
    Assumes F is (mostly) monotone in the region of interest.
    """
    # Ensure strictly increasing F for np.interp; if not, sort by F for this step
    order = np.argsort(F)
    F_sorted, x_sorted = F[order], x[order]

    # Guard: target outside measured force range
    if F_target < F_sorted[0] or F_target > F_sorted[-1]:
        return np.nan

    x_eq = np.interp(F_target, F_sorted, x_sorted)
    return x_eq


def local_keff(x, F, x0, window_pts=7, deg=3):
    """
    Estimate dF/dx at x0 by fitting a local polynomial in a neighborhood.
    """
    if np.isnan(x0):
        return np.nan

    # pick nearest window_pts by |x - x0|
    i_sorted = np.argsort(np.abs(x - x0))[:window_pts]
    i_sorted = np.sort(i_sorted)
    xx, FF = x[i_sorted], F[i_sorted]

    # If not enough unique points for the desired degree, lower degree
    unique_x = np.unique(xx)
    d = min(deg, len(unique_x) - 1)
    if d < 1:
        # fallback to simple finite difference with closest two unique points
        i2 = np.argsort(np.abs(x - x0))[:2]
        return np.diff(F[i2]) / np.diff(x[i2])

    # Fit polynomial F(x) = a0 + a1 x + a2 x^2 + ...
    coeffs = np.polyfit(xx, FF, deg=d)
    # derivative polynomial
    dcoeffs = np.polyder(coeffs)
    k_eff = np.polyval(dcoeffs, x0)
    return k_eff


def main():
    x, F = load_fx(data_path)

    print(f"{'Mass (kg)':>10} {'x_eq (mm)':>12} {'k_eff (N/m)':>14} {'ω (rad/s)':>12} {'f (Hz)':>10}")
    for m in masses:
        F_target = weight_sign * m * g
        x_eq = interp_x_for_force(x, F, F_target)
        k_eff = local_keff(x, F, x_eq, window_pts=fit_window_pts, deg=poly_deg)

        if np.isnan(x_eq) or np.isnan(k_eff) or k_eff <= 0:
            print(f"{m:10.4f} {'—':>12} {'—':>14} {'—':>12} {'—':>10}")
            continue

        omega = np.sqrt(k_eff / m)            # rad/s
        freq = omega / (2 * np.pi)            # Hz
        print(f"{m:10.4f} {x_eq*1e3:12.3f} {k_eff:14.1f} {omega:12.2f} {freq:10.2f}")


if __name__ == "__main__":
    main()
