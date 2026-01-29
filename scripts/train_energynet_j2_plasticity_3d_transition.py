import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ============================================================
# 0) TORCH PRECISION (meilleur gradient)
# ============================================================
torch.set_default_dtype(torch.float64)


# ============================================================
# 1) MECANIQUE : J2 Plasticité + énergie
# ============================================================

def isotropic_C_voigt(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    C = np.zeros((6, 6), dtype=float)

    # Partie normale
    C[0, 0] = lam + 2 * mu;  C[0, 1] = lam;           C[0, 2] = lam
    C[1, 0] = lam;           C[1, 1] = lam + 2 * mu;  C[1, 2] = lam
    C[2, 0] = lam;           C[2, 1] = lam;           C[2, 2] = lam + 2 * mu

    # Cisaillement (engineering)
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu

    return C, mu


def voigt_to_tensor_sigma(sig_v):
    return np.array([
        [sig_v[0], sig_v[3], sig_v[5]],
        [sig_v[3], sig_v[1], sig_v[4]],
        [sig_v[5], sig_v[4], sig_v[2]]
    ], dtype=float)


def deviatoric_sigma_voigt(sig_v):
    sig = voigt_to_tensor_sigma(sig_v)
    tr = np.trace(sig)
    s = sig - (tr / 3.0) * np.eye(3)

    return np.array([
        s[0, 0], s[1, 1], s[2, 2],
        s[0, 1], s[1, 2], s[0, 2]
    ])


def von_mises(sig_v):
    s_v = deviatoric_sigma_voigt(sig_v)
    s11, s22, s33, s12, s23, s13 = s_v
    ss = s11**2 + s22**2 + s33**2 + 2*(s12**2 + s23**2 + s13**2)
    return np.sqrt(1.5 * ss)


def psi_energy(eps, eps_p, C, H, k):
    ee = eps - eps_p
    return 0.5 * float(ee.T @ (C @ ee)) + 0.5 * H * k**2


def elastoplastic_update(eps, eps_p_old, k_old, C, mu, sig_y0, H):
    """
    Return mapping J2 + écrouissage isotrope.
    eps: (6,)
    """
    # trial
    eps_e_trial = eps - eps_p_old
    sigma_trial = C @ eps_e_trial

    sigma_eq_trial = von_mises(sigma_trial)
    f_trial = sigma_eq_trial - (sig_y0 + H * k_old)

    if f_trial <= 0.0:
        sigma = sigma_trial
        eps_p_new = eps_p_old.copy()
        k_new = k_old
        Psi = psi_energy(eps, eps_p_new, C, H, k_new)
        return sigma, eps_p_new, k_new, Psi, f_trial, False

    # retour radial
    dgamma = f_trial / (3.0 * mu + H)

    s_trial = deviatoric_sigma_voigt(sigma_trial)
    s11, s22, s33, s12, s23, s13 = s_trial
    norm_s = np.sqrt(s11**2 + s22**2 + s33**2 + 2*(s12**2 + s23**2 + s13**2))

    n_v = s_trial / (norm_s + 1e-20)

    dep_voigt = dgamma * 1.5 * np.array([
        n_v[0], n_v[1], n_v[2],
        2.0*n_v[3], 2.0*n_v[4], 2.0*n_v[5]
    ])

    eps_p_new = eps_p_old + dep_voigt
    k_new = k_old + dgamma

    sigma = C @ (eps - eps_p_new)
    Psi = psi_energy(eps, eps_p_new, C, H, k_new)

    return sigma, eps_p_new, k_new, Psi, f_trial, True


# ============================================================
# 2) DATASET : transition (eps, eps_p_old, k_old) -> (Psi, sigma, eps_p_new, k_new)
# ============================================================

def generate_background_eps(n=20, step=0.002, mode="uniaxial"):
    """
    mode="uniaxial": epsilon11 augmente, le reste = 0
    mode="all": toutes composantes augmentent
    """
    eps = np.zeros((n, 6), dtype=float)
    for i in range(n):
        if mode == "uniaxial":
            eps[i, 0] = i * step
        else:
            eps[i, :] = i * step
    return eps


def build_dataset_transition(ncases=20, seed=123, mode="all"):
    np.random.seed(seed)

    # matériau
    E = 210e9
    nu = 0.30
    sig_y0 = 250e6
    H = 1e9
    C, mu = isotropic_C_voigt(E, nu)

    eps_list = generate_background_eps(n=ncases, step=0.001, mode=mode)

    eps_p = np.zeros(6)
    k = 0.0

    X = []
    Y_psi = []
    Y_sigma = []
    Y_epsp_new = []
    Y_k_new = []

    for eps in eps_list:
        eps_p_old = eps_p.copy()
        k_old = float(k)

        sigma, eps_p, k, Psi, _, _ = elastoplastic_update(eps, eps_p, k, C, mu, sig_y0, H)

        x = np.concatenate([eps, eps_p_old, np.array([k_old])])  # (13,)
        X.append(x)

        Y_psi.append(Psi)
        Y_sigma.append(sigma)
        Y_epsp_new.append(eps_p.copy())
        Y_k_new.append(float(k))

    return (
        np.array(X, float),                          # (N,13)
        np.array(Y_psi, float).reshape(-1, 1),       # (N,1)
        np.array(Y_sigma, float),                    # (N,6)
        np.array(Y_epsp_new, float),                 # (N,6)
        np.array(Y_k_new, float).reshape(-1, 1)      # (N,1)
    )


# ============================================================
# 3) NN multi-têtes : X(13) -> Psi + eps_p_new + k_new
# ============================================================

class MultiHeadEnergyNet(nn.Module):
    def __init__(self, in_dim=13, width=128, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.Softplus(beta=2.0))
            d = width
        self.trunk = nn.Sequential(*layers)

        self.head_psi = nn.Linear(width, 1)
        self.head_epsp = nn.Linear(width, 6)
        self.head_k = nn.Linear(width, 1)

    def forward(self, x):
        h = self.trunk(x)
        psi = self.head_psi(h)
        epsp = self.head_epsp(h)
        k = self.head_k(h)
        return psi, epsp, k


# ============================================================
# 4) Frames + GIF
# ============================================================

def save_frame(epoch, y_real, y_pred, ylabel, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    cases = np.arange(1, len(y_real) + 1)
    shift = 0.12

    plt.figure(figsize=(8, 4.5))
    plt.plot(cases, y_real, "o", markersize=7, label=f"{ylabel} réel (o)")
    plt.plot(cases + shift, y_pred, "*", markersize=11, label=f"{ylabel} NN (*)")

    for i in range(len(cases)):
        plt.plot([cases[i], cases[i] + shift], [y_real[i], y_pred[i]], "-", alpha=0.30)

    plt.title(f"{title} - Epoch {epoch}")
    plt.xlabel("Échantillons de déformation")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(outdir, f"frame_{epoch:04d}.png")
    plt.savefig(fname, dpi=140)
    plt.close()


def make_gif(frame_dir, gif_name, duration=0.9):
    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    if len(files) == 0:
        print("Aucune image trouvée pour créer le GIF :", gif_name)
        return

    images = [imageio.imread(os.path.join(frame_dir, f)) for f in files]
    imageio.mimsave(gif_name, images, duration=duration)
    print("GIF créé :", gif_name)


# ============================================================
# 5) Training
# ============================================================

def main():
    # Dossier de sortie global
    out_dir = "outputs2"
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # A) Dataset transition
    # --------------------------------------------------------
    X, Y_Psi, Y_sigma, Y_epsp_new, Y_k_new = build_dataset_transition(
        ncases=20, seed=123, mode="all"
    )

    # --------------------------------------------------------
    # B) Normalisations
    # --------------------------------------------------------
    Xmean, Xstd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Xn = (X - Xmean) / Xstd

    Pmean, Pstd = Y_Psi.mean(axis=0), Y_Psi.std(axis=0) + 1e-12
    Pn = (Y_Psi - Pmean) / Pstd

    Epmean, Epstd = Y_epsp_new.mean(axis=0), Y_epsp_new.std(axis=0) + 1e-12
    Epn = (Y_epsp_new - Epmean) / Epstd

    Kmean, Kstd = Y_k_new.mean(axis=0), Y_k_new.std(axis=0) + 1e-12
    Kn = (Y_k_new - Kmean) / Kstd

    sigma_scale = float(np.mean(Y_sigma.std(axis=0) + 1e-12))

    # --------------------------------------------------------
    # C) Torch tensors
    # --------------------------------------------------------
    Xn_t = torch.tensor(Xn, requires_grad=True)
    Pn_t = torch.tensor(Pn)
    Epn_t = torch.tensor(Epn)
    Kn_t = torch.tensor(Kn)
    Ysigma_t = torch.tensor(Y_sigma)

    Xstd_t = torch.tensor(Xstd)
    Xstd_eps_t = Xstd_t[:6]
    Pstd0 = float(Pstd[0])

    # --------------------------------------------------------
    # D) Modèle + optim
    # --------------------------------------------------------
    model = MultiHeadEnergyNet(in_dim=13, width=128, depth=3)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)

    mse = nn.MSELoss()

    w_psi = 1.0
    w_sig = 1.0
    w_epsp = 1.0
    w_k = 1.0

    # Frames (dans outputs2/)
    dir_energy = os.path.join(out_dir, "frames_energy")
    dir_sigma11 = os.path.join(out_dir, "frames_sigma11")

    snapshots = [1, 1000, 3500]
    nepochs = 3500

    for epoch in range(1, nepochs + 1):
        model.train()

        psi_n, epsp_n, k_n = model(Xn_t)

        Psi_sum = psi_n.sum()
        dPsi_dXn = torch.autograd.grad(Psi_sum, Xn_t, create_graph=True)[0]
        dPsi_deps_n = dPsi_dXn[:, :6]

        sigma_pred = (Pstd0 * dPsi_deps_n / Xstd_eps_t)

        loss_psi = mse(psi_n, Pn_t)
        loss_sig = torch.mean(((sigma_pred - Ysigma_t) / sigma_scale) ** 2)
        loss_epsp = mse(epsp_n, Epn_t)
        loss_k = mse(k_n, Kn_t)

        loss = w_psi * loss_psi + w_sig * loss_sig + w_epsp * loss_epsp + w_k * loss_k

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss={loss.item():.3e} "
            )

        if epoch in snapshots:
            model.eval()

            with torch.no_grad():
                psi_n_eval, _, _ = model(torch.tensor(Xn))
                psi_eval = psi_n_eval.detach().cpu().numpy() * Pstd + Pmean

            Xtmp = torch.tensor(Xn, requires_grad=True)
            psi_n2, _, _ = model(Xtmp)
            dPsi_dXn2 = torch.autograd.grad(psi_n2.sum(), Xtmp, create_graph=False)[0]
            sigma_pred2 = (Pstd0 * dPsi_dXn2[:, :6] / Xstd_eps_t).detach().cpu().numpy()
            sigma11_pred = sigma_pred2[:, 0]

            save_frame(epoch, Y_Psi.flatten(), psi_eval.flatten(), "Psi", "Energie", dir_energy)
            save_frame(epoch, Y_sigma[:, 0].flatten(), sigma11_pred.flatten(),
                       "sigma_11", "Contrainte sigma11", dir_sigma11)

    gif_energy = os.path.join(out_dir, "training_energy_j2_3d.gif")
    gif_sigma11 = os.path.join(out_dir, "training_sigma11_j2_3d.gif")

    make_gif(dir_energy, gif_energy, duration=0.9)
    make_gif(dir_sigma11, gif_sigma11, duration=0.9)


if __name__ == "__main__":
    main()
