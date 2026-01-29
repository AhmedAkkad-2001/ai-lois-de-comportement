import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import os
import imageio.v2 as imageio


# ============================================================
# 1) Données epsilon imposées (chargement uniquement)
# ============================================================
eps = np.array([
    0.0000, 0.0015, 0.0030, 0.0045, 0.0060, 0.0075, 0.0090, 0.0105, 0.0120,
    0.0135, 0.0150, 0.0165, 0.0180, 0.0190, 0.0210, 0.0240, 0.0270
], dtype=float)

# ============================================================
# 2) Paramètres matériau (MPa)
# ============================================================
E = 210e3
sigma_y0 = 250.0
H = 3000.0


# ============================================================
# 3) Modèle élastoplastique 1D traction (écrouissage isotrope)
#    Calcule sigma(eps) sur tout l'historique de chargement.
# ============================================================
def elastoplastic_1d_history(eps_hist, E, sigma_y0, H):
    eps_p = 0.0  # déformation plastique
    p = 0.0      # variable interne d'écrouissage cumulée

    sigma_hist = np.zeros_like(eps_hist, dtype=float)
    eps_p_hist = np.zeros_like(eps_hist, dtype=float)

    for i, e in enumerate(eps_hist):
        # Essai élastique
        sigma_trial = E * (e - eps_p)

        # Fonction de charge 1D
        f_trial = abs(sigma_trial) - (sigma_y0 + H * p)

        # Cas élastique
        if f_trial <= 1e-12:
            sigma = sigma_trial
        # Cas plastique : correction + mise à jour des variables internes
        else:
            dlam = f_trial / (E + H)
            s = np.sign(sigma_trial) if sigma_trial != 0 else 1.0
            eps_p += dlam * s
            p += dlam
            sigma = E * (e - eps_p)

        sigma_hist[i] = sigma
        eps_p_hist[i] = eps_p

    return sigma_hist, eps_p_hist


# Calcul de la réponse en chargement
sigma_load, eps_p_hist = elastoplastic_1d_history(eps, E, sigma_y0, H)


# ============================================================
# 4) Décharge élastique correcte jusqu'à sigma=0
#    On décharge linéairement avec pente E jusqu'à eps = eps_p_final.
# ============================================================
eps_p_final = eps_p_hist[-1]
eps_peak = eps[-1]
sigma_peak = sigma_load[-1]

N_unload = 50
eps_unload = np.linspace(eps_peak, eps_p_final, N_unload)
sigma_unload = E * (eps_unload - eps_p_final)

# Assemblage final chargement + décharge
eps_plot = np.concatenate([eps, eps_unload])
sigma_plot = np.concatenate([sigma_load, sigma_unload])


# ============================================================
# 5) Dataset PyTorch sans normalisation
#    Entrée : epsilon en % (100*eps)
#    Sortie : sigma en MPa
# ============================================================
x = torch.tensor(100 * eps_plot, dtype=torch.float32).view(-1, 1)  # ε en %
y = torch.tensor(sigma_plot, dtype=torch.float32).view(-1, 1)      # σ en MPa


# ============================================================
# 6) Réseau de neurones (MLP)
# ============================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x_in):
        return self.net(x_in)


model = MLP()
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-4)


def predict_sigma():
    """Retourne la contrainte prédite (MPa) sur les mêmes entrées x."""
    model.eval()
    with torch.no_grad():
        pred = model(x)
    return pred.numpy().flatten()


# ============================================================
# 7) Training + plots sauvegardés
#    Modif : tout sauvegarder dans outputs/ (frames + GIF)
# ============================================================
epochs = 5000
plot_every = 500

out_dir = "outputs"
frames_dir = os.path.join(out_dir, "frames_nn")
os.makedirs(frames_dir, exist_ok=True)

frame_paths = []

for epoch in range(1, epochs + 1):
    model.train()

    pred = model(x)
    loss = loss_fn(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % plot_every == 0:
        pred_sigma = predict_sigma()

        plt.figure(figsize=(9, 5))
        plt.plot(eps_plot, sigma_plot, "k-", linewidth=3, label="Contrainte réelle")
        plt.plot(eps_plot, pred_sigma, "r--", linewidth=2,
                 label=f"Contrainte prédite (NN) - Epoch {epoch}")

        plt.xlabel("Déformation ε")
        plt.ylabel("Contrainte σ (MPa)")
        plt.title("Apprentissage réseau de neurones")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        frame_file = os.path.join(frames_dir, f"frame_{epoch:05d}.png")
        plt.savefig(frame_file, dpi=120)
        frame_paths.append(frame_file)

        plt.show()
        plt.close()

        print(f"Epoch {epoch}/{epochs}  Loss = {loss.item():.6e}")


# ============================================================
# 8) Création du GIF (3 secondes par frame + pause finale 3 sec)
#    Modif : GIF sauvegardé dans outputs/
# ============================================================
gif_name = os.path.join(out_dir, "training_nn_elastoplastic_1d_load_unload.gif")

with imageio.get_writer(gif_name, mode="I", duration=3.0) as writer:
    for fp in frame_paths:
        img = imageio.imread(fp)
        writer.append_data(img)

    if len(frame_paths) > 0:
        last_img = imageio.imread(frame_paths[-1])
        writer.append_data(last_img)

print(f"\nGIF généré : {gif_name}")


# ============================================================
# 9) Etat final
# ============================================================
print("\n--- Etat final ---")
print(f"eps_p_final = {eps_p_final:.6f}")
print(f"sigma_peak  = {sigma_peak:.3f} MPa")


