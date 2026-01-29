import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import os
import imageio.v2 as imageio


# ============================================================
# Script : apprentissage NN d'une loi élastoplastique 1D
# - Chargement : epsilon sinusoïdal monotone (20 points)
# - Cible : sigma (MPa) calculée par un modèle élastoplastique 1D
# - Réseau : MLP simple (1 entrée -> 1 sortie)
# - Normalisation : sigma en GPa (= MPa/1000) pour stabiliser la loss
# - Sauvegardes : figures + GIF dans outputs/
# ============================================================

def main():
    # ============================================================
    # 1) Génération du chargement : epsilon sinusoïdal monotone
    #    x dans [0, pi/2] => sin(x) augmente => pas d'hystérésis
    # ============================================================
    N = 20
    eps_max = 0.005  # amplitude (adapter si tu veux plus/moins de plasticité)

    x_sin = np.linspace(0, np.pi/2, N)   # monotone croissant
    eps = eps_max * np.sin(x_sin)        # epsilon sinusoïdal monotone

    # Visualisation des 20 valeurs de epsilon
    plt.figure(figsize=(6, 4))
    plt.plot(range(N), eps, "o-", linewidth=2)
    plt.xlabel("Indice (0..19)")
    plt.ylabel("ε")
    plt.title("ε sinusoïdal monotone (20 points)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 2) Paramètres matériau (MPa)
    # ============================================================
    E = 210e3        # module d'Young (MPa)
    sigma_y0 = 250.0 # limite élastique initiale (MPa)
    H = 3000.0       # écrouissage isotrope (MPa)

    # ============================================================
    # 3) Modèle élastoplastique 1D (écrouissage isotrope)
    #    - calcule sigma(eps) sur tout l'historique eps
    # ============================================================
    def elastoplastic_1d_history(eps_hist, E, sigma_y0, H):
        eps_p = 0.0   # déformation plastique
        p = 0.0       # variable d'écrouissage cumulée

        sigma_hist = np.zeros_like(eps_hist, dtype=float)
        eps_p_hist = np.zeros_like(eps_hist, dtype=float)

        for i, e in enumerate(eps_hist):
            # Essai élastique
            sigma_trial = E * (e - eps_p)

            # Fonction de charge (1D) : |sigma_trial| - (sigma_y0 + H*p)
            f_trial = abs(sigma_trial) - (sigma_y0 + H * p)

            # Cas élastique
            if f_trial <= 1e-12:
                sigma = sigma_trial
            # Cas plastique : correction + mise à jour des variables internes
            else:
                dlam = f_trial / (E + H)  # multiplicateur plastique (1D)
                s = np.sign(sigma_trial) if sigma_trial != 0 else 1.0

                eps_p += dlam * s         # mise à jour eps plastique
                p += dlam                 # mise à jour écrouissage
                sigma = E * (e - eps_p)   # sigma corrigée

            sigma_hist[i] = sigma
            eps_p_hist[i] = eps_p

        return sigma_hist, eps_p_hist

    sigma_plot, eps_p_hist = elastoplastic_1d_history(eps, E, sigma_y0, H)
    eps_plot = eps

    # ============================================================
    # 4) Dataset PyTorch
    #    - entrée : epsilon en % (100*eps)
    #    - sortie : sigma en GPa (= MPa/1000) pour stabiliser la loss
    # ============================================================
    x = torch.tensor(100 * eps_plot, dtype=torch.float32).view(-1, 1)       # ε en %
    y_mpa = torch.tensor(sigma_plot, dtype=torch.float32).view(-1, 1)       # σ en MPa
    y = y_mpa / 1000.0                                                     # σ en GPa (training)

    # ============================================================
    # 5) Réseau de neurones (MLP)
    # ============================================================
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        def forward(self, x_in):
            return self.net(x_in)

    model = MLP()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=3e-3)  # ok avec la sortie normalisée en GPa

    def predict_sigma_mpa():
        """
        Prédit sigma en MPa :
        - le réseau prédit en GPa (car entraînement en GPa)
        - on reconvertit en MPa en multipliant par 1000
        """
        model.eval()
        with torch.no_grad():
            pred_gpa = model(x)           # sortie réseau (GPa)
            pred_mpa = pred_gpa * 1000.0  # conversion en MPa
        return pred_mpa.numpy().flatten()

    # ============================================================
    # 6) Entraînement + sauvegarde des figures
    # ============================================================
    epochs = 5000
    plot_every = 500

    # Dossiers de sortie (modif proposée : tout mettre dans outputs/)
    out_dir = "outputs"
    frames_dir = os.path.join(out_dir, "frames_nn")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []

    for epoch in range(1, epochs + 1):
        model.train()

        pred = model(x)          # prédiction en GPa
        loss = loss_fn(pred, y)  # MSE en (GPa)^2

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Tous les plot_every epochs : figure + sauvegarde
        if epoch % plot_every == 0:
            pred_sigma = predict_sigma_mpa()

            plt.figure(figsize=(9, 5))
            plt.plot(eps_plot, sigma_plot, "k-o", linewidth=2, label="Contrainte réelle (MPa)")
            plt.plot(eps_plot, pred_sigma, "r--o", linewidth=2, label=f"NN - Epoch {epoch}")

            plt.xlabel("Déformation ε")
            plt.ylabel("Contrainte σ (MPa)")
            plt.title("Apprentissage NN simple (ε sinusoïdal monotone, 20 points)")
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
    # 7) Création du GIF (dans outputs/)
    # ============================================================
    gif_name = os.path.join(out_dir, "training_nn_elastoplastic_1d_sinus_monotone.gif")

    with imageio.get_writer(gif_name, mode="I", duration=3.0) as writer:
        for fp in frame_paths:
            img = imageio.imread(fp)
            writer.append_data(img)

        # répéter la dernière image pour finir le GIF proprement
        if len(frame_paths) > 0:
            last_img = imageio.imread(frame_paths[-1])
            writer.append_data(last_img)

    print(f"\nGIF généré : {gif_name}")

    # ============================================================
    # 8) Etat final : erreurs en MPa
    # ============================================================
    pred_sigma_final = predict_sigma_mpa()
    err = pred_sigma_final - sigma_plot

    print("\n--- Etat final ---")
    print(f"eps_p_final = {eps_p_hist[-1]:.6f}")
    print(f"RMSE (MPa)  = {np.sqrt(np.mean(err**2)):.3f}")
    print(f"MaxErr(MPa) = {np.max(np.abs(err)):.3f}")
    print(f"sigma_min   = {sigma_plot.min():.3f} MPa")
    print(f"sigma_max   = {sigma_plot.max():.3f} MPa")


if __name__ == "__main__":
    main()
