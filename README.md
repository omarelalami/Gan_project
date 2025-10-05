# StyleGAN3 — Déploiement Docker (GPU)

Application Gradio pour générer des visages avec **StyleGAN3** (checkpoint `.pkl`) et amélioration **GFPGAN** optionnelle.  
Tout est packagé dans un conteneur CUDA (NVIDIA).

---

## ✨ Fonctions

- Génération conditionnée par **Âge / Genre / Ethnie** (FairFace-like).
- UI **Gradio** (seed, ψ/truncation, noise mode, batch, variations).
- **GFPGAN** (optionnel) pour restaurer/améliorer les visages.
- Chemins et options **configurables via variables d’environnement**.

---

## 🧱 Prérequis

- GPU NVIDIA + drivers récents.
- **Docker** + **NVIDIA Container Toolkit** (`--gpus all`).
- Checkpoint StyleGAN3 (`.pkl`).  
  > Exemple : `network-snapshot-000000.pkl`
- (Optionnel) Poids GFPGAN `GFPGANv1.4.pth`.

---

## 📂 Arborescence conseillée

```text
stylegan3-app/
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
└─ models/
   ├─ network-snapshot.pkl        # ← votre checkpoint (obligatoire)
   └─ GFPGANv1.4.pth              # ← optionnel pour restauration visage
