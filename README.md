# StyleGAN3 — Déploiement Docker (GPU)
<img width="958" height="427" alt="Capture d’écran 2025-10-05 124010" src="https://github.com/user-attachments/assets/186019d4-263e-481f-817f-84685bafc5b2" />
<img width="242" height="364" alt="5" src="https://github.com/user-attachments/assets/d1dd8562-ffb2-45e9-87a1-777d683e81fd" />



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
   ├─ network-snapshot.pkl        
   └─ GFPGANv1.4.pth             


## 🚀 Lancement rapide

### Avec **docker-compose** (recommandé)
```bash
# 1) Construire l’image
docker compose build

# 2) Lancer le conteneur
docker compose up

