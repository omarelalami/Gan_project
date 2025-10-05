import os, io, random, zipfile, tempfile
import numpy as np
from PIL import Image
import torch
import gradio as gr

# === Import API StyleGAN3 (repo officiel) ===
import sys
sys.path.append("/opt/stylegan3")  # ajouté: le repo est cloné ici dans l'image
import legacy
import dnnlib

# -------------------- CONFIG --------------------
NETWORK = os.getenv("NETWORK", "/models/network-snapshot.pkl")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

AGE = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"}
GENDER = {0: "Female", 1: "Male"}
RACE = {0: "Black", 1: "EastAsian", 2: "Indian", 3: "Latino_Hispanic", 4: "MiddleEastern", 5: "SoutheastAsian", 6: "White"}

# === (OPTIONNEL) GFPGAN pour améliorer les visages ===
USE_GFPGAN = False
restorer = None
GFPGAN_MODEL = os.getenv("GFPGAN_MODEL", "/models/GFPGANv1.4.pth")
if os.getenv("ENABLE_GFPGAN", "1") == "1":
    try:
        from gfpgan import GFPGANer
        if os.path.isfile(GFPGAN_MODEL):
            restorer = GFPGANer(model_path=GFPGAN_MODEL, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
            USE_GFPGAN = True
        else:
            print(f"⚠️ Fichier GFPGAN non trouvé à {GFPGAN_MODEL} — amélioration visage désactivée")
    except ImportError:
        print("⚠️ GFPGAN non installé — amélioration visage désactivée")

# ----------------- CHARGEMENT RÉSEAU -----------------
assert os.path.isfile(NETWORK), f"Checkpoint introuvable: {NETWORK}"
with open(NETWORK, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)
G.eval()

# ----------------- UTILS -------------------
def make_z(seeds, z_dim):
    return np.stack([np.random.RandomState(s).randn(z_dim) for s in seeds])

def labels_from_names(age_name, gender_name, race_name):
    age = list(AGE.keys())[list(AGE.values()).index(age_name)]
    gender = list(GENDER.keys())[list(GENDER.values()).index(gender_name)]
    race = list(RACE.keys())[list(RACE.values()).index(race_name)]
    return np.array([[age, gender, race]], dtype=np.int64)

def synth_images(z, c, truncation_psi=1.0, noise_mode='const'):
    z = torch.from_numpy(z).to(DEVICE)
    c = torch.from_numpy(c).to(DEVICE)
    img = G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    imgs = [Image.fromarray(im, 'RGB') for im in img]
    if USE_GFPGAN and restorer is not None:
        enhanced = []
        for im in imgs:
            _, _, restored = restorer.enhance(np.array(im), has_aligned=False, only_center_face=False, paste_back=True)
            enhanced.append(Image.fromarray(restored))
        imgs = enhanced
    return imgs

def pil_to_zip(images):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, im in enumerate(images):
            bio = io.BytesIO()
            im.save(bio, format='PNG')
            zf.writestr(f"image_{i:03d}.png", bio.getvalue())
    tmp.flush()
    return tmp.name

def save_single_image(image):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(tmp, format="PNG")
    tmp.flush()
    return tmp.name

# ----------------- CALLBACKS -------------------
def cb_generate(seed, trunc, noise_mode, age_name, gender_name, race_name, n_images):
    seeds = [int(seed)] if n_images == 1 else [random.randint(0, 2**31-1) for _ in range(n_images)]
    z = make_z(seeds, G.z_dim)
    c = labels_from_names(age_name, gender_name, race_name)
    imgs = synth_images(z, c.repeat(len(seeds), axis=0), truncation_psi=trunc, noise_mode=noise_mode)
    if n_images > 1:
        return None, imgs, pil_to_zip(imgs)
    return save_single_image(imgs[0]), None, None

def cb_random_seed():
    return random.randint(0, 2**31-1)

def cb_surprise(age, gender, race, trunc, noise_mode, n_images):
    seeds = [random.randint(0, 2**31-1) for _ in range(n_images)]
    z = make_z(seeds, G.z_dim)
    c = labels_from_names(age, gender, race).repeat(len(seeds), axis=0)
    imgs = synth_images(z, c, trunc, noise_mode)
    return None, imgs, pil_to_zip(imgs)

def cb_variations(seed, trunc, noise_mode, age, gender, race, n_variations=4):
    base = np.random.RandomState(seed).randn(G.z_dim)
    zs = [base + 0.05 * np.random.randn(G.z_dim) for _ in range(n_variations)]
    c = labels_from_names(age, gender, race).repeat(n_variations, axis=0)
    imgs = synth_images(np.stack(zs), c, trunc, noise_mode)
    return None, imgs, pil_to_zip(imgs)

# ---------------------- UI ---------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🎨 StyleGAN3 — FairFace (déploiement amélioré)")
    gr.Markdown(
        "Interface pour générer des visages conditionnés par âge / genre / ethnie.\n"
        "_⚠️ Les données d’entraînement peuvent contenir des biais._"
    )

    with gr.Tab("Générer"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Number(value=0, label="Seed (z)")
                seed_btn = gr.Button("🎲 Seed aléatoire")
                trunc = gr.Slider(0.5, 1.2, value=1.0, step=0.05, label="Diversité / Réalisme (ψ)")
                noise_mode = gr.Dropdown(["const", "random", "none"], value="const", label="Noise mode")
                age = gr.Dropdown(choices=list(AGE.values()), value="20-29", label="Âge")
                gender = gr.Dropdown(choices=list(GENDER.values()), value="Male", label="Genre")
                race = gr.Dropdown(choices=list(RACE.values()), value="White", label="Race")
                n_images = gr.Slider(1, 16, value=1, step=1, label="Nombre d’images (batch)")
                gen_btn = gr.Button("🚀 Générer")
                surprise_btn = gr.Button("✨ Surprise ! (batch aléatoire)")
                var_btn = gr.Button("♻️ Variations (proches du seed courant)")
            with gr.Column(scale=2):
                out_single = gr.Image(type="filepath", label="Aperçu (si 1 image)")
                out_gallery = gr.Gallery(label="Galerie (batch)", columns=4, height="auto")
                out_file = gr.File(label="Téléchargement (PNG ou ZIP)")

        gen_btn.click(cb_generate, [seed, trunc, noise_mode, age, gender, race, n_images],
                      [out_single, out_gallery, out_file])
        seed_btn.click(cb_random_seed, outputs=seed)
        surprise_btn.click(cb_surprise, [age, gender, race, trunc, noise_mode, n_images],
                           [out_single, out_gallery, out_file])
        var_btn.click(cb_variations, [seed, trunc, noise_mode, age, gender, race],
                      [out_single, out_gallery, out_file])

PORT = int(os.getenv("PORT", "7862"))
demo.launch(server_name="0.0.0.0", server_port=PORT)
