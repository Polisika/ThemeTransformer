import os
from pathlib import Path
import subprocess

models = ["model_25min.ckpt", "model_35min.ckpt", "model_1hour.ckpt", "model_1.5hour.ckpt", "model_15000_epochs.ckpt", "model_2days.ckpt"]
# 214 epochs 25 min
# 300 epochs 35 min
# 514 epochs 1 hour
# 771 epochs 1.5 hour
# 15000 epochs == 29 hour
# 20290 epochs == 48 hours
tracks = ['theme_files_musegan/model_1200k_2_normalised_theme.mid',
 'theme_files_musegan/model_200k_2_normalised_theme.mid',
 'theme_files_musegan/model_1200k_3_normalised_theme.mid',
 'theme_files_musegan/model_100k_3_normalised_theme.mid',
 'theme_files_musegan/model_40k_3_normalised_theme.mid',
 'theme_files_musegan/model_5k_3_normalised_theme.mid',
 'theme_files_musegan/model_200k_3_normalised_theme.mid',
 'theme_files_musegan/model_40k_1_normalised_theme.mid',
 'theme_files_musegan/model_1200k_1_normalised_theme.mid',
 'theme_files_musegan/model_40k_2_normalised_theme.mid',
 'theme_files_musegan/model_5k_1_normalised_theme.mid',
 'theme_files_musegan/model_100k_2_normalised_theme.mid',
 'theme_files_musegan/model_5k_2_normalised_theme.mid',
 'theme_files_musegan/model_200k_1_normalised_theme.mid',
 'theme_files_musegan/model_100k_1_normalised_theme.mid']

print("Start research")
for model in models:
    dir_name = model.split(".")[0]
    Path(dir_name).mkdir(exist_ok=True)
    print("directory created")
    for track in tracks:
        model_name = model.split(".")[0]
        track_name = track.split("/")[-1].split(".")[0]
        print(f"{model} make track for {track}")
        if not Path(f"{dir_name}/{model_name}_{track_name}.mid").exists():
            subprocess.run(["venv/bin/python3", "inference.py", "--theme", track, "--model_path", model, "--seed", "20220529", "--cuda", "--out_midi", f"{dir_name}/{model_name}_{track_name}.mid"])