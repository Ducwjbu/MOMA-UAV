"""
build_bundle.py
===============
Dong goi code can de train thanh kaggle/moma_uav_code.zip, de upload len Kaggle Dataset.
Chay tu thu muc goc cua repo:
  py kaggle/build_bundle.py
"""

import os
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "kaggle", "moma_uav_code.zip")
FILES = [
    "agents_moma_ac.py", "environment.py", "train_moma_ac.py", "utils.py", "compare_runs.py",
    "config/environment/default.yaml",
    "config/train/moma_ac_random.yaml",
    "config/train/moma_ac_gpi_ls.yaml",
    "config/train/moma_ac_gpi_pd.yaml",
    "kaggle/run_kaggle.py",
]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        z.write(os.path.join(REPO, name), arcname=name)
print(f"Built {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB, {len(FILES)} files)")
