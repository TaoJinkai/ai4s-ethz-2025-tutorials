# AI in the Sciences & Engineering

Personal repository for tracking my **Colab notebooks** while following ETH Zürich CAMLab’s course **401-4656-21L — AI in the Sciences and Engineering (Fall 2025)**.

> This repo intentionally contains **tutorial notebooks only** (no `src/` and no lecture notebooks).  
> Each notebook is Colab-friendly and starts with a one-click “Open in Colab” badge.


## Open in Colab

Each notebook includes a badge at the top. After pushing this repo to GitHub, the badge will work once you replace `USERNAME/REPO` with your GitHub path.

Example badge (already embedded in notebooks):
```markdown
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/tutorials/<file>.ipynb)

Or open directly with the URL pattern:

https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/tutorials/<file>.ipynb

Tip: after your first push, search & replace USERNAME/REPO across the repo to set your path once.


Tutorials mapping
	•	2025-09-22 — Function Approximation with PyTorch
notebooks/tutorials/2025-09-22_function_approx.ipynb
	•	2025-09-25 — Cross-Validation & Intro to CNNs
notebooks/tutorials/2025-09-25_cv_cnn.ipynb
	•	2025-10-06 — PINN Training
notebooks/tutorials/2025-10-06_pinn_training.ipynb
	•	2025-10-13 — Fourier Neural Operator (FNO)
notebooks/tutorials/2025-10-13_fno.ipynb
	•	2025-10-27 — Convolutional Neural Operator (CNO)
notebooks/tutorials/2025-10-27_cno.ipynb

Add more tutorials by following the same naming convention:
YYYY-MM-DD_topic.ipynb under notebooks/tutorials/.


How to use

A) Colab (recommended)
	1.	Create a public GitHub repository and push this folder structure.
	2.	Click the Open in Colab badge at the top of any notebook, or use the URL pattern above.

B) Local (optional)

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook    # or: jupyter lab



Notebook template

Each tutorial notebook starts with:
	•	A Colab badge (update USERNAME/REPO once your repo is live);
	•	A small Setup cell (install deps on Colab if needed);
	•	A Work area section to outline objectives, key equations, experiments, and takeaways.

Keep cells short and composable; prefer saving large figures or artifacts outside of git (they’re ignored by .gitignore).


Versioning & hygiene
	•	Don’t commit large data or model files; keep them on external storage or releases.
	•	Commit early and often; write informative messages.
	•	If you need to share outputs, place images in a /docs folder or attach to issues/PRs.



License

MIT — see LICENSE￼.
Replace the placeholder copyright holder with your name.


Acknowledgements

This repository is a personal learning log inspired by the ETH Zürich CAMLab course AI in the Sciences and Engineering (401-4656-21L, Fall 2025).
This project is not affiliated with or endorsed by ETH Zürich; all materials here are my own notes and code.
