# Data

We use **AG News** (news articles sorted into 4 topic labels). The course handout also mentioned SST-2 and MRPC; we picked AG News.

**Why this folder is almost empty:** we do **not** store the dataset inside the Git repo. The training code calls Hugging Face `datasets` with the name `ag_news`. The **first time** you run training, that library **downloads** AG News from the internet and keeps it in a **cache on your computer** (not in this `data/` folder). So the `data/` folder is really just this explanation file.

**Where the data actually lives after download:** usually under your user folder in something like `~/.cache/huggingface/datasets/` (exact path can vary). You don’t need to open that by hand; the code finds it.

**What the code does with it:** it uses the **train** split to learn and the **test** split for the accuracy / F1 we save in `results/`.
