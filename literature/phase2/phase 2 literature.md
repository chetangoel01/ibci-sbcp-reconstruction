# HPC


---

## NYU HPC Access Guide (Torch)

### SSH → VS Code Remote‑SSH (Required for This Course)

This course uses **NYU HPC (Torch)** for neuroimaging workflows.
Before running any code, you must be able to log in via **SSH** and connect using **VS Code Remote‑SSH**.

---

###  Prerequisites

* NYU NetID with **HPC access approved**
* **VS Code** installed
* VS Code extension: **Remote – SSH** (by Microsoft)
* Mac / Linux terminal
  *(Windows users: WSL or PowerShell)*

---

## First Login: Plain SSH (MANDATORY)

> This step establishes your NYU identity.
> **Do not skip this** — VS Code will fail without it.

Open a **normal terminal** (not VS Code):

```bash
ssh NETID@login.torch.hpc.nyu.edu
```

You will see:

```
Authenticate with PIN XXXXXXXX at https://microsoft.com/devicelogin
```

1. Open the link in a browser
2. Enter the PIN
3. Approve the login

✅ Success:

```
[NETID@login.torch.hpc.nyu.edu ~]$
```

Exit:

```bash
exit
```

---

### Create SSH Shortcut (Strongly Recommended)

```bash
nano ~/.ssh/config
```

Add:

```sshconfig

# -------- Torch login ----------
Host torch
  HostName login.torch.hpc.nyu.edu
  User vsp7230
  ServerAliveInterval 60
  ServerAliveCountMax 10
  ForwardAgent yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR

# -------- Torch compute nodes (dynamic) ----------
Host gh* gl* cs*
  HostName %h.hpc.nyu.edu
  User vsp7230
  ProxyJump torch
  ConnectTimeout 120
  ServerAliveInterval 60
  ServerAliveCountMax 10

```

Test:

```bash
ssh torch
```

You may see the Microsoft code **one more time** — this is normal.

---

### Important Note About the Microsoft Code

* This is **not a password**
* It is NYU’s identity system
* Interrupting or opening multiple SSH attempts can cause repeated prompts

⚠️ **Finish login cleanly once before opening VS Code**

---

### VS Code: Remote‑SSH Connection

1. Open **VS Code**
2. `Cmd + Shift + P`
3. **Remote‑SSH: Connect to Host**
4. Select: `torch`
5. OS: **Linux**
6. Wait (first connection takes ~1 minute)

✅ Success:

* New VS Code window opens
* Bottom‑left shows: `SSH: torch`
* Terminal shows:

```
[NETID@login.torch.hpc.nyu.edu ~]$
```

---

### If VS Code Keeps Asking for the Code

Cause: VS Code attempted to connect **before** terminal login finished.

**Fix:**

1. Close VS Code completely
2. In normal terminal:

```bash
ssh torch
```

Finish login → `exit`
3. Reopen VS Code → Remote‑SSH again

❌ Do NOT retry multiple times inside VS Code

---

###Mental Model (Remember This)

> **Terminal proves who you are.**
> **VS Code reuses that authenticated connection.**

---
``` bash
sbatch   --account=torch_pr_60_tandon_priority   --job-name=cpu-dev   --nodes=1   --tasks-per-node=1   --cpus-per-task=8   --mem=32GB   --time=08:00:00   --output=/scratch/vsp7230/cpu-dev.%j.out   --wrap "hostname && sleep infinity"
```
```bash
squeue -u vsp7230
```
##You will then be able to see the node

##Then remote ssh and enter the node name
---

https://ood.torch.hpc.nyu.edu/
https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/torch/torch-new-user-guide

Imports
Unfortunately, you will face a lot of such errors when loading brain_data libraries: 

The only way around this would be to either use conda envs for separate tasks or load the github source code and modify code accordingly.

Depending on the type of analysis that you are doing, you will want to start your search from either the session or insertion endpoint.



*   A `session` refers to an *experimental session* and can contain multiple insertions.
*   An `insertion` is a *single Neuropixels recording* within a session.

For example, if you are interested in the correlation of activity across different brain areas within a single session, you would want to enter your search through the session endpoint. If, however, you are interested in doing an analysis across multiple Neuropixels recordings within a single brain region, the insertion endpoint should be your point of entry.

**Access:**
For this semester, Cloud Bursting access is provided only through Open OnDemand (OOD): https://ood-burst-001.hpc.nyu.edu/


NYU VPN is required when working off campus.

From this OOD server, students can:


* Launch compute nodes

* Run Jupyter notebooks

* Open terminal sessions

* Transfer some data from local computers

* Submit batch jobs directly from Jupyter notebook terminals

Students must use their assigned Slurm account for this course.


**Slurm Account and Resource Allocation**

Each student is assigned the following Slurm account:

**Account:** cs_gy_9223-2026sp

300 GPU hours (18,000 minutes)

Sufficient CPU time for coursework

Allowed partitions:

* interactive
* n2c48m24 —> CPU only
* g2-standard-12 —> 1 L4 GPU
* g2-standard-24 —> 2 L4 GPUs
* g2-standard-48 —> 4 L4 GPUs
* c12m85-a100-1 —> 1 A100 40GB GPU
* c24m170-a100-2 —> 2 A100 40GB GPUs
* n1s8-t4-1 —> 1 T4 GPU

**Spot Instance Policy (Important)**:

Cloud resources are running on Google Cloud spot instances, which may be preempted at any time.

More information:


* https://cloud.google.com/compute/docs/instances/spot

* https://cloud.google.com/compute/docs/instances/preemptible



Best practices:


* Enable checkpoint/restart for production runs

* Save checkpoints to your /scratch/$NETID directory

* Jobs will be automatically requeued if instances are shut down by GCP

Add the following directive to all Slurm scripts:

```
#SBATCH --requeue
```

**Conda Environments with Singularity and Overlay Files**:

Instructions for setting up Conda environments using Singularity and overlay files: https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/

Resources:

* Overlay file templates: /share/apps/overlay-fs-ext3

* Singularity OS images: /share/apps/images

There are useful instructions on SLURM jobs included in the tutorial: https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/#using-your-singularity-container-in-a-slurm-batch-job

You will call the sbatch job in the terminal with the command:

sbatch [YOUR_SBATCH_FILEPATH.SBATCH]
The terminal can be accessed here:

# neurolab_autoresearch

An autonomous ML experiment loop that runs, evaluates, and improves a model overnight — without you having to babysit it.

---

## The Problem It Solves

Training deep learning models requires trying hundreds of hyperparameter combinations. Normally you:
1. Change something in the code
2. Wait hours for training to finish
3. Check if it helped
4. Repeat

This is slow, tedious, and wastes your time sleeping while the GPU sits idle.

**This framework does all of that automatically.**

---

## What It Does

```
while True:
    1. Pick the next experiment from a queue
    2. Apply the change to train.py (e.g. change learning rate from 1e-4 to 3e-4)
    3. Train the model for a fixed time budget (e.g. 2 hours)
    4. Evaluate: fit a linear probe on frozen embeddings → get accuracy
    5. Did it improve?
         YES → save the change (git commit)
         NO  → undo the change (git checkout)
    6. Log the result to results.tsv
    7. If out of ideas → search arxiv + HuggingFace for relevant papers
                       → ask an LLM to propose the next experiment
    8. Repeat forever
```

You wake up in the morning with a full experiment history and a better model.

---

## Key Concepts

**`train.py`** — the only file the loop ever modifies. It trains your model and saves a checkpoint.

**`evaluate.py`** — locked, never touched. Loads the checkpoint, fits a linear classifier on frozen embeddings, returns a single accuracy number. This is the ground truth metric.

**`proposal_queue.json`** — a list of experiments to try next. You (or an LLM) fill this in. Each proposal is a description + a list of string replacements to apply to `train.py`.

**`results.tsv`** — append-only log of every experiment: what changed, what the accuracy was, keep or discard.

**Git as undo** — every kept experiment is a git commit. Discarded ones are reverted with `git checkout`. The git log IS the experiment history.

---

## Why Linear Probe?

Instead of fine-tuning the whole model, we freeze it and train a simple logistic regression on top. If the frozen representations are good, the probe accuracy will be high. This is a fast, unbiased way to measure representation quality without overfitting to the downstream task.

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set your data paths in train.py
DATA_DIR = "/path/to/train"
VAL_DIR  = "/path/to/val"

# 3. Add your first experiment to the queue
# (empty changes = baseline run)
echo '[{"description": "baseline", "changes": []}]' > proposal_queue.json

# 4. Start the loop
nohup ~/.local/bin/uv run python3 autoloop.py >> autoloop_stdout.log 2>&1 &

# 5. Watch it run
uv run python3 dashboard.py
```

---

## File Map

```
autoloop.py          ← the engine — never edit this
train.py             ← your training script — loop modifies this
evaluate.py          ← your metric — never edit this
proposal_queue.json  ← experiments waiting to run
results.tsv          ← full history of every run
dashboard.py         ← live terminal UI
program.md           ← describe your problem here (LLM reads this)
search_server.py     ← arxiv + HuggingFace paper search for LLM proposals
```

---

## LLM Backends (for automatic proposal generation)

When the queue runs out and the loop has been stuck for 3+ iterations, it searches recent papers and asks an LLM to suggest the next experiment:

| Backend | How to use |
|---------|-----------|
| Claude (default) | requires `claude` CLI installed |
| Ollama (local, free) | `AUTOLOOP_LLM=ollama` + `ollama pull qwen3:8b` |
| OpenAI | `AUTOLOOP_LLM=openai OPENAI_API_KEY=sk-...` |
| None | fill the queue manually |

---

## Built For

This framework was built for the **Alphabrain / LFP2Vec** project at NYU — self-supervised learning on brain signals (LFP) to classify 105 brain regions. But it works for any project where you have a `train.py` and a metric you want to maximize.


# Project 2
# CS-GY 9223 / CS-UY 3943 - Neuroinformatics, Spring 2026

## Project 2 Phase 2: Kinematic Decoding from Neural Activity

## Important Links & Deadline

```
Kaggle Competition: Join Competition (Kaggle Link)
Team Sign-Up Sheet: Sign Up Here (Google Sheets)
Deadline: March 29, 2026 at 11:59 PM EST (Sunday midnight)
Code Submission: Upload zipped code to Brightspace
```
```
Phase 2 of 2
This is Phase 2 of a two-phase project.
```
- **Phase 1** focused on predicting masked neural activity (self-supervised recon-
    struction).
- **Phase 2** focuses on **decoding finger movements from neural signals** (super-
    vised prediction of kinematics).
- Representations and insights learned in Phase 1 may carry over to Phase 2.

**Note:** You are allowed to use pre-built libraries (PyTorch, TensorFlow, scikit-learn, etc.) to
implement your solution. You are free to use any pretrained models or additional datasets
as well.

**Submission Instructions:**

- Submit a **final zipped code file** as your project submission on Brightspace.
- Use **Markdown cells** in notebook or neatly comment your code to include explanations
    wherever needed.
- You are allowed to use **LLMs**. If you do:
    **-** Include the **chat history** you had with the LLM along with your submission.
    **-** This helps us understand your **thinking style**.


- Participate in the **Kaggle competition** and submit your submission.csv file there.
- **Participation in Kaggle is mandatory, and your** **_private leaderboard_** **ranking will be**
    **used for final Project 2 Phase 2 scoring.**

**Competition At A Glance:**

**Item Details**

Training Data 187 sessions of 96-channel intracortical neural recordings with kinematics
Test Data 125 sessions (SBP with∼30% channels zeroed, kinematics withheld)
Evaluation R^2 on∼3.0M time bins× 2 position channels
Baseline Per-session channel mean (R^2 = 0. 0 )

**Quick Start:**

1. Sign up your team on the Team Sign-Up Sheet
2. Join the Kaggle competition and download the dataset
3. Explore the 187 training sessions (SBP, kinematics, trial info)
4. Train a decoder to predict finger positions from neural activity
5. Generate predictions for all∼3.0M time bins across 125 test sessions
6. Submit your submission.csv to Kaggle
7. Submit your code/notebook to Brightspace before the deadline


## Contents

- 1 Problem Statement
   - 1.1 Background
   - 1.2 Challenge Goal
   - 1.3 What You Will Learn
- 2 Data Description
   - 2.1 Dataset Overview
   - 2.2 Recording Setup
   - 2.3 Target Variables
   - 2.4 Decoding Approaches
   - 2.5 Channel Dropout
   - 2.6 Neural Drift
   - 2.7 The Task
   - 2.8 File Structure
   - 2.9 metadata.csv
   - 2.10 NumPy File Details
- 3 Evaluation Metrics
   - 3.1 R^2 (Coefficient of Determination)
   - 3.2 Final Scoring
   - 3.3 Public/Private Leaderboard Split
- 4 Baseline Results
   - 4.1 Key Takeaways
- 5 Submission Instructions
   - 5.1 Required Output
   - 5.2 Important Requirements
   - 5.3 Creating a Submission
   - 5.4 Loading Data
   - 5.5 Report Submission Deadline
- 6 Glossary


## 1 Problem Statement

### 1.1 Background

Intracortical brain–computer interfaces (iBCIs) record neural activity from electrode ar-
rays implanted in the brain. To restore motor control, these devices must **decode** neural
signals into movement commands in real time. A fundamental challenge is **neural drift**
— the statistical relationship between neural activity and behavior shifts over days, weeks,
and months due to electrode migration, tissue remodeling, and neural plasticity. A decoder
trained on one day’s data can fail on recordings from weeks later.

### 1.2 Challenge Goal

```
Can we decode finger kinematics from neural activity across sessions
spanning months to years of neural drift?
```
**Task:** Given long-term intracortical recordings spanning hundreds of sessions over multi-
ple years from a non-human primate performing a finger-pressing task, **predict 2 finger
position variables** (index and MRP finger positions) from 96-channel Spiking Band Power
(SBP) — with∼30% of channels randomly zeroed per session — at each time bin across 125
held-out test sessions.

### 1.3 What You Will Learn

- How neural activity encodes movement (neural tuning, population coding)
- Techniques for building neural decoders (Wiener filters, RNNs, transformers, etc.)
- How to handle distribution shift when training and test data are separated by months
    of neural drift

## 2 Data Description

### 2.1 Dataset Overview

The dataset consists of recordings from a 96-channel intracortical electrode array (two 64-
channel Utah arrays) implanted in the motor cortex of a non-human primate. During each
recording session, the subject performed a finger-movement task while neural activity and
finger kinematics were captured simultaneously.

```
Table 1: Dataset Summary
Split Sessions Channels Time Bins / Session Sampling Rate
Train 187 96 (∼68 active) ∼21K–35K 50 Hz
Test 125 96 (∼68 active) ∼21K–35K 50 Hz
```

### 2.2 Recording Setup

- **96 channels** from two 64-channel Utah arrays in motor cortex
- **Spiking Band Power (SBP):** power in the spiking frequency band, extracted at 50 Hz
    (20 ms bins). Shape: (Nbins, 96). ∼ **30% of channels are randomly zeroed per ses-**
    **sion** (simulating electrode degradation).
- **Finger Kinematics:** 4 variables — index position, MRP position, index velocity, MRP ve-
    locity. Positions normalized to [0, 1]; velocities are derived from positions via discrete
    differentiation (vel[t] = pos[t+1] - pos[t]). Shape: (Nbins, 4). **Only the 2 po-**
    **sition channels are evaluated.**
- Each session: ∼375 trials,∼21K–35K time bins (∼7–11 min of data)
- Recording span: ∼1,200+ days across 312 sessions

### 2.3 Target Variables

```
Table 2: Evaluated Position Channels
Column Description Range
indexpos Index finger position (0 = extended, 1 = flexed) [0, 1]
mrppos Middle-ring-pinky position (0 = extended, 1 = flexed) [0, 1]
```
**Note:** The kinematics files also contain velocity columns (indexvel, mrpvel), which are
simply the discrete time derivative of position: vel[t] = pos[t+1] - pos[t]. These are
provided for convenience but are **not evaluated**. You may use velocity internally (e.g., as
an auxiliary training target) if you wish.

### 2.4 Decoding Approaches

There are two natural strategies for predicting finger position:

1. **Direct position decoding:** Predict position directly from neural activity.
2. **Velocity-first decoding:** Decode velocity from neural activity, then integrate to re-
    cover position (since position =

#### R

```
velocity(t) dt, or in discrete form, pos = cumsum(vel)).
Velocity is zero-anchored while position is relative to an arbitrary reference point,
which can make it easier to decode. If you take this approach, calibrate your inte-
grated output so the final position values are on the correct scale.
```
### 2.5 Channel Dropout

In each session,∼30% of channels (28 out of 96) are randomly zeroed out (set to 0.0 for all
time bins). The zeroed channels **differ per session**. You can identify active channels by
checking which columns are all-zero:

sbp = np.load(”train/D042 ̇sbp.npy”) # (N, 96)
active = ̃(sbp == 0).all(axis=0) # True for active channels
print(f”Active: –active.sum() ̋/96”) # ̃68/


Your model must handle varying subsets of active channels across sessions.

### 2.6 Neural Drift

Mean SBP amplitude drops∼42% over the recording span. The train/test split is chronolog-
ical: training sessions span initial 60 % of data, test sessions span the rest. This ensures all
test data is in the “future” relative to training, making neural drift the central challenge.

### 2.7 The Task

For each of the 125 test sessions, you are given SBP neural activity (with∼30% of channels
randomly zeroed) but **kinematics are withheld**. Your goal is to predict the 2 finger position
variables at every time bin. You have 187 training sessions with paired SBP and kinematics
to learn the neural-to-position mapping, then must apply your decoder to the test sessions.


### 2.8 File Structure

```
Listing 1: Dataset File Structure
train/
–session ̇id ̋ ̇sbp.npy # (N, 96) ̃30% channels
zeroed
–session ̇id ̋ ̇kinematics.npy # (N, 4) kinematics,
float
–session ̇id ̋ ̇trial ̇info.npz # trial start/end bins
test/
–session ̇id ̋ ̇sbp.npy # (N, 96) ̃30% channels
zeroed
–session ̇id ̋ ̇trial ̇info.npz # trial start/end bins
metadata.csv # session ̇id, day, split, n ̇bins, n ̇trials
test ̇index.csv # sample ̇id, session ̇id, time ̇bin
sample ̇submission.csv # template for predictions
metric.py # official scoring function
```
### 2.9 metadata.csv

```
Table 3: Metadata Columns
Column Description
sessionid Anonymous session identifier (D001, D002,... ) in random order
day Relative recording day (day 0 = first session)
split train or test
nbins Number of time bins in the session (∼21K–35K at 50 Hz)
ntrials Number of trials in the session
```
### 2.10 NumPy File Details

**Training sessions** (train/):

- {sid}sbp.npy — Float32 (N, 96). SBP values with∼30% channels zeroed per session.
- {sid}kinematics.npy — Float32(N, 4). Columns: [indexpos, mrppos, indexvel,
    mrpvel]. Positions in [0, 1]; velocities derived via vel[t] = pos[t+1] - pos[t]. **Only**
    **positions (first 2 columns) are evaluated.**
- {sid}trialinfo.npz — Contains startbins, endbins (arrays of trial boundary in-
    dices), and ntrials.

**Test sessions** (test/):

- {sid}sbp.npy — Float32 (N, 96). SBP values with∼30% channels zeroed. **Kinematics**
    **are NOT provided.**
- {sid}trialinfo.npz — Same format as training.


## 3 Evaluation Metrics

Submissions are evaluated by comparing your predicted positions to the hidden ground
truth at all time bins across **125 test sessions** (∼3.0M total time bins).

### 3.1 R^2 (Coefficient of Determination)

R^2 measures how well your predictions capture the variance in the true positions. For ses-
sion s and position channel c:

```
R^2 (s, c) = 1 −
```
```
Xns
```
```
i=
```
#### 

```
ˆiy − yi
```
####  2

```
Xns
```
```
i=
```
#### 

```
yi− ̄y
```
####  2

#### (1)

where nsis the number of time bins in session s and ̄ is the mean of the true values fory
channel c in session s.

### 3.2 Final Scoring

The final leaderboard score is the average R^2 across all (session, channel) groups:

```
Final Score =
```
#### 1

#### 250

#### X^125

```
s=
```
#### X^2

```
c=
```
```
R^2 (s, c) (2)
```
This ensures all 125 sessions and both position channels contribute equally.
**Range:** (−∞, 1] where 1 indicates perfect prediction (higher is better). On the Kaggle
leaderboard, a **higher** R^2 corresponds to a **higher rank**.

```
Table 4: R^2 Interpretation
R^2 Meaning
= 1. 0 Perfect prediction
> 0. 0 Model captures meaningful position variance
= 0. 0 Equivalent to predicting the per-session channel mean
< 0. 0 Worse than the trivial baseline
```
### 3.3 Public/Private Leaderboard Split

```
Table 5: Leaderboard Configuration
Leaderboard Test Sessions Count
Public 64 sessions 51%
Private 61 sessions 49%
```

The public/private split uses **blockwise allocation** along the recording timeline — contigu-
ous groups of test sessions are assigned to public or private. The final ranking uses the
**private** portion.
**Goal:** Achieve R^2 well above 0.0, demonstrating that your decoder captures meaningful
kinematic structure beyond the trivial mean baseline.

## 4 Baseline Results

```
Table 6: Baseline Model Results (R^2 , with 30% channel dropout)
Model R^2
Session mean (trivial) 0.
Ridge Regression (Wiener filter, 10 lags) − 4. 119
LSTM (2-layer, hidden=128) 0.
GRU (2-layer, hidden=128) 0.
Transformer + SSL pretraining 0.
```
Figure 1: GRU baseline: predicted vs true finger positions on best, median, and worst test
sessions (4-second windows). The GRU captures the general movement shape on good ses-
sions but struggles with timing and amplitude on harder ones.


Figure 2: Transformer+SSL baseline: predicted vs true finger positions on best, median, and
worst test sessions (4-second windows). Self-supervised pretraining helps the Transformer
track finger movements more accurately, especially on best and median sessions.

### 4.1 Key Takeaways

- **The problem is solvable** : LSTM/GRU decoders achieve meaningful R^2 across sessions.
- **It is not trivially easy** : a naive Wiener filter (ridge regression) scores R^2 ≈ − 4 (much
    worse than predicting the mean).
- **Drift handling matters** : the key ingredient is per-session z-score normalization. Without
    it, cross-session models fail.
- **Channel dropout is challenging** : the ∼30% randomly zeroed channels require models
    that handle varying input subsets.
- **Room to improve** : the Transformer+SSL baseline achieves R^2 = 0. 45. Self-supervised
    pretraining and domain adaptation can push higher.

## 5 Submission Instructions

### 5.1 Required Output

Submit a CSV file with five columns and∼3.0M rows (one per test time bin):


```
Table 7: Submission File Format
Column Description
sampleid Integer index matching samplesubmission.csv
sessionid Which test session (e.g., D009)
timebin Time bin index into the session’s NumPy arrays
indexpos Predicted index finger position (float)
mrppos Predicted MRP finger position (float)
```
```
Example rows:
```
sample ̇id,session ̇id,time ̇bin,index ̇pos,mrp ̇pos
0,D009,0,0.5,0.
1,D009,1,0.5,0.

### 5.2 Important Requirements

1. **Column names** : Exactly sampleid,sessionid,timebin,indexpos,mrppos
2. **Row count** : Must match samplesubmission.csv (∼3.0M rows)
3. **Row ordering** : Your submission **must use the samesampleid ordering** assamplesubmission.csv.
    Kaggle matches predictions to ground truth using sampleid. The easiest approach is
    to load samplesubmission.csv, fill in your predictions, and save.
4. **No nulls** : All position values must be finite numbers (no NaN or Inf )
5. **Encoding** : UTF-8 CSV

### 5.3 Creating a Submission

```
Listing 2: Generating a Submission
```
import pandas as pd

sub = pd.read ̇csv(”sample ̇submission.csv”)
# sub columns: sample ̇id, session ̇id, time ̇bin,
# index ̇pos, mrp ̇pos

# Replace kinematic columns with your predictions
# ... your model logic here ...

sub.to ̇csv(”submission.csv”, index=False)

### 5.4 Loading Data


Listing 3: Loading Data Example
import numpy as np, pandas as pd

# Training session
sbp = np.load(”train/D042 ̇sbp.npy”) # (N, 96)
kin = np.load(”train/D042 ̇kinematics.npy”) # (N, 4)
trials = np.load(”train/D042 ̇trial ̇info.npz”) # start ̇bins, end ̇bins

# Test session (no kinematics!)
sbp ̇test = np.load(”test/D100 ̇sbp.npy”) # (N, 96)
trials ̇t = np.load(”test/D100 ̇trial ̇info.npz”)

# Metadata
meta = pd.read ̇csv(”metadata.csv”)
train ̇ids = meta[meta[”split”]==”train”][”session ̇id”].tolist()
test ̇ids = meta[meta[”split”]==”test”][”session ̇id”].tolist()

### 5.5 Report Submission Deadline

In addition to the Kaggle submission, you must submit a brief report describing the **meth-
ods and experiments** you performed.
**Deadline: Sunday, March 29, 2026 at 11:59 PM EST.**
The report may be provided either as:

- a PDF document, or
- documented Markdown / text cells within your notebook.

## 6 Glossary

```
Table 8: Terminology
Term Definition
iBCI Intracortical Brain–Computer Interface
SBP Spiking Band Power (neural feature extracted from electrode signals)
Utah Array Microelectrode array for intracortical recording
R^2 Coefficient of Determination (evaluation metric, higher is better)
Motor Cortex Brain region controlling voluntary movement
Neural Drift Change in neural signal statistics over time
MRP Middle-Ring-Pinky (finger group)
Time Bin 20 ms interval (50 Hz sampling)
Wiener Filter Linear decoder using time-lagged neural features
```
**Good luck!** We look forward to seeing innovative approaches to this challenging decoding
problem.


# related lecture 
# CS-GY 9223 /

# CS-UY 3943

# Neuroinformatics

## Prof. Erdem Varol

## Lecture 8: Neural Decoding


#### Today’s menu

**Announcements**

Neural Decoding

Break

LFP2Vec 2.0 Code Lab

Break

Presentations


### Project 2 Phase 2 is due Sunday March 29th

```
● We will assume you are continuing with the same team, if this is not the case
please reach out to the TAs
● As a reminder, 1-3 members per team (no more than 2 grad students per
team)
● Make sure to sign up with your NYU email on Kaggle
● Project 2 sign up sheet:
https://docs.google.com/spreadsheets/d/1FkuCEk2ETyqNtteWR_IoJg2DhGQnBsFzMuGByVw21O
0/edit?gid=0#gid=
```

### Project Details

```
● Data: Recordings from an electrode array implanted into the motor cortex
of a single non-human primate - Spiking Band Power (SBP) plus tabular
kinematics data
○ 312 sessions spanning 1,200 days
○ Each session is ∼375 trials, ∼21K–35K time bins (∼7–11 min of
data)
● Your goal for this phase: Predict 2 finger position variables (index and
MRP finger positions)
○ For each session s and channel c
```
```
○
● Interpolation vs. Extrapolation (Reminder)
○ Signal Drift: Mean amplitude drops ~42% over the recording
span, and test sessions can be up to 300 days from the training
session
○ Extrapolation: 30% of channels are randomly zeroed out per
session
```

### Example predictions


### Possible ways to approach kinematics data

**1. Direct position decoding:** Predict position directly from
neural activity.
**2. Velocity-first decoding:** Decode velocity from neural
activity, then integrate to re-cover position

```
● position = ∫velocity(t) dt
● discrete form, pos = ∑vel(t)Δt
● Velocity is zero-anchored while position is relative to
an arbitrary reference point, which can make it easier
to decode. If you take this approach, calibrate your
integrated output so the final position values are
on the correct scale.
```
Other physics-based approaches could be interesting to
explore too (e.g., KANs, but given your objective it’s also
important to consider how much information is encoded in a
transformer architecture vs. a more human-interpretable
MLP)

```
Acceleration data is not
provided, but included here
for broader context/physics
refresher
```
```
http://hyperphysics.phy-astr.gsu.edu/hbase/acons.html
```

### Project 2 Grading (20% of grade total)

```
● 10% for passing baseline of phase 1
● 30% competition grading for phase 1
● 10% for passing baseline of phase 2
● 30% competition grading for phase 2
● 20% for report (after phase 2)
```
_All curved to allow for B+ median as seen in Project 1_

```
We will keep the same curve, resulting in
~76% for median score (assuming full
marks on baseline and report), which will
then be curved up to a B+; phase 1 will
have a less steep curve to account
for closer ties in performance
```

##### Project 2 Phase 2 Leaderboard

```
Masked transformer is the baseline
that you need to beat to get the full 10%
that isn’t competition-related
```

### Presentations scheduled for today!

```
https://docs.google.com/spreadsheets/d/1BCIVOoo6inOxDZbB6jAlqIs8T8HhlyWWGdXqH
_RnDyY/edit?gid=0#gid=
```
```
If you are missing from
this list, please let us
know!
```

### Take advantage of office hours for this next

### assignment!

Prof. Varol scheduler:

https://calendar.app.google/3sw3oKtqYWh

e944x

```
TA office hours zoom link:
https://nyu.zoom.us/j/
```

#### Today’s menu

Announcements

**Neural Decoding**

Break

LFP2Vec 2.0 Code Lab

Break

Presentations


**Outline**

1. Neural coding problem
    ○ Encoding
    ○ Decoding
    ○ Self-prediction
2. Multi-animal reduced rank regression (RRR)
    ○ Linear decoding/encoding model (interpretable!)
**3. POYO-1: A Unified, Scalable Framework for Neural Population Decoding**
○ Transformer-based model for decoding.
4. Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution
    ○ Transformer-based model for self-prediction


**Outline**

1. Neural coding problem
    ○ Encoding
    ○ Decoding
    ○ Self-prediction
2. Multi-animal reduced rank regression (RRR)
    ○ Linear decoding/encoding model (interpretable!)
3. POYO-1: A Unified, Scalable Framework for Neural Population Decoding
    ○ Transformer-based model for decoding.
4. Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution
    ○ Transformer-based model for self-prediction


**Neural coding problem #1:**

How is information represented in the brain?

```
behaviors encoding^ neural signals^
```
```
Linear models:
Linear regression, reduced-rank regression
```
```
Nonlinear models:
Generalized linear models (GLMs), neural networks
```
```
P(neural signals|behaviors)
```

**Why study encoding?**

```
● Allows us to examine how behavior/tasks are represented in neural activity.
● Can rank importance of task variables for driving neural activity.
● Can potentially be used for write-in, e.g. retinal brain computer interfaces (BCIs).
```
```
Single-trial neural dynamics are dominated by movement
```
```
[Musall et al. 2019]
```

**Neural coding problem #2:**

What information is in the brain and where is it?

```
Linear models:
Logistic regression, linear regression, reduced-rank regression
```
```
Nonlinear models:
Neural networks, EIT, NDT 1/2/3, POYO-
```
neural signals (^) **decoding**
behaviors
P(behaviors|neural signals)


**Why study decoding?**

```
● Determine the amount of information about a behavior/task in the neural activity.
● Enable development of BCIs for restoring movement and communication.
● Clinical diagnostics (e.g., epilepsy, etc.).
```
[Willet et al. 2023]


[Willet et al. 2023]

**Why study decoding?**

```
● Determine the amount of information about a behavior/task in the neural activity.
● Enable development of BCIs for restoring movement and communication.
● Clinical diagnostics (e.g., epilepsy, etc.).
```

**Neural coding problem #3:**

What is the structure of the neural code?

```
neural signals Self-prediction neural signals
```
```
Linear models:
Linear regression, reduced-rank regression
```
```
Nonlinear models:
Neural networks, LFADS, NDT 1/2/3, Multi-task Masking (MtM)
```
```
P(neural signals)
```

**Why study self-prediction?**

```
● Allows you study how brain areas communicate to produce behavior + cognition.
● Allows for estimating low-dimensional neural dynamics that govern neural activity.
● Can study correlations of neural activity across different trials and animals.
```
```
[Gokcen et al. 2023]
```
```
[Safaie et al. 2023]
```

**Traditional pipeline for each of these tasks (single-session)**

1. Gather data with recording modality (extracellular recording, two photon imaging, etc.).
2. Perform preprocessing of the recording (spike sorting, quality controls).
3. Bin the data into small temporal windows (e.g., 20 ms).
4. Define trial length based on the experimental task (e.g., 2 seconds)
5. Divide trials into training, validation, and test trials.
6. Train your encoding, decoding, or self-prediction method.
7. Tune your method on validation and report test performance.
8. Profit?


**Challenges for single-session models**

1. Limited number of trials for training (overfits to noise/sensitive to parameters).
2. Low-number of neurons for training (overfits to specific neurons).
3. Ignores useful neural correlations across animals performing the same task.
4. Limited view of the brain (~10 brain regions sampled for Neuropixels).
5. Limited view of the behavioral task space (usually the animal is performing 1 task).


**Can we use more data to improve these techniques?**

```
The success of large language models
(LLMs) underscores the importance of
more data.
```
```
Dataset sizes are rapidly increasing in
neuroscience.
```
```
Developing techniques that can take
advantage of these large-scale datasets
will be essential.
```
```
Urai et al. 2021;
Stevenson and
Kording et al., 2011
```
```
Wei et al. 2022
```

```
24
```
**Stitching together activity from over one hundred animals**

```
International Brain Laboratory (IBL) Brainwide Map
```
```
699 insertions, 139 mice
```

**Foundation models across domains**

```
Large Language Models
(GPT, BERT, T5)
```
```
Large Vision Models
(CLIP, DALLE, SAM)
```
```
Foundation models for life sciences + medicine
```
```
Audio foundation models
(wav2vec2, whisper, Audiobox)
```

**What is a foundation model?**

```
2
```
**Pretrained at scale** : Deep neural
networks trained on massive,
heterogeneous datasets.

**Flexible adaptation** : Easily fine-tuned
for many downstream tasks.

**Efficient learning:** Reduce reliance on
large labeled datasets through
transfer learning.

```
https://blogs.nvidia.com/blog/what-are-foundation-models/
```

**What is a “neurofoundation” model?**

```
2
```
```
Azabou et al 2024
```

**Challenges for neurofoundation models**

```
Scientific
```
1. Each recording samples different brain regions and neural populations.
2. Different behavioral contexts lead to different neural activity.
3. Different species have different anatomical and functional brain structures.
4. What is the correct pretraining technique to understand the brain
    a. Next token prediction? Masking? Decoding? Encoding?

```
Practical
```
1. Compute is a bottleneck - need large GPUs for training large models.
2. LLMs are largely uninterpretable; Neuroscientists want interpretability.


**Outline**

1. Neural coding problem
    ○ Encoding
    ○ Decoding
    ○ Self-prediction
2. Multi-animal reduced rank regression (RRR)
    ○ Linear decoding/encoding model (interpretable!)
3. POYO-1: A Unified, Scalable Framework for Neural Population Decoding
    ○ Transformer-based model for decoding.
4. Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution
    ○ Transformer-based model for self-prediction


● Across sessions and animals, the neurons that are being recorded change.

● Single-session models: Separate models are fitted to the data from each individual session.

**Linear Regression**

```
Neural activity in a trial from session i
```
```
Task variable in a trial from session i
```
```
Session-specific parameters
```
```
Session-specific intercept
```
```
B^1
```
```
B^2
```
```
B^3
```
```
BI^
```

**Reduced Rank Regression (RRR)**

● Neural activity is correlated across animals performing the same task.^

● How can we take advantage of this with classic linear techniques?


● A linear multi-animal model that shares information across multiple sessions.

```
Neural activity in a trial from session i
```
```
Task variable in a trial from session i
```
```
Session-specific neural basis
```
```
Across-session temporal basis
```
```
Session-specific intercept
```
**Reduced Rank Regression (RRR)**


**Quantitative comparison**

● RRR outperforms baseline linear and shallow nonlinear decoders (MLP, LSTM).

● Consistent improvement for both discrete and continuous behaviors and across many different

```
brain regions.
```

**Behaviorally relevant neural dynamics**

● Embeddings from RRR are more informative than PCA embeddings.

● Exploiting multi-session correlations among animals improves the ability of the model to extract

```
these dynamics.
```
```
B^1
B^2
B^3
```
```
BI^
```

**Single-neuron selectivity**

● RRR can quantify the importance of each neuron for decoding choice across many brain areas.

● Ablating neurons identified as important by RRR leads to a more rapid decline in performance

```
compared to ablating unimportant neurons.
```

**Single-neuron selectivity**

● RRR can quantify the importance of each neuron for decoding choice across many brain areas.

● Ablating neurons identified as important by RRR leads to a more rapid decline in performance

```
compared to ablating unimportant neurons.
```

**Identify behaviorally relevant timescales**

● Visualize the temporal basis of the multi-animal, across-region RRR.

● Map behaviorally relevant timescales across the whole brain.

```
Actual peak activation
times in recordings
```
```
Actual activation
width
```
```
Improvement in
decoding
```

**Outline**

1. Neural coding problem
    ○ Encoding
    ○ Decoding
    ○ Self-prediction
2. Multi-animal reduced rank regression (RRR)
    ○ Linear decoding/encoding model (interpretable!)
3. POYO-1: A Unified, Scalable Framework for Neural Population Decoding
    ○ Transformer-based model for decoding.
4. Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution
    ○ Transformer-based model for self-prediction


**Transformers (a brief aside)**

```
[Neptune.ai]
```
```
● Powerful deep learning architecture used in most
modern LLMs.
```
```
● “Attention Is All You Need” - Vaswani et al. 2017.
```
```
● Scalable and high amount of parallelism unlike
recurrent architectures.
```
```
● Requires converting your data into discrete
“tokens” for processing (hard!).
```
```
● Won’t cover this in detail in this lecture.
```

###### Unit embeddings

```
● Unlike words in natural language, the “vocabulary” of neurons are different across different
animals ( tokenization is hard and still an open problem!).
● Can learn unit (“neuron”) embeddings for cross-animal alignment.
```

**Spike tokenization**

```
● Each spike is a token defined by the unit it came from, and its timestamp
● Allows us to process the activity of a population of neurons without any binning
```

**POYO-1 (Pretraining On manY neurOns)**

```
[Azabou et al. 2023]
```
```
● How to handle the long input sequence of spike-level tokens?
● POYO uses a PerceiverIO backbone ( latent tokenization ) to map a sequence of spikes to a
sequence of behavior outputs.
```
```
https://poyo-brain.github.io/ (cool interactive website)
```

**Perceiver IO architecture**

```
https://arxiv.org/pdf/2107.14795
```

**Benefit of pretraining**

```
● POYO pre-trained on 150 datasets outperforms single-session POYO.
● POYO demonstrates positive transfer across multiple tasks and individuals.
```

**Scaling laws for neural data**

● POYO is the first (along with NDT2) to show scaling with increasing model and data size.^


**Outline**

1. Neural coding problem
    ○ Encoding
    ○ Decoding
    ○ Self-prediction
2. Multi-animal reduced rank regression (RRR)
    ○ Linear decoding/encoding model (interpretable!)
3. POYO-1: A Unified, Scalable Framework for Neural Population Decoding
    ○ Transformer-based model for decoding.
4. Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution
    ○ Transformer-based model for self-prediction


**Whole-brain insights from multi-animal datasets**

● Recent efforts to build a brain-wide map of

```
neural activity across many animals.
```
● Fusing together datasets from many animals

```
and brain areas can enable whole-brain insights.
```
● What’s missing? Large-scale models that can

```
integrate data from many different sources.
```
```
Decoding, encoding, latent dynamics,
forecasting, cross-region interactions, etc.
```
```
699 insertions, 139 mice
```

**Current models for cross-animal, cross-region analyses are limited**

```
● POYO-1 (Azabou et al. 2023) is a supervised model for neural
decoding. No modeling of dynamics or brain regions.
```
```
● NDT2 (Ye et al. 2023) is a self-supervised model for self-
prediction but ignores inter- and intra-region interactions.
```
```
● mDLAG (Gokcen et al. 2023) models multi-area interactions
but can not be scaled to many animals and brain regions.
```
Gokcen et al. 2023^
**Our goal** : Build a self-supervised model that can predict activity at
different spatiotemporal scales for many animals and brain regions.


**Multi-task-masking (MtM) (especially relevant for the LFP project 2!!)**

A pretraining recipe for learning spatiotemporal dynamics


**Datasets**

● _IBL repeated site dataset_ - Neuropixels

```
recordings targeting the same brain locations
in mice performing a decision-making task.
```
● Trial-aligned, sorted data from 39 mice and

```
26,376 neurons (~676 neurons per session).
```
● Neural activity is binned into 20 ms windows,

```
with a fixed trial length of 2 seconds.
```
```
stimulus
```
```
choice
```
```
IBL et al. 2023
```

**Metrics**

```
Co-Smoothing : Predict the activity of a held-out neuron using all other
neurons (Macke et al. 2011, Ye et al. 2021).
```
```
Forward prediction : Predict future activity from past activity. We predict
the last 10% (200 ms) of the trial-aligned activity (2 seconds).
```
```
Intra-region co-smoothing: Predict the activity of a held-out neuron using
neurons in the same brain region.
```
```
Inter-region co-smoothing: Predict the activity of a held-out neuron using
neurons in other brain regions.
```
```
Choice decoding : Predict the choice the mouse makes using all
trial-aligned neural activity.
```
```
Motion energy decoding : Predict motion energy of the mouse's whiskers
using all trial-aligned neural activity.
```

**Single-session results for MtM vs. temporal masking**

Masking scheme ablation


**Pretraining MtM across many different animals + insertions**

● Session-specific matrices
(“stitchers”) map sessions with
different numbers of neurons
into a fixed dimension.

● Session-embeddings are also
used to distinguish sessions.

● “Stitchers” require many more
parameters than unit
embeddings = not great!

```
MtM
```
```
Input
Stitcher
1
```
```
Input
Stitcher
2
```
```
Input
Stitcher
N
```
```
Output
Stitcher
1
```
```
Output
Stitcher
2
```
```
Output
Stitcher
N
```
```
Data from
Session 1
```
```
Data from
Session 2
```
```
Data from
Session N
```
```
Session Embed
```
+

```
......
......
```

**Generalization of pretrained MtM to unseen animals**

```
● We pre-train MtM and the temporal masking baseline on 34 animals.
```
**● We fine-tune the final models on the training data from 5 unseen animals.**


**Scaling curves**

MtM shows improvement after pretraining on more animals


**Estimation of “functional connectivity” matrix (Region B -> Region A prediction)**

```
● Fine-tuning a pretrained MtM improves performance (outperforms single-session regression).
```

**Outline**

1. Neural coding problem
    ○ Encoding
    ○ Decoding
    ○ Self-prediction
2. Multi-animal reduced rank regression (RRR)
    ○ Linear decoding/encoding model (interpretable!)
3. POYO-1: A Unified, Scalable Framework for Neural Population Decoding
    ○ Transformer-based model for decoding.
4. Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution
    ○ Transformer-based model for self-prediction
5. Neural Encoding and Decoding at Scale (NEDS)
    ○ Transformer-based model for decoding and encoding.Transformer-based model for self-prediction
6. In vivo cell-type and brain region classification via multimodal contrastive learning ( **NEMO** )
    ○ Contrastive pre-training for cell-type and brain region classification


**Modeling the relationship between neural activity and behavior**

```
decoding
neural signals
behaviors
```
P(behaviors|neural signals)

**encoding**

P(neural signals|behaviors)

```
Methods were traditionally designed for modeling one of these conditional distributions
```

**Neural Encoding and Decoding at Scale (NEDS)**
Allows for simultaneous encoding and decoding through multimodal masking


```
We again trained NEDS on a subset of the
IBL repeated site dataset.
Trial-aligned, sorted data from 78 mice and
~27,000 neurons (73 mice used for training).
Neural activity is binned into 20 ms
windows, with a fixed trial length of 2
seconds.
We exclude trials based on reaction time
outliers for behavior decoding.
```
```
stimulus
```
```
choice
```
```
IBL et al. 2023
```
**Datasets (similar to MtM)**


**Multi-session pre-training improves performance on all tasks**


**Qualitative comparison of single vs. multi-session models**

Neural Encoding

Neural Decoding


**NEDS embeddings are region-selective without any training**

```
Latent embeddings of pretrained models can reveal region-level functional structure!
```

**Outline**

1. Neural coding problem
    ○ Encoding
    ○ Decoding
    ○ Self-prediction
2. Multi-animal reduced rank regression (RRR)
    ○ Linear decoding/encoding model (interpretable!)
3. POYO-1: A Unified, Scalable Framework for Neural Population Decoding
    ○ Transformer-based model for decoding.
4. Towards a "Universal Translator" for Neural Dynamics at Single-Cell, Single-Spike Resolution
    ○ Transformer-based model for self-prediction
5. Neural Encoding and Decoding at Scale (NEDS)
    ○ Transformer-based model for decoding and encoding.Transformer-based model for self-prediction
6. In vivo cell-type and brain region classification via multimodal contrastive learning ( **NEMO** )
    ○ Contrastive pre-training for cell-type and brain region classification


**Can we use foundation modeling to understand cell-type diversity?**

```
The brain has a staggering amount
of cell-type diversity.
```
```
Characterizing cell-types is
essential for understanding brain
development, microcircuit function,
and brain disorders.
```
```
Can we use IBL data + foundation
modeling to quantify cell-type
diversity across the mouse brain?
```
```
Vizgen MERFISH Mouse Brain Receptor Map
```

Question: How can we learn about the identity of individual neurons using these morphoelectrical features?

**How can we characterize cell-types in extracellular recordings?**

```
Beau et al. 2025
```

**Contrastive alignment of waveforms and neural activity**

**Neuronal Embeddings via MultimOdal Contrastive Learning**

```
67
```
```
Input : Extracellular action potentials (‘templates’) and
neural activity (ACG images) for individual neurons.
```
```
Output : Neuron-specific embeddings that are
informative about cell-type and brain region.
```
```
Trained in a fully self-supervised way (no labels required)!
6
```

**NEMO significantly outperforms current methods across 2 datasets**

```
Beau et al. 2025 Lee et al. 2024
```
```
Beau et al. 2025 Lee et al. 2024
```

## NEMO learns cell-type selective embeddings without any labels

###### Visual cortex

###### Cerebellum

```
Multiple modalities
improves cell-type
selectivity
```

```
what you observe
from the recording
```
```
the probe location we
would like to infer
```
```
70
```
**Brain region classification**

###### ?

```
7
```

## Pretraining NEMO on the Brain Wide Map


### Region decoding by ensembling predictions of nearby

### neurons


```
Outperforms purely supervised method with less labels
```
```
Linear MLP
```
## NEMO’s performance scales with the number of labeled data


### NEMO representations cluster into meaningful clusters with a graph-based method


## The cluster distributions are closely correlated with the anatomical regions


**How can a foundation model help us understand the brain?**

```
● Infer neural dynamics in missing brain regions for simultaneous recordings.
```
```
● Analyze intrinsic and behaviorally relevant brain-wide timescales (e.g. interpret attention, time lags).
```
```
● Identify subnetworks of brain regions by predicting the activity of one region from another.
```
```
● Cross-species transfer: Pre-train on monkey recordings to mitigate sparsity in human data.
```
```
● Unify representational space: Training across multiple animals performing different tasks under diverse
conditions allows us to compare what is shared among animals, and what is individual.
```
```
● Simultaneously translate across neural activity, behavior, and other modalities.
```
```
● Explore the potential for zero-shot learning and approach the theoretical limits of how much information
can be read out from the brain.
```
```
● Emergent capabilities? What can the model learn to do without any specific training?
```
```
● What about LFP? More on that next...
```

#### Today’s menu

Announcements

Neural Decoding

**Break**

LFP2Vec 2.0 (Subhrajit, Lawrence, Sirish) and Code Lab

Break

Presentations


#### Today’s menu

Announcements

Neural Decoding

Break

**LFP2Vec 2.0 Code Lab**

Break

Presentations


#### Today’s menu

Announcements

Neural Decoding

Break

LFP2Vec 2.0 Code Lab

Break

**Presentations**


### Presentations scheduled for today!

```
https://docs.google.com/spreadsheets/d/1BCIVOoo6inOxDZbB6jAlqIs8T8HhlyWWGdXqH
_RnDyY/edit?gid=0#gid=0
```
```
If you are missing from
this list, please let us
know!
```

