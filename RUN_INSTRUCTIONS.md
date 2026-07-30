# How to run this analysis — instructions

You are running a statistical analysis of PISA 2018 education data. You do not
need to understand the statistics. You need to run seven notebook cells in
order and report what they print.

**Time:** about 1 hour of attention, then a long unattended run (up to 2 days).

---

## Before you start

You need a Google account with **at least 4 GB free in Google Drive**.

If Colab Pro is available, use it — the free tier disconnects often, which
makes the long run take several days instead of one.

---

## Step 1 — Open the notebook

Go to <https://github.com/yazanjer/An_Explainable_AI_Education> and click the
**"Open In Colab"** badge next to `RUN_ALL.ipynb`.

## Step 2 — Set the runtime

In Colab: **Runtime → Change runtime type**

* Hardware accelerator: **T4 GPU**
* Runtime shape: **High-RAM**

This matters. Step 4 reads a 1.8 GB file and will crash without High-RAM.

## Step 3 — Run cells 1 to 5, one at a time

Click a cell, press **Shift+Enter**, wait for it to finish, then move to the
next. Do not run them all at once.

| Cell | What it does | Time | You should see at the end |
|---|---|---|---|
| 1 | Downloads the code, installs software | 5 min | `Cell 1 OK - continue to Cell 2.` |

**Cell 1 prints a line like `Commit : d2ad12e ...`.** If Dr. Yazan asks which
version you are running, that is the answer. **Always run Cell 1 after he tells
you he has pushed an update** — it is what pulls the new code.
| 2 | Checks the software works | 2 min | `163 passed` (the number may differ slightly; **it must say "passed" and not "failed"**) |
| 3 | Downloads the PISA data (500 MB) | 20 min | `Data ready: ...CY07_MSU_STU_QQQ.sav` |
| 4 | Checks the data is readable | 1 min | `Cell 4 OK - ready to run.` |
| 5 | Short test run | 15 min | a table of numbers |

**If Cell 1 says "RESTART THE RUNTIME NOW":** go to Runtime → Restart session,
then run Cell 1 again. This is normal and happens at most twice.

## Step 4 — Check Cell 5 against these numbers

Cell 5 prints performance figures. They should be close to:

| Task | Expected |
|---|---|
| Low vs. High | about **0.88** |
| Low vs. Medium | about **0.74** |
| Medium vs. High | about **0.69** |

**Anything near 1.00 is wrong — stop and report it.** These numbers are
supposed to be moderate.

**Send Dr. Yazan a screenshot of Cell 5's output and wait for approval before
Step 5.**

## Step 5 — The long run

1. In **Cell 1**, change `RUN_FULL = False` to `RUN_FULL = True`
2. Re-run Cell 1
3. Run **Cell 6**

This takes **20 to 40 hours**. Colab will disconnect — that is expected and
nothing is lost. When it disconnects:

* re-run Cell 1, then Cell 6
* it continues from where it stopped

Check once or twice a day. Keep the browser tab open while it runs.

## Step 6 — Finish

When Cell 6 prints `done in ... min`, run **Cell 7**.

Then send Dr. Yazan:

1. A screenshot of Cell 7's output
2. The folder `MyDrive/An_Explainable_AI_Education/results/tables/`
   (right-click → Download)

---

## If something goes wrong

**If Dr. Yazan says results looked wrong and he has pushed a fix**, delete the
saved progress before re-running, so old results are not reused:

```python
!rm -rf /content/drive/MyDrive/An_Explainable_AI_Education/results/checkpoints/*
```

Then re-run Cell 1 and Cell 5.

**First, try this** — it fixes most problems:

In Cell 1, set `FRESH_CLONE = True`, run Cell 1, then set it back to `False`.
Your downloaded data is kept; only the code is refreshed.

| Message | What to do |
|---|---|
| `credential propagation was unsuccessful` | Runtime → Restart session, run Cell 1 again |
| `RESTART THE RUNTIME NOW` | Do exactly that, then re-run Cell 1 |
| `LeakageError` | **Not a bug** — a safety check doing its job. Screenshot it and report. |
| Kernel crashed / out of memory | Check the runtime is set to **High-RAM** (Step 2) |
| `numpy.dtype size changed` | Runtime → Restart session, re-run Cell 1 |
| Anything in Cell 2 says `failed` | **Stop.** Screenshot and report. Do not continue. |
| Disk full | Delete `data/raw/SPSS_STU_QQQ.zip` from Drive — it is no longer needed |
| Cell 1 warns `working tree differs` or `Drive may be serving a cached copy` | Set `FORCE_REMOUNT = True` in Cell 1, run it, then set back to `False` |
| A fix was pushed but nothing changed | Re-run **Cell 1** (not just the failing cell), then check the `Commit :` line |

**Do not** edit any file other than the two switches in Cell 1
(`RUN_FULL`, `FRESH_CLONE`, `FORCE_REMOUNT`).

**Do not** upload the PISA data anywhere or share it. It is licensed to be
downloaded from the OECD, not redistributed.

---

## What to report

Send a short message at each of these points:

1. After Cell 5 — screenshot of the numbers, **wait for approval**
2. When the long run starts — "started, [date/time]"
3. If it fails and re-running does not fix it — screenshot of the red error
4. When it finishes — screenshot of Cell 7 plus the `results/tables/` folder

Questions are welcome at any point. Guessing is not — if a message is
unfamiliar, screenshot it and ask.
