# scripts/plot_results.py
import json, glob, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = "workdir_vocab10000_lr0.001_embd256"
rows = []
for e in range(20):
    r = json.load(open(f"{d}/eval_results_epoch{e}.json"))
    rows.append((e, r["validation_loss"], r["bleu"]))

with open("eval_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "val_loss", "bleu"])
    w.writerows(rows)

ep = [r[0] for r in rows]
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(ep, [r[1] for r in rows], "o-", color="tab:blue")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(ep, [r[2] for r in rows], "s-", color="tab:orange")
ax2.set_ylabel("BLEU", color="tab:orange")
plt.title("De→En training curve (lr=0.001)")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
