"""report.py: LaTeX tables and macros with the results, for the progress report.

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: writes .tex files that report/progress_report.tex \\input's
READS     the results dicts built by train.py
WRITES    figures/results_table.tex     printed-font metrics per model
          figures/results_macros.tex    \\AnnValAcc, \\CnnValAcc, ... used in the report text
          figures/near_identical.tex    the merged groups, as a LaTeX list
          figures/hasy_table.tex        handwriting: zero-shot (+ fine-tuned) per model
          figures/confusions_table.tex  most frequent CNN confusions on val (+ ANN count)
          figures/tuning_table.tex      hyperparameter search results (tune.py)
          figures/hasy_confusions_table.tex   most frequent handwriting errors (val writers)
          figures/test_table.tex, test_macros.tex   final test results (final_test.py)
USED BY   training/train.py, training/tune.py, evaluation/final_test.py
RUN       not run directly
------------------------------------------------------------------------------
"""
import numpy as np

from plotting import NAME, short_config, top_runs
from symbols import NEAR_IDENTICAL


def tex(s):
    """Class label (e.g. \\varepsilon, \\{) as verbatim-looking LaTeX text."""
    esc = {"\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "#": "\\#", "$": "\\$",
           "%": "\\%", "&": "\\&", "_": "\\_", "^": "\\^{}", "~": "\\~{}"}
    return "\\texttt{" + "".join(esc.get(c, c) for c in s) + "}"


def results_table(runs, out, test):
    """LaTeX table, one row per model: val metrics (+ test metrics with --test)."""
    rows = []
    for name, r in runs.items():
        v = r["val"]
        row = (f"{NAME[name]} & {r['params'] / 1e3:.0f}k & {100 * v['clean']['acc']:.1f} & "
               f"{100 * v['clean']['acc_merged']:.1f} & "
               f"{100 * v['clean']['top5']:.1f} & {100 * v['clean']['macro_f1']:.1f} & "
               f"{100 * v['aug']['acc']:.1f}")
        if test:
            row += f" & {100 * r['test']['clean']['acc']:.1f} & {100 * r['test']['clean']['acc_merged']:.1f}"
        rows.append(row + " \\\\")
    head = ("Model & Params & Acc & Acc (merged) & Top-5 & Macro-F1 & Acc (aug.)"
            + (" & Test acc & Test (merged)" if test else ""))
    cols = "lrrrrrr" + ("rr" if test else "")
    (out / "results_table.tex").write_text(
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def results_macros(runs, out, test, device, hasy_info=None):
    """\\AnnValAcc, \\CnnValAcc, ... so the report text fills itself in."""
    L = []
    for name, r in runs.items():
        P = name.capitalize()
        v = r["val"]; pf = v["per_font_clean_acc"]
        lo, hi = min(pf, key=pf.get), max(pf, key=pf.get)
        vals = dict(Params=f"{r['params']:,}".replace(",", "{,}"), BestEpoch=r["best_epoch"],
                    TrainMin=f"{r['train_time_s'] / 60:.1f}",
                    TrainAcc=f"{100 * (1 - r['history']['train_err'][r['best_epoch'] - 1]):.1f}",
                    ValAcc=f"{100 * v['clean']['acc']:.1f}", ValTopFive=f"{100 * v['clean']['top5']:.1f}",
                    ValFOne=f"{100 * v['clean']['macro_f1']:.1f}", ValAugAcc=f"{100 * v['aug']['acc']:.1f}",
                    ValAccMerged=f"{100 * v['clean']['acc_merged']:.1f}",
                    ValFOneMerged=f"{100 * v['clean']['macro_f1_merged']:.1f}",
                    WorstFont=lo, WorstFontAcc=f"{100 * pf[lo]:.1f}",
                    BestFont=hi, BestFontAcc=f"{100 * pf[hi]:.1f}")
        if test:
            vals.update(TestAcc=f"{100 * r['test']['clean']['acc']:.1f}",
                        TestFOne=f"{100 * r['test']['clean']['macro_f1']:.1f}",
                        TestAugAcc=f"{100 * r['test']['aug']['acc']:.1f}",
                        TestAccMerged=f"{100 * r['test']['clean']['acc_merged']:.1f}")
        pc = lambda x: f"{100 * x:.1f}"
        if "hasy_zero" in r:
            hz = r["hasy_zero"]["val"]
            vals.update(HasyZeroAcc=pc(hz["acc"]), HasyZeroAccMerged=pc(hz["acc_merged"]),
                        HasyZeroTopFive=pc(hz["top5"]), HasyZeroFOne=pc(hz["macro_f1"]))
            if "test" in r["hasy_zero"]:
                vals.update(HasyZeroTestAcc=pc(r["hasy_zero"]["test"]["acc"]))
        if "hasy_ft" in r:
            ft = r["hasy_ft"]
            vals.update(HasyFtAcc=pc(ft["hasy_val"]["acc"]), HasyFtAccMerged=pc(ft["hasy_val"]["acc_merged"]),
                        HasyFtTopFive=pc(ft["hasy_val"]["top5"]), HasyFtBestEpoch=ft["best_epoch"],
                        HasyFtPrintedAcc=pc(ft["printed_val"]["clean"]["acc"]))
            if "hasy_test" in ft:
                vals.update(HasyFtTestAcc=pc(ft["hasy_test"]["acc"]))
        L += [f"\\newcommand{{\\{P}{k}}}{{{x}}}" for k, x in vals.items()]
    L.append(f"\\newcommand{{\\Device}}{{{device}}}")
    L.append(f"\\newcommand{{\\NearIdenticalGroups}}{{{len(NEAR_IDENTICAL)}}}")
    if hasy_info is not None:
        for s in ("train", "val", "test"):
            S = s.capitalize()
            L.append(f"\\newcommand{{\\HasySplit{S}Images}}{{{len(hasy_info[s]['raw']):,}}}".replace(",", "{,}"))
            if s != "train":
                L.append(f"\\newcommand{{\\HasySplit{S}Writers}}{{{len(np.unique(hasy_info[s]['user']))}}}")
    (out / "results_macros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def near_identical_tex(out):
    """The NEAR_IDENTICAL groups as one LaTeX line (for the report text)."""
    (out / "near_identical.tex").write_text(
        "; ".join("/".join(tex(s) for s in g) for g in NEAR_IDENTICAL) + "\n", encoding="utf-8")


def hasy_table(runs, out):
    """LaTeX table: handwriting accuracy zero-shot (+ after fine-tuning) per model."""
    ft = any("hasy_ft" in r for r in runs.values())
    rows = []
    for name, r in runs.items():
        z = r["hasy_zero"]["val"]
        row = f"{NAME[name]} & {100 * z['acc']:.1f} & {100 * z['acc_merged']:.1f} & {100 * z['top5']:.1f}"
        if ft:
            f = r.get("hasy_ft")
            row += (f" & {100 * f['hasy_val']['acc']:.1f} & {100 * f['hasy_val']['acc_merged']:.1f} & "
                    f"{100 * f['printed_val']['clean']['acc']:.1f}") if f else " & -- & -- & --"
        rows.append(row + " \\\\")
    head = "Model & Acc & Acc (merged) & Top-5" + (" & Acc & Acc (merged) & Printed acc" if ft else "")
    top = ("& \\multicolumn{3}{c}{Zero-shot}" + (" & \\multicolumn{3}{c}{Fine-tuned}" if ft else "")
           + " \\\\\n" + ("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n" if ft else "\\cmidrule(lr){2-4}\n"))
    (out / "hasy_table.tex").write_text(
        f"\\begin{{tabular}}{{l{'rrrrrr' if ft else 'rrr'}}}\n\\toprule\n{top}{head} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def confusions_table(info, preds, names, out, k=8):
    """LaTeX table: the k most frequent (true, predicted) CNN errors on val, with the ANN's count for the same pair."""
    y = info["y"]
    cnt = {}
    for t, p in zip(y, preds["cnn"]):
        if t != p:
            cnt[(t, p)] = cnt.get((t, p), 0) + 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:k]
    lines = []
    for (t, p), c in top:
        n_true = int((y == t).sum())
        ann = int(((y == t) & (preds["ann"] == p)).sum()) if "ann" in preds else None
        lines.append(f"{tex(names[t])} & {tex(names[p])} & {c}/{n_true}"
                     + (f" & {ann}/{n_true}" if ann is not None else "") + " \\\\")
    head = "True & Predicted & CNN" + (" & ANN" if "ann" in preds else "")
    (out / "confusions_table.tex").write_text(
        f"\\begin{{tabular}}{{ll{'rr' if 'ann' in preds else 'r'}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def tuning_table(grid, out):
    """LaTeX table of the hyperparameter search: the best TOP_K settings per model with their
    printed and handwritten validation accuracy; the selected setting (lowest mean error) in bold."""
    lines = []
    for name, runs in grid.items():
        top = top_runs(runs)
        for i, r in enumerate(top):
            # model name (with how many settings were tried) only on its first row
            model = f"{NAME[name]}, best {len(top)} of {len(runs)}" if i == 0 else ""
            # "--" for runs without a handwriting score (missing or NaN)
            hasy = r.get("hasy_val_err", float("nan"))
            accs = [f"{100 * (1 - r['best_val_err']):.1f}",
                    f"{100 * (1 - hasy):.1f}" if hasy == hasy else "--"]
            if i == 0:
                accs = [f"\\textbf{{{x}}}" for x in accs]
            lines.append(f"{model} & {short_config(r['config'])} & {r['params'] / 1e3:.0f}k & "
                         f"{r['best_epoch']} & " + " & ".join(accs) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]  # no rule after the last model
    (out / "tuning_table.tex").write_text(
        "\\begin{tabular}{llrrrr}\n\\toprule\nModel & Setting & Params & Best epoch & Val acc & "
        "Hand. val acc \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def test_table(res, out):
    """LaTeX table of the final test: printed test fonts and HASY test writers per model."""
    rows = []
    ft = any("ft" in r for r in res.values())
    for name, r in res.items():
        p, h = r["printed_test"]["clean"], r["hasy_test"]
        row = (f"{NAME[name]} & {100 * p['acc']:.1f} & {100 * p['acc_merged']:.1f} & "
               f"{100 * p['macro_f1']:.1f} & {100 * h['acc']:.1f} & {100 * h['acc_merged']:.1f}")
        if "ft" in r:
            row += f" & {100 * r['ft']['hasy_test']['acc']:.1f} & {100 * r['ft']['printed_test']['clean']['acc']:.1f}"
        elif ft:
            row += " & -- & --"
        rows.append(row + " \\\\")
    head = ("Model & Acc & Acc (merged) & Macro-F1 & Hand. acc & Hand. (merged)"
            + (" & Hand. acc (ft) & Printed acc (ft)" if ft else ""))
    (out / "test_table.tex").write_text(
        f"\\begin{{tabular}}{{l{'rrrrrrr' if ft else 'rrrrr'}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def test_macros(res, out):
    """\\CnnTestAcc, \\CnnTestHasyAcc, ... for the report text."""
    pc = lambda x: f"{100 * x:.1f}"
    L = []
    for name, r in res.items():
        P = name.capitalize(); p = r["printed_test"]
        vals = dict(TestAcc=pc(p["clean"]["acc"]), TestAccMerged=pc(p["clean"]["acc_merged"]),
                    TestFOne=pc(p["clean"]["macro_f1"]), TestAugAcc=pc(p["aug"]["acc"]),
                    TestHasyAcc=pc(r["hasy_test"]["acc"]), TestHasyAccMerged=pc(r["hasy_test"]["acc_merged"]))
        if "ft" in r:
            vals.update(TestHasyFtAcc=pc(r["ft"]["hasy_test"]["acc"]),
                        TestFtPrintedAcc=pc(r["ft"]["printed_test"]["clean"]["acc"]))
        L += [f"\\newcommand{{\\{P}{k}}}{{{x}}}" for k, x in vals.items()]
    (out / "test_macros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def hasy_confusions(info, pred, names, G, out, label, k=15):
    """Most frequent handwriting errors on the HASY val writers, for one model's predictions.
    `merged` marks pairs inside one near-identical group (ambiguous, not learnable from
    the image). Writes figures/hasy_confusions_table.tex; returns the summary dict."""
    y = info["y"]
    wrong = pred != y
    cnt = {}
    for t, p in zip(y[wrong], pred[wrong]):
        cnt[(t, p)] = cnt.get((t, p), 0) + 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:k]
    n_wrong = int(wrong.sum())
    in_group = int((wrong & (G[pred] == G[y])).sum())
    lines, rows = [], []
    for (t, p), c in top:
        n_true = int((y == t).sum())
        merged = bool(G[t] == G[p])
        rows.append(dict(true=names[t], pred=names[p], count=c, of=n_true, merged=merged))
        lines.append(f"{tex(names[t])} & {tex(names[p])} & {c}/{n_true} & "
                     + ("yes" if merged else "") + " \\\\")
    (out / "hasy_confusions_table.tex").write_text(
        f"% {label} on HASY val writers: {n_wrong} errors, {in_group} inside near-identical groups\n"
        "\\begin{tabular}{llrc}\n\\toprule\nTrue & Predicted & Count & Merged \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    return dict(model=label, errors=n_wrong, errors_in_groups=in_group,
                top_k_share=sum(r["count"] for r in rows) / max(n_wrong, 1), top=rows)
