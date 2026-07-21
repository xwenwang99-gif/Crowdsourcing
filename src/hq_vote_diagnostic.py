# -*- coding: utf-8 -*-
"""
HQ-vote diagnostic.

For each PREDICTED task group g, look at every vote cast by g's predicted-HQ
workers on the tasks assigned to g, and report how those votes distribute
across the C label classes. Row g of the returned matrix is

    vote_frac[g, c] = P(an HQ vote in predicted group g equals c).

This is exactly the input to the per-group majority vote in _mc_infer: the
label the HQ pool of group g elects is argmax_c counts[g, c]. The winning
segment is the majority-vote outcome for that group; a group whose largest
segment is not the intended label is one the aggregation gets wrong.

Because the bars are PREDICTED groups, the plot does not know the "correct"
class per bar a priori -- predicted group g is not guaranteed to correspond to
true label g once clustering is imperfect. Each bar is therefore annotated with
the majority (winning) class and its share, and, when y_true is supplied, with
the modal TRUE label among the group's tasks so you can see whether the vote
winner matches what the group actually contains.

HQ workers are defined per predicted group g, and a task i in predicted group g
is voted on by the HQ pool of g -- so the vote set for group g is assembled by
walking tasks, not by assuming a global HQ set.

Public API
----------
compute_hq_vote_distribution(rating, pred_group, hq_workers_pred, n_classes,
                             y_true=None) -> dict
plot_hq_vote_distribution(dist, path=None, draw=True, title=None) -> path|None
hq_vote_report(..., out_dir, method_name, y_true=None, draw=True) -> dict
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # file backend; safe on headless servers
import matplotlib.pyplot as plt


def _pred_group_1d(pred_group, n_task):
    """
    Normalise pred_group to a length-n_task int vector.

    Accepts (n_task,), (n_task, 1), or (n_task, 2) -- the last is the
    (task_id, group) form returned by _mc_infer_top2, whose group is column 1.
    """
    pg = np.asarray(pred_group)
    if pg.ndim == 1:
        out = pg
    elif pg.ndim == 2 and pg.shape[1] == 1:
        out = pg[:, 0]
    elif pg.ndim == 2 and pg.shape[1] == 2:
        out = pg[:, 1]
    else:
        raise ValueError(f"pred_group has unexpected shape {pg.shape}.")
    if out.shape[0] != n_task:
        raise ValueError(
            f"pred_group length {out.shape[0]} != n_task {n_task}."
        )
    return out.astype(int)


def compute_hq_vote_distribution(
    rating,
    pred_group,
    hq_workers_pred,
    n_classes,
    y_true=None,
):
    """
    Parameters
    ----------
    rating : (n_record, 3) array   columns [task, worker, label]
    pred_group : (n_task,) or (n_task, 2) int   predicted task group per task
    hq_workers_pred : list of length n_groups
        hq_workers_pred[g] = array of worker indices tiered HQ for predicted
        group g (exactly the object returned by _hq_and_label_infer, or
        [np.where(V[:, g] == 1)[0] for g in range(G)] on the likelihood side).
    n_classes : int
    y_true : (n_task,) int, optional
        True label per task. Used only to mark each predicted group's true
        label via maximum task overlap; not needed to build the vote counts.

    Returns
    -------
    dict, indexed by PREDICTED group g (rows), vote class c (cols)
        counts        : (n_groups, C) int   HQ votes = c in predicted group g
        vote_frac     : (n_groups, C) float row-normalised counts
        n_votes       : (n_groups,) int      HQ votes available per group
        n_tasks       : (n_groups,) int      tasks assigned to each group
        n_hq          : (n_groups,) int      # predicted-HQ workers in each group
        mv_label      : (n_groups,) int      argmax_c counts[g] (majority winner, -1 if empty)
        mv_frac       : (n_groups,) float    share of votes for the winner
        mv_task_acc   : (n_groups,) float    fraction of group-g tasks whose TRUE label
                                             == mv_label[g]  (= per-group label accuracy
                                             of group-mode voting); NaN if no votes
        second_label  : (n_groups,) int      2nd-most-voted class in group g (-1 if none)
        second_frac   : (n_groups,) float    share of HQ votes for the runner-up
        second_task_acc:(n_groups,) float    fraction of group-g tasks whose TRUE label
                                             == second_label[g]; NaN if no runner-up
        plurality     : (n_groups,) int      most common true label in group g (-1 if empty)
        plurality_frac: (n_groups,) float    share of group-g tasks that are the plurality class
        mv_is_plurality:(n_groups,) bool     mv_label[g] == plurality[g]
        overall_mv_acc : float               task-weighted mean of mv_task_acc
                                             (= overall group-mode label accuracy)
    """
    rating = np.asarray(rating)
    C = int(n_classes)

    task_col = rating[:, 0].astype(int)
    worker_col = rating[:, 1].astype(int)
    label_col = rating[:, 2].astype(int)

    n_task = int(task_col.max()) + 1
    pg = _pred_group_1d(pred_group, n_task)
    n_groups = len(hq_workers_pred)

    # Per-task HQ membership test: "is worker w HQ for THIS task's group?"
    n_worker = int(worker_col.max()) + 1
    hq_lookup = np.zeros((n_groups, n_worker), dtype=bool)
    n_hq = np.zeros(n_groups, dtype=int)   # predicted-HQ workers per group
    for g in range(n_groups):
        idx = np.unique(np.asarray(hq_workers_pred[g], dtype=int))
        if idx.size:
            hq_lookup[g, idx] = True
            n_hq[g] = idx.size

    # counts[g, c] : HQ votes for class c among tasks in predicted group g.
    counts = np.zeros((n_groups, C), dtype=np.int64)
    for rec in range(rating.shape[0]):
        i = task_col[rec]
        g = pg[i]
        if g < 0 or g >= n_groups:
            continue
        w = worker_col[rec]
        if hq_lookup[g, w]:
            c = label_col[rec]
            if 0 <= c < C:
                counts[g, c] += 1

    n_votes = counts.sum(axis=1)
    vote_frac = counts / np.maximum(n_votes, 1)[:, None]
    vote_frac[n_votes == 0] = 0.0

    mv_label = np.where(n_votes > 0, counts.argmax(1), -1)
    mv_frac = np.divide(
        counts.max(1), np.maximum(n_votes, 1),
        out=np.zeros(n_groups), where=n_votes > 0,
    )

    # Second-most-voted label per group (the runner-up). Needs C >= 2 and at
    # least two classes with votes; second_label = -1 when the runner-up would
    # have zero votes (only one class was voted) or the group is empty.
    order = np.argsort(-counts, axis=1)             # descending vote count
    second_label = np.full(n_groups, -1, dtype=int)
    second_frac = np.zeros(n_groups)
    if C >= 2:
        cand = order[:, 1]                          # 2nd column of the ordering
        cand_count = counts[np.arange(n_groups), cand]
        has_second = (n_votes > 0) & (cand_count > 0)
        second_label[has_second] = cand[has_second]
        second_frac[has_second] = cand_count[has_second] / n_votes[has_second]

    n_tasks = np.array([(pg == g).sum() for g in range(n_groups)], dtype=int)

    # ---- correctness of the majority vote, per predicted group -----------
    # For each predicted group g, mv_label[g] is the class the HQ pool elects.
    # We report how much of the group actually IS that class:
    #
    #   mv_task_acc[g] = (# tasks in group g with true label == mv_label[g])
    #                    / (# tasks in group g)
    #
    # i.e. the label accuracy achieved on group g by stamping the MV winner on
    # all its tasks (the group-mode inference in _mc_infer). The task-weighted
    # mean over groups is the overall group-mode label accuracy.
    #
    # `plurality` (the single most common true label in the group) is kept only
    # to flag groups where the MV winner is not even the plurality class -- a
    # clear miss. It is NOT a one-to-one assignment and may repeat across groups;
    # that is fine, since it drives only the red/black marker, not identity.
    plurality = np.full(n_groups, -1, dtype=int)
    plurality_frac = np.zeros(n_groups)      # share of group that is the plurality class
    mv_task_acc = np.full(n_groups, np.nan)  # share of group that is the MV winner
    second_task_acc = np.full(n_groups, np.nan)  # share of group that is the runner-up label
    mv_is_plurality = np.zeros(n_groups, dtype=bool)

    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)
        for g in range(n_groups):
            task_mask = (pg == g)
            n_g = task_mask.sum()
            if n_g == 0:
                continue
            overlap = np.bincount(y_true[task_mask], minlength=C)
            pl = int(overlap.argmax())
            plurality[g] = pl
            plurality_frac[g] = overlap[pl] / n_g
            if mv_label[g] >= 0:
                mv_task_acc[g] = overlap[mv_label[g]] / n_g
                mv_is_plurality[g] = (mv_label[g] == pl)
            if second_label[g] >= 0:
                second_task_acc[g] = overlap[second_label[g]] / n_g

    known = (~np.isnan(mv_task_acc)) & (n_tasks > 0)
    overall_mv_acc = (
        float(np.average(mv_task_acc[known], weights=n_tasks[known]))
        if known.any() else float("nan")
    )

    return {
        "counts": counts,
        "vote_frac": vote_frac,
        "n_votes": n_votes,
        "n_tasks": n_tasks,
        "n_hq": n_hq,
        "mv_label": mv_label,
        "mv_frac": mv_frac,
        "mv_task_acc": mv_task_acc,
        "second_label": second_label,
        "second_frac": second_frac,
        "second_task_acc": second_task_acc,
        "plurality": plurality,
        "plurality_frac": plurality_frac,
        "mv_is_plurality": mv_is_plurality,
        "overall_mv_acc": overall_mv_acc,
    }


def plot_hq_vote_distribution(dist, path=None, draw=True, title=None):
    """
    Stacked horizontal bars: one bar per PREDICTED group, segments = HQ-vote
    share per class. The segment of the overlap-matched TRUE label is outlined
    in black so you can see whether the majority vote (the largest segment)
    lands on it. Each bar is annotated with the true label, and a red x marks
    the true-label segment when the majority vote misses it.

    Saves to `path` if given; returns the path (or None if draw=False).
    draw=False short-circuits with no file I/O and no figure created.
    """
    if not draw:
        return None

    vote_frac = dist["vote_frac"]
    mv_label = dist["mv_label"]
    mv_task_acc = dist["mv_task_acc"]
    second_label = dist.get("second_label", np.full(vote_frac.shape[0], -1))
    second_task_acc = dist.get("second_task_acc", np.full(vote_frac.shape[0], np.nan))
    mv_is_plurality = dist.get("mv_is_plurality", np.ones(vote_frac.shape[0], dtype=bool))
    n_hq = dist.get("n_hq", np.full(vote_frac.shape[0], -1))
    n_groups, C = vote_frac.shape

    fig, ax = plt.subplots(figsize=(11.5, max(3, 0.55 * n_groups + 2)))
    cmap = plt.get_cmap("tab20" if C > 10 else "tab10")
    ys = np.arange(n_groups)

    # segment boundaries so we can locate the MV-winner segment for marking
    left_edges = np.zeros((n_groups, C))
    acc = np.zeros(n_groups)
    for c in range(C):
        left_edges[:, c] = acc
        acc = acc + vote_frac[:, c]

    for c in range(C):
        for g in range(n_groups):
            seg = vote_frac[g, c]
            if seg <= 0:
                continue
            is_winner = (c == mv_label[g])
            is_second = (c == second_label[g])
            # winner: solid black outline; runner-up: dashed grey outline
            if is_winner:
                ec, lw, ls = "black", 1.8, "-"
            elif is_second:
                ec, lw, ls = "0.35", 1.2, (0, (3, 2))
            else:
                ec, lw, ls = "white", 0.3, "-"
            ax.barh(
                ys[g], seg, left=left_edges[g, c], color=cmap(c % cmap.N),
                edgecolor=ec, linewidth=lw, linestyle=ls,
            )

    # markers + annotations per predicted group
    for g in range(n_groups):
        mv = mv_label[g]
        if mv >= 0:
            # tick the centre of the MV-winner segment
            centre = left_edges[g, mv] + vote_frac[g, mv] / 2
            acc_g = mv_task_acc[g]
            # "good" if the elected label is at least the group's plurality class
            good = bool(mv_is_plurality[g])
            ax.plot(
                centre, ys[g],
                marker="o" if good else "X",
                color="black" if good else "red",
                markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                zorder=5,
            )
            acc_str = "n/a" if np.isnan(acc_g) else f"{acc_g*100:4.1f}%"
            note = f"MV={mv} correct={acc_str}"
            sl = second_label[g]
            if sl >= 0:
                s_str = ("n/a" if np.isnan(second_task_acc[g])
                         else f"{second_task_acc[g]*100:4.1f}%")
                note += f"  |  2nd={sl} correct={s_str}"
            note += f"  |  HQ={int(n_hq[g])}"
            if not good:
                note += "  ✗ not plurality"
        else:
            note = f"(no votes)  HQ={int(n_hq[g])}"
        ax.text(1.015, ys[g], note, va="center", ha="left", fontsize=8,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(ys)
    ax.set_yticklabels([f"pred grp {g}" for g in range(n_groups)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of HQ votes")
    ax.set_ylabel("predicted task group")
    mv_acc = dist.get("overall_mv_acc", float("nan"))
    ax.set_title(
        title or
        f"HQ vote distribution by predicted group "
        f"(group-mode label accuracy {mv_acc*100:.1f}%)"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(c % cmap.N)) for c in range(C)
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="black", ls="", markersize=8,
                   markeredgecolor="white", label="MV = plurality class"),
        plt.Line2D([0], [0], marker="X", color="red", ls="", markersize=8,
                   markeredgecolor="white", label="MV != plurality"),
    ]
    labels = [f"voted {c}" for c in range(C)] + ["MV = plurality", "MV != plurality"]
    # Legend placed BELOW the plot so it never collides with the right-side
    # per-bar annotations. Columns scale with the number of classes.
    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(6, C + 2), fontsize=8, frameon=False,
        title="vote / marker",
    )

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def hq_vote_report(
    rating,
    pred_group,
    hq_workers_pred,
    n_classes,
    out_dir,
    method_name,
    y_true=None,
    draw=True,
):
    """
    Convenience wrapper: compute + (optionally) plot + save the raw matrix.
    Returns the dist dict; writes <out_dir>/hq_votes_<method>.png and
    hq_votes_<method>.csv when draw=True. The CSV carries the per-group true
    label and majority-vote winner alongside the count matrix.
    """
    dist = compute_hq_vote_distribution(
        rating, pred_group, hq_workers_pred, n_classes, y_true=y_true
    )
    if draw:
        png = os.path.join(out_dir, f"hq_votes_{method_name}.png")
        plot_hq_vote_distribution(
            dist, path=png, draw=True,
            title=f"HQ vote distribution by predicted group ({method_name})",
        )
        os.makedirs(out_dir, exist_ok=True)
        C = n_classes
        header = (
            ",".join(f"voted_{c}" for c in range(C))
            + ",mv_label,mv_correct_frac,second_label,second_correct_frac,"
            + "plurality,n_votes,n_tasks"
        )
        mv_acc = dist["mv_task_acc"].astype(float)
        sec_acc = dist["second_task_acc"].astype(float)
        block = np.column_stack([
            dist["counts"].astype(float),
            dist["mv_label"][:, None].astype(float),
            mv_acc[:, None],
            dist["second_label"][:, None].astype(float),
            sec_acc[:, None],
            dist["plurality"][:, None].astype(float),
            dist["n_votes"][:, None].astype(float),
            dist["n_tasks"][:, None].astype(float),
        ])
        # counts + label/count columns are ints; the two correct-frac cols are rates
        fmt = ["%d"] * C + ["%d", "%.4f", "%d", "%.4f", "%d", "%d", "%d"]
        np.savetxt(
            os.path.join(out_dir, f"hq_votes_{method_name}.csv"),
            block, fmt=fmt, delimiter=",", header=header, comments="",
        )
    return dist