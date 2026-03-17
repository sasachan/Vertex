"""
Vertex — Session Visualizer
Generates a multi-panel analysis chart from a TheSighter session CSV.

Usage:
    python -m vertex.visualize                          # latest session
    python -m vertex.visualize sessions/session_XXX.csv # specific file
"""

import csv
import glob
import os
import sys

import numpy as np

from .models import GOLD


def load_session(path):
    """Load session CSV into list of dicts with numeric conversion."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    if v in ("True", "False"):
                        row[k] = v == "True"
                    else:
                        row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            rows.append(row)
    return rows


def find_latest_session(sessions_dir):
    """Find the most recent session CSV."""
    pattern = os.path.join(sessions_dir, "session_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def visualize(path):
    """Generate visualization from session CSV."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch  # noqa: F401
    import matplotlib.gridspec as gridspec

    rows = load_session(path)
    if not rows:
        print("No data in session file.")
        return

    n = len(rows)
    shots = [int(r.get("shot_number", i + 1)) for i, r in enumerate(rows)]
    holds = [r.get("hold_seconds", 0) for r in rows]
    anc_var = [r.get("anchor_distance_var", 0) for r in rows]
    anc_mean = [r.get("anchor_distance_mean", 0) for r in rows]

    rel_mag = []
    for r in rows:
        if "release_jump_mag" in r:
            rel_mag.append(r["release_jump_mag"])
        elif "release_jump" in r:
            rel_mag.append(r["release_jump"])
        else:
            rel_mag.append(0)

    has_bio = "bow_shoulder_angle" in rows[0]
    has_sway = "sway_velocity" in rows[0]
    if has_bio:
        bsa = [r.get("bow_shoulder_angle", 0) for r in rows]
        dea = [r.get("draw_elbow_angle", 0) for r in rows]
        stilt = [abs(r.get("shoulder_tilt_deg", 0)) for r in rows]
        tlean = [abs(r.get("torso_lean_deg", 0)) for r in rows]
        dlen = [r.get("draw_length_norm", 0) for r in rows]
        is_valid = [r.get("is_valid", True) for r in rows]
    else:
        is_valid = [True] * n
    if has_sway:
        sway_vel = [r.get("sway_velocity", 0) for r in rows]
        sway_rx = [r.get("sway_range_x", 0) for r in rows]
        sway_ry = [r.get("sway_range_y", 0) for r in rows]
        dfl = [r.get("dfl_angle", 0) for r in rows]

    shot_colors = ["#2ecc71" if v else "#e74c3c" for v in is_valid]

    n_rows = 3
    if has_bio:
        n_rows += 1
    if has_sway:
        n_rows += 1
    fig = plt.figure(figsize=(16, 4 * n_rows))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(n_rows, 3, hspace=0.35, wspace=0.3)

    title = os.path.basename(path).replace(".csv", "")
    fig.suptitle(f"VERTEX \u2014 TheSighter Session Analysis\n{title}",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    def style_ax(ax, title, ylabel=""):
        ax.set_facecolor("#16213e")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444")

    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "Hold Time per Shot (seconds)", "Hold (s)")
    ax1.bar(shots, holds, color=shot_colors, alpha=0.85, edgecolor="none")
    ax1.axhspan(GOLD["hold_min"], GOLD["hold_max"], alpha=0.15, color="#2ecc71",
                label=f"Gold range ({GOLD['hold_min']}-{GOLD['hold_max']}s)")
    ax1.axhline(GOLD["hold_min"], color="#2ecc71", linestyle="--", linewidth=0.8)
    ax1.axhline(GOLD["hold_max"], color="#2ecc71", linestyle="--", linewidth=0.8)
    if holds:
        ax1.axhline(np.mean(holds), color="#f39c12", linestyle="-", linewidth=1.2,
                     label=f"Session avg ({np.mean(holds):.1f}s)")
    ax1.set_xlabel("Shot #", color="white", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, "Hold Distribution", "Count")
    if holds:
        ax2.hist(holds, bins=max(8, n // 3), color="#3498db", alpha=0.8, edgecolor="#1a1a2e")
        ax2.axvline(GOLD["hold_min"], color="#2ecc71", linestyle="--", linewidth=1)
        ax2.axvline(GOLD["hold_max"], color="#2ecc71", linestyle="--", linewidth=1)
        ax2.axvline(np.mean(holds), color="#f39c12", linestyle="-", linewidth=1.2)
    ax2.set_xlabel("Hold (s)", color="white", fontsize=9)

    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3, "Anchor Stability Index (lower = more elite)", "Variance")
    ax3.bar(shots, anc_var, color=shot_colors, alpha=0.85, edgecolor="none")
    ax3.axhline(GOLD["anchor_var_elite"], color="#2ecc71", linestyle="-",
                linewidth=1, label=f"Elite (<{GOLD['anchor_var_elite']})")
    ax3.axhline(GOLD["anchor_var_good"], color="#f39c12", linestyle="--",
                linewidth=1, label=f"Good (<{GOLD['anchor_var_good']})")
    ax3.axhline(GOLD["anchor_var_poor"], color="#e74c3c", linestyle="--",
                linewidth=1, label=f"Poor (>{GOLD['anchor_var_poor']})")
    ax3.set_xlabel("Shot #", color="white", fontsize=9)
    ax3.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    ax3.set_yscale("log")

    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, "Anchor Distance per Shot", "Norm. Distance")
    ax4.plot(shots, anc_mean, "o-", color="#3498db", markersize=4, linewidth=1)
    if anc_mean:
        ax4.axhline(np.mean(anc_mean), color="#f39c12", linestyle="-", linewidth=1,
                     label=f"Mean ({np.mean(anc_mean):.3f})")
        ax4.fill_between(shots,
                         np.mean(anc_mean) - np.std(anc_mean),
                         np.mean(anc_mean) + np.std(anc_mean),
                         alpha=0.2, color="#3498db", label=f"\u00b11\u03c3 ({np.std(anc_mean):.3f})")
    ax4.set_xlabel("Shot #", color="white", fontsize=9)
    ax4.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    ax5 = fig.add_subplot(gs[2, :2])
    style_ax(ax5, "Release Jump Magnitude", "Magnitude")
    ax5.bar(shots, rel_mag, color="#9b59b6", alpha=0.85, edgecolor="none")
    if rel_mag:
        ax5.axhline(np.mean(rel_mag), color="#f39c12", linestyle="-", linewidth=1,
                     label=f"Mean ({np.mean(rel_mag):.3f})")
    ax5.set_xlabel("Shot #", color="white", fontsize=9)
    ax5.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor("#16213e")
    ax6.axis("off")

    valid_holds = [h for h, v in zip(holds, is_valid) if v]
    valid_vars = [v for v, ok in zip(anc_var, is_valid) if ok]

    summary = [
        ("Total Shots", f"{n}"),
        ("Valid Shots", f"{sum(is_valid)}/{n}"),
        ("Avg Hold", f"{np.mean(valid_holds):.1f}s" if valid_holds else "--"),
        ("Hold \u03c3", f"{np.std(valid_holds):.2f}s" if valid_holds else "--"),
        ("Best Hold", f"{max(valid_holds):.1f}s" if valid_holds else "--"),
        ("In Gold Range", f"{sum(1 for h in valid_holds if GOLD['hold_min'] <= h <= GOLD['hold_max'])}/{len(valid_holds)}" if valid_holds else "--"),
        ("Avg Anchor Var", f"{np.mean(valid_vars):.6f}" if valid_vars else "--"),
        ("Anchor Rating", (
            "ELITE" if valid_vars and np.mean(valid_vars) < GOLD["anchor_var_elite"]
            else "GOOD" if valid_vars and np.mean(valid_vars) < GOLD["anchor_var_good"]
            else "DEVELOPING" if valid_vars and np.mean(valid_vars) < GOLD["anchor_var_poor"]
            else "NEEDS WORK"
        )),
    ]

    for i, (label, value) in enumerate(summary):
        y = 0.92 - i * 0.11
        ax6.text(0.05, y, label, transform=ax6.transAxes, fontsize=9,
                 color="#bbb", fontweight="normal")
        color = "#2ecc71" if "ELITE" in str(value) or "GOOD" in str(value) else (
            "#f39c12" if "DEVELOPING" in str(value) else "white")
        ax6.text(0.65, y, value, transform=ax6.transAxes, fontsize=10,
                 color=color, fontweight="bold")

    if has_bio:
        ax7 = fig.add_subplot(gs[3, 0])
        style_ax(ax7, "Bow Shoulder Angle", "Degrees")
        ax7.plot(shots, bsa, "o-", color="#e67e22", markersize=4, linewidth=1)
        ax7.axhspan(GOLD["bow_shoulder_min"], GOLD["bow_shoulder_max"],
                     alpha=0.15, color="#2ecc71")
        ax7.axhline(GOLD["bow_shoulder_min"], color="#2ecc71", linestyle="--", linewidth=0.8)
        ax7.axhline(GOLD["bow_shoulder_max"], color="#2ecc71", linestyle="--", linewidth=0.8)
        ax7.set_xlabel("Shot #", color="white", fontsize=9)

        ax8 = fig.add_subplot(gs[3, 1])
        style_ax(ax8, "Draw Elbow Angle", "Degrees")
        ax8.plot(shots, dea, "o-", color="#1abc9c", markersize=4, linewidth=1)
        ax8.axhspan(GOLD["draw_elbow_min"], GOLD["draw_elbow_max"],
                     alpha=0.15, color="#2ecc71")
        ax8.axhline(GOLD["draw_elbow_min"], color="#2ecc71", linestyle="--", linewidth=0.8)
        ax8.axhline(GOLD["draw_elbow_max"], color="#2ecc71", linestyle="--", linewidth=0.8)
        ax8.set_xlabel("Shot #", color="white", fontsize=9)

        ax9 = fig.add_subplot(gs[3, 2])
        style_ax(ax9, "Posture Quality", "Degrees")
        ax9.plot(shots, stilt, "o-", color="#e74c3c", markersize=3,
                 linewidth=1, label="Shoulder tilt")
        ax9.plot(shots, tlean, "s-", color="#3498db", markersize=3,
                 linewidth=1, label="Torso lean")
        ax9.axhline(GOLD["shoulder_tilt_max"], color="#f39c12", linestyle="--",
                     linewidth=0.8, label=f"Max ({GOLD['shoulder_tilt_max']}\u00b0)")
        ax9.set_xlabel("Shot #", color="white", fontsize=9)
        ax9.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    if has_sway:
        sway_row = 4 if has_bio else 3

        ax_sv = fig.add_subplot(gs[sway_row, 0])
        style_ax(ax_sv, "Body Sway Velocity (CoP proxy)", "SW/frame")
        sway_colors = [
            "#2ecc71" if v < GOLD["sway_velocity_elite"]
            else "#f39c12" if v < GOLD["sway_velocity_good"]
            else "#e74c3c" for v in sway_vel
        ]
        ax_sv.bar(shots, sway_vel, color=sway_colors, alpha=0.85, edgecolor="none")
        ax_sv.axhline(GOLD["sway_velocity_elite"], color="#2ecc71", linestyle="--",
                       linewidth=0.8, label=f"Elite (<{GOLD['sway_velocity_elite']})")
        ax_sv.axhline(GOLD["sway_velocity_good"], color="#f39c12", linestyle="--",
                       linewidth=0.8, label=f"Good (<{GOLD['sway_velocity_good']})")
        ax_sv.set_xlabel("Shot #", color="white", fontsize=9)
        ax_sv.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

        ax_dfl = fig.add_subplot(gs[sway_row, 1])
        style_ax(ax_dfl, "Draw Force Line Angle", "Degrees")
        ax_dfl.plot(shots, dfl, "o-", color="#e67e22", markersize=4, linewidth=1)
        if dfl:
            ax_dfl.axhline(np.mean(dfl), color="#f39c12", linestyle="-", linewidth=1,
                           label=f"Mean ({np.mean(dfl):.1f}\u00b0)")
            ax_dfl.fill_between(shots,
                                np.mean(dfl) - np.std(dfl),
                                np.mean(dfl) + np.std(dfl),
                                alpha=0.2, color="#e67e22", label=f"\u00b11\u03c3 ({np.std(dfl):.1f}\u00b0)")
        ax_dfl.set_xlabel("Shot #", color="white", fontsize=9)
        ax_dfl.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

        ax_sr = fig.add_subplot(gs[sway_row, 2])
        style_ax(ax_sr, "Sway Range (X vs Y)", "Y range (SW)")
        ax_sr.scatter(sway_rx, sway_ry, c=shot_colors, s=40, alpha=0.8, edgecolors="none")
        ax_sr.set_xlabel("X range (SW)", color="white", fontsize=9)
        if sway_rx and sway_ry:
            ax_sr.axvline(np.mean(sway_rx), color="#f39c12", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_sr.axhline(np.mean(sway_ry), color="#f39c12", linestyle="--", linewidth=0.8, alpha=0.6)

    out_path = path.replace(".csv", ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Visualization saved: {out_path}")
    return out_path


def main():
    sessions_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sessions")

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_session(sessions_dir)

    if not path or not os.path.exists(path):
        print("No session file found. Run a session first.")
        return

    print(f"  Loading: {path}")
    rows = load_session(path)
    print(f"  Shots:   {len(rows)}")
    visualize(path)


if __name__ == "__main__":
    main()
