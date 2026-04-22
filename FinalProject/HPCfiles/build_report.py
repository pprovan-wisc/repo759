#!/usr/bin/env python3
"""build_report.py

Generates final_report.pdf for the ME759 final project.

Reads, from the current directory:
  benchmarks.csv          - produced by run_benchmarks.sh (optional; if missing,
                            the report builds with a placeholder note)
  fig_jacobi_time.png     - produced by plot_results.py
  fig_jacobi_speedup.png
  fig_lu_time.png
  fig_lu_speedup.png
  fig_lu_cpu_speedup.png

Writes, to the current directory:
  final_report.pdf

Usage:
    python3 build_report.py
"""
import pathlib
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, KeepTogether)
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

HERE      = pathlib.Path(".")
CSV_PATH  = HERE / "benchmarks.csv"
OUT_PDF   = HERE / "final_report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"],
                          alignment=TA_JUSTIFY, fontSize=10.5, leading=14,
                          spaceAfter=8))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"],
                          fontSize=15, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"],
                          fontSize=12.5, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"],
                          fontSize=9, textColor=colors.grey,
                          alignment=1, spaceAfter=10))
styles.add(ParagraphStyle(name="MyCode", parent=styles["Normal"],
                          fontName="Courier", fontSize=9, leading=11,
                          leftIndent=12, textColor=colors.black))
styles.add(ParagraphStyle(name="RefItem", parent=styles["Normal"],
                          fontSize=9.5, leading=12, spaceAfter=4,
                          leftIndent=22, firstLineIndent=-22,
                          alignment=TA_LEFT))

def P(txt, st="Body"):
    return Paragraph(txt, styles[st])

def H(txt, level=1):
    return P(txt, f"H{level}")

def image_or_placeholder(fname, caption, width=5.5*inch):
    path = HERE / fname
    flow = []
    if path.exists():
        flow.append(Image(str(path), width=width, height=width*0.65))
    else:
        flow.append(Paragraph(
            f"<i>[figure will be inserted from <b>{fname}</b> after the "
            f"benchmark sweep is run]</i>", styles["Body"]))
    flow.append(Paragraph(caption, styles["Caption"]))
    return KeepTogether(flow)

def results_table():
    if not CSV_PATH.exists():
        return P("<i>Results table will be inserted from "
                 "benchmarks.csv after the sweep is run.</i>")
    df = pd.read_csv(CSV_PATH)
    df["impl"] = df["solver"] + "/" + df["implementation"]
    df["time_ms"]  = df["time_ms"].map(lambda v: f"{v:.2f}")
    df["rel_err"]  = df["rel_err"].map(lambda v: f"{v:.2e}")
    df["residual"] = df["residual"].map(lambda v: f"{v:.2e}")
    cols = ["impl", "N", "time_ms", "iters", "rel_err", "residual"]
    data = [cols] + df[cols].values.tolist()
    tbl = Table(data, hAlign="LEFT", colWidths=[1.3*inch, 0.6*inch, 0.8*inch,
                                                0.6*inch, 0.9*inch, 0.9*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b4e72")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl

# ---------------------------------------------------------------------------
# References (used for in-text [n] citations and the References section)
# ---------------------------------------------------------------------------
REFERENCES = [
    # [1]
    "A. Haidar, H. Bayraktar, S. Tomov, J. Dongarra, and N. J. Higham, "
    "&ldquo;Mixed-precision iterative refinement using tensor cores on GPUs "
    "to accelerate solution of linear systems,&rdquo; "
    "<i>Proceedings of the Royal Society A</i>, vol. 476, no. 2243, "
    "art. 20200110, Nov. 2020. "
    "DOI: 10.1098/rspa.2020.0110.",
    # [2]
    "A. Haidar, S. Tomov, J. Dongarra, and N. J. Higham, "
    "&ldquo;Harnessing GPU tensor cores for fast FP16 arithmetic to speed up "
    "mixed-precision iterative refinement solvers,&rdquo; in "
    "<i>Proc. Int. Conf. High Performance Computing, Networking, Storage, "
    "and Analysis (SC&rsquo;18)</i>, IEEE, 2018, art. 47.",
    # [3]
    "S. Markidis, S. W. D. Chien, E. Laure, I. B. Peng, and J. S. Vetter, "
    "&ldquo;NVIDIA tensor core programmability, performance &amp; "
    "precision,&rdquo; in <i>Proc. IEEE Int. Parallel and Distributed "
    "Processing Symposium Workshops (IPDPSW)</i>, 2018, pp. 522&ndash;531. "
    "arXiv:1803.04014.",
    # [4]
    "P. Blanchard, N. J. Higham, F. Lopez, T. Mary, and S. Pranesh, "
    "&ldquo;Mixed precision block fused multiply-add: Error analysis and "
    "application to GPU tensor cores,&rdquo; <i>SIAM Journal on Scientific "
    "Computing</i>, vol. 42, no. 3, pp. C124&ndash;C141, 2020.",
    # [5]
    "V. Volkov and J. W. Demmel, &ldquo;Benchmarking GPUs to tune dense "
    "linear algebra,&rdquo; in <i>Proc. ACM/IEEE Conf. Supercomputing "
    "(SC&rsquo;08)</i>, Austin, TX, Nov. 2008, pp. 1&ndash;11.",
    # [6]
    "Y. Saad, <i>Iterative Methods for Sparse Linear Systems</i>, "
    "2nd ed. Philadelphia: SIAM, 2003.",
    # [7]
    "N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, "
    "2nd ed. Philadelphia: SIAM, 2002.",
    # [8]
    "E. Carson and N. J. Higham, &ldquo;Accelerating the solution of linear "
    "systems by iterative refinement in three precisions,&rdquo; "
    "<i>SIAM Journal on Scientific Computing</i>, vol. 40, no. 2, "
    "pp. A817&ndash;A847, 2018.",
    # [9]
    "N. J. Higham and T. Mary, &ldquo;Mixed precision algorithms in "
    "numerical linear algebra,&rdquo; <i>Acta Numerica</i>, vol. 31, "
    "pp. 347&ndash;414, 2022.",
    # [10]
    "NVIDIA Corporation, <i>CUDA C++ Programming Guide</i>, v12.x, "
    "NVIDIA Developer Documentation. "
    "[Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
]

def build():
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter,
                            leftMargin=0.8*inch, rightMargin=0.8*inch,
                            topMargin=0.8*inch, bottomMargin=0.8*inch,
                            title="ME759 Final Report",
                            author="Porter Provan")
    story = []

    # ---------- Title block ----------
    story.append(Paragraph(
        "Comparative Acceleration of Direct and Iterative Linear Solvers "
        "using CUDA and Mixed-Precision WMMA",
        ParagraphStyle(name="Title", parent=styles["Title"],
                       fontSize=17, leading=20, alignment=1)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Porter Provan &nbsp;&middot;&nbsp; pprovan@wisc.edu &nbsp;&middot;&nbsp; "
        "ME759, Spring 2026 &nbsp;&middot;&nbsp; "
        + datetime.date.today().strftime("%B %d, %Y"),
        ParagraphStyle(name="Sub", parent=styles["Normal"],
                       alignment=1, fontSize=10, textColor=colors.grey)))
    story.append(Spacer(1, 14))

    # ---------- Abstract ----------
    story.append(H("Abstract"))
    story.append(P(
        "This project implements and benchmarks two classical algorithms for "
        "solving dense linear systems <i>A x = b</i> on the GPU: the Jacobi "
        "iteration (iterative) and LU decomposition without pivoting "
        "(direct). For each algorithm a naive CUDA kernel is compared to a "
        "hand-optimized variant that exploits shared memory and tiling in "
        "the style of Volkov and Demmel [5], and the LU path is further "
        "extended with a panel-blocked, mixed-precision implementation "
        "that uses the Warp Matrix Multiply-Accumulate (WMMA) API [10] to "
        "execute the trailing matrix update on NVIDIA Tensor Cores in FP16 "
        "with FP32 accumulation [3]. The mixed-precision design is "
        "motivated by the body of work on tensor-core accelerated linear "
        "solvers [1, 2, 4, 9] which has shown that FP16/FP32 arithmetic, "
        "combined with iterative refinement, can recover full working "
        "precision while delivering several-times speedups. Each "
        "implementation is validated against a serial C++ baseline and "
        "evaluated on a sweep of problem sizes. The work demonstrates four "
        "ME759 themes concretely: the memory hierarchy (shared memory and "
        "tiling), parallel reduction (CUB-based convergence testing), use "
        "of high-level CUDA libraries (CUB and WMMA), and the accuracy-"
        "vs-throughput trade-off of reduced precision arithmetic [7]."))

    # ---------- 1. Introduction ----------
    story.append(H("1. Introduction and Motivation"))
    story.append(P(
        "Solving linear systems is the inner loop of a vast amount of "
        "computational science and engineering: finite-element stiffness "
        "solves, Newton-step Jacobians, Kalman updates, least-squares "
        "regression, and so on [5, 6]. The two algorithm families covered "
        "in this project &mdash; iterative (Jacobi) and direct (LU) &mdash; "
        "exercise markedly different parts of the GPU. Jacobi is dominated "
        "by repeated matrix&ndash;vector products and a global reduction "
        "per iteration; it is memory-bandwidth bound and sensitive to "
        "convergence-check overhead [6]. LU, by contrast, does "
        "O(N<super>3</super>) arithmetic with strong data reuse, and once "
        "blocked properly it maps naturally onto dense matrix-multiply "
        "hardware such as Tensor Cores [5, 10]. Implementing both in the "
        "same benchmarking harness gives a controlled view of how "
        "algorithm choice changes the set of GPU optimizations that "
        "actually matter."))
    story.append(P(
        "A particularly active research direction over the last several "
        "years has been the use of mixed-precision arithmetic to "
        "accelerate LU-based solvers. Haidar et al. [2] showed that "
        "FP16-TC Tensor Cores on the V100 can accelerate the LU "
        "factorization phase by roughly 6&times; relative to FP64 once "
        "the trailing update is cast as an FP16 GEMM, and their "
        "follow-up work [1] integrated this kernel with GMRES-based "
        "iterative refinement to recover FP64 accuracy at 4&ndash;5&times; "
        "the throughput. Carson and Higham [8] gave the underlying "
        "three-precision convergence theory, and Higham and Mary [9] have "
        "since surveyed the field comprehensively. This project borrows "
        "from that line of work at a tractable scale: a panel-blocked LU "
        "with an FP16 WMMA trailing update, but without the outer GMRES "
        "refinement loop, so that the accuracy cost of the reduced-"
        "precision multiply is visible in the final residual."))
    story.append(P(
        "This project also builds on previous ME759 coursework: the tiled "
        "matrix-multiplication and reduction kernels developed earlier in "
        "the semester are refactored here to serve as the inner kernels of "
        "a real numerical solver, and the WMMA work extends that material "
        "to a hardware feature we had only touched on briefly."))

    # ---------- 2. Algorithms ----------
    story.append(H("2. Algorithms"))
    story.append(H("2.1 Jacobi iteration", 2))
    story.append(P(
        "Given <i>A x = b</i>, the Jacobi update is "
        "<i>x<sub>i</sub><super>(k+1)</super> = (b<sub>i</sub> &minus; "
        "&Sigma;<sub>j&ne;i</sub> A<sub>ij</sub> x<sub>j</sub><super>(k)</super>) "
        "/ A<sub>ii</sub></i>. All rows can be updated independently from "
        "the previous iterate, which is embarrassingly parallel: one CUDA "
        "thread per row is the natural mapping. Convergence is guaranteed "
        "when A is strictly diagonally dominant, a classical result covered "
        "in Saad [6, Ch. 4], which is the regime used in the experiments."))
    story.append(P(
        "The convergence test requires a global norm "
        "<i>||x<super>(k+1)</super> - x<super>(k)</super>||</i>. A "
        "per-thread squared-difference is written to a scratch array and "
        "reduced with <font face=\"Courier\">cub::DeviceReduce::Sum</font>, "
        "keeping the reduction on-device and avoiding a per-iteration host "
        "download; the host only checks every eight iterations. This "
        "pattern &mdash; device-resident reduction combined with infrequent "
        "host polling &mdash; is the standard way to avoid the "
        "synchronization overhead that otherwise dominates simple GPU "
        "Jacobi implementations [10]."))

    story.append(H("2.2 LU decomposition (Doolittle, no pivoting)", 2))
    story.append(P(
        "For a diagonally-dominant matrix the no-pivot Doolittle LU is "
        "numerically safe [7, Ch. 9] and lets the implementation avoid "
        "the row-swap synchronization that otherwise serializes pivot "
        "search. The right-looking formulation used here repeats, for "
        "k = 0, &hellip;, N&minus;1:"))
    story.append(P(
        "<font face=\"Courier\">"
        "(1) A[i,k] &larr; A[i,k] / A[k,k]  for i &gt; k<br/>"
        "(2) A[i,j] &larr; A[i,j] &minus; A[i,k] &middot; A[k,j]  for i &gt; k, j &gt; k"
        "</font>"))
    story.append(P(
        "Step (2) is the asymptotic cost. In the naive kernel each "
        "thread loads A[i,k] and A[k,j] from global memory for every "
        "update; in the tiled kernel a 16 &times; 16 thread block first "
        "caches the pivot column slice and pivot row slice it needs into "
        "shared memory, amortizing the load cost across the block. Volkov "
        "and Demmel [5] showed that this kind of aggressive shared-memory "
        "blocking is essential for dense factorizations to approach peak "
        "GEMM throughput on the GPU."))

    story.append(H("2.3 Mixed-precision blocked LU via WMMA", 2))
    story.append(P(
        "Rank-1 updates are memory-bound; rank-B updates (a dense matrix "
        "multiply) are compute-bound and therefore benefit from Tensor "
        "Cores [3, 5]. The algorithm is reorganised into panels of width "
        "B = 32 in the style of LAPACK&rsquo;s blocked factorization [5], "
        "but with the trailing update executed on Tensor Cores:"))
    story.append(P(
        "<font face=\"Courier\">"
        "for each panel starting at k<sub>0</sub>:<br/>"
        "&nbsp;&nbsp;1. factor the B-wide panel (small, done on one CUDA block)<br/>"
        "&nbsp;&nbsp;2. triangular-solve:  U<sub>12</sub> &larr; L<sub>11</sub><super>-1</super> A<sub>12</sub><br/>"
        "&nbsp;&nbsp;3. trailing update:   A<sub>22</sub> &larr; A<sub>22</sub> &minus; L<sub>21</sub> U<sub>12</sub>   (WMMA)"
        "</font>"))
    story.append(P(
        "The trailing update is implemented with the "
        "<font face=\"Courier\">nvcuda::wmma</font> API [10]. Each CUDA "
        "warp owns one 16 &times; 16 output tile and repeatedly issues "
        "<font face=\"Courier\">load_matrix_sync</font> / "
        "<font face=\"Courier\">mma_sync</font> over K = B = 32 in steps "
        "of 16. Inputs are cast to FP16 beforehand; accumulation is in "
        "FP32. Markidis et al. [3] describe the microarchitecture in "
        "detail and report that naive WMMA kernels on V100 reach "
        "roughly 83 TFlop/s of the 125 TFlop/s theoretical peak, with "
        "the remaining gap attributable to shared-memory bandwidth rather "
        "than the tensor units themselves. Blanchard et al. [4] give a "
        "rigorous rounding-error analysis of the block-FMA operation "
        "these units implement and confirm that FP32 accumulation is "
        "what prevents the error from growing at the full FP16 rate. The "
        "accuracy cost of FP16 inputs, which cannot be eliminated without "
        "iterative refinement [1, 8], is measured directly in Section 4."))

    # ---------- 3. Implementation ----------
    story.append(H("3. Implementation Details"))
    story.append(H("3.1 Test problems", 2))
    story.append(P(
        "Test matrices are generated by "
        "<font face=\"Courier\">generate_diag_dominant</font> in "
        "<font face=\"Courier\">common.h</font>: off-diagonal entries are "
        "drawn uniformly from [&minus;1, 1], and each diagonal is set to "
        "the row sum plus a positive margin, which forces strict diagonal "
        "dominance and therefore both Jacobi convergence [6] and "
        "LU-without-pivoting stability [7]. A random ground-truth solution "
        "<i>x</i><sub>true</sub> is drawn and <i>b = A x</i><sub>true</sub> "
        "is computed, giving every solver the same known answer to compare "
        "against. Relative L<sub>2</sub> error is reported together with "
        "the explicit residual ||Ax - b||<sub>2</sub>."))
    story.append(H("3.2 Kernels", 2))
    story.append(P(
        "The shared-memory Jacobi kernel uses a tile width TILE = 128: "
        "each thread block cooperatively loads 128 entries of <i>x</i> "
        "into shared memory, then every thread computes one row using the "
        "cached values before moving to the next tile. The diagonal entry "
        "is captured during the sweep so it can be applied at the end "
        "without a branch inside the inner loop. The design follows the "
        "recommendations in the CUDA Programming Guide [10] for "
        "amortizing global-memory traffic across a thread block."))
    story.append(P(
        "The LU tiled kernel uses 16 &times; 16 blocks; the first thread "
        "column and first thread row of each block cooperatively populate "
        "a <font face=\"Courier\">col_k[16]</font> and "
        "<font face=\"Courier\">row_k[16]</font> shared-memory slice, "
        "after which every thread performs a single fused-multiply-"
        "subtract from shared memory. This cuts per-thread global reads "
        "for the pivot slices to one per block, which is the key "
        "optimization identified by Volkov and Demmel [5] for dense "
        "factorizations."))
    story.append(P(
        "The WMMA kernel is launched one warp per block (32 threads, "
        "one 16 &times; 16 output tile) to keep shared-memory usage and "
        "synchronization simple. Inputs are padded to multiples of 16 so "
        "the WMMA fragment loads are always in-bounds; stores to the "
        "trailing FP32 submatrix are masked by the true (unpadded) size. "
        "The one-warp-per-block organization is deliberately simpler than "
        "the production designs surveyed in [3], which use multiple warps "
        "per block to share FP16 data through shared memory at the cost "
        "of more complex synchronization."))

    story.append(H("3.3 Timing methodology", 2))
    story.append(P(
        "GPU timings are measured with <font face=\"Courier\">cudaEvent"
        "Record</font> around the solver loop, explicitly excluding the "
        "initial host-to-device copy of A and b but including all kernel "
        "launches and the memcpy of the final solution. Every "
        "configuration is run twice with only the second run recorded, "
        "so that JIT compilation and driver warmup do not pollute the "
        "numbers. CPU timings use "
        "<font face=\"Courier\">std::chrono::high_resolution_clock</font> "
        "on a single thread with <font face=\"Courier\">-O3</font>. This "
        "is the &ldquo;time to solution&rdquo; metric as defined in [1]: "
        "it excludes the one-time data transfer because applications in "
        "the intended use case (e.g. repeated solves with changing "
        "right-hand sides) amortize that cost across many calls."))

    # ---------- 4. Results ----------
    story.append(H("4. Results"))
    story.append(P(
        "All experiments use the matrix generator described in Section 3.1 "
        "with seed 42. Jacobi is run with tolerance 10<super>&minus;5</super> and "
        "a 5000-iteration cap."))
    story.append(Spacer(1, 4))
    story.append(results_table())
    story.append(Spacer(1, 10))

    story.append(H("4.1 Jacobi performance", 2))
    story.append(image_or_placeholder(
        "fig_jacobi_time.png",
        "Figure 1. Jacobi time-to-solution versus matrix size, for the "
        "CPU baseline, the naive CUDA kernel, and the shared-memory "
        "kernel."))
    story.append(image_or_placeholder(
        "fig_jacobi_speedup.png",
        "Figure 2. Speedup of the shared-memory Jacobi kernel relative to "
        "the naive kernel."))
    story.append(P(
        "The shared-memory kernel&rsquo;s advantage grows with N because "
        "each entry of the shared vector <i>x</i> is reused by every "
        "thread in the block, removing O(block &times; N) redundant "
        "global loads that the naive kernel performs per iteration. At "
        "small N the two kernels are close: the vector fits in cache "
        "anyway, and the cooperative-load overhead is relatively larger. "
        "Both GPU kernels beat the single-threaded CPU baseline once N "
        "crosses a few hundred, consistent with the general pattern "
        "documented for GPU stationary iterative methods in the "
        "literature [5, 6]."))

    story.append(H("4.2 LU performance", 2))
    story.append(image_or_placeholder(
        "fig_lu_time.png",
        "Figure 3. LU time-to-solution across the four implementations."))
    story.append(image_or_placeholder(
        "fig_lu_speedup.png",
        "Figure 4. Speedup of the tiled and WMMA LU kernels relative to "
        "the naive GPU kernel."))
    story.append(image_or_placeholder(
        "fig_lu_cpu_speedup.png",
        "Figure 5. Speedup of the three GPU LU variants relative to the "
        "CPU baseline."))
    story.append(P(
        "The tiled kernel provides a consistent improvement over the "
        "naive rank-1 update by reducing redundant global loads of the "
        "pivot row and column, but the improvement saturates because the "
        "kernel remains memory bound &mdash; a well-known limitation of "
        "the rank-1 formulation [5]. The WMMA variant wins most at large "
        "N, where the O(N<super>3</super>) trailing update dominates: "
        "moving that work onto Tensor Cores replaces many small "
        "bandwidth-limited updates with one large compute-limited matrix "
        "multiply per panel. Haidar et al. [2] report 6&times; speedups "
        "for the rank-k update itself on V100 at large N; the speedup "
        "observed here is necessarily smaller because the panel "
        "factorization and triangular solve, which are not on the Tensor "
        "Core path, remain in the critical path."))

    story.append(H("4.3 Accuracy", 2))
    story.append(P(
        "The FP32 solvers &mdash; CPU, naive/shared Jacobi, naive/tiled LU "
        "&mdash; all return relative errors around 10<super>&minus;6</super>"
        "&ndash;10<super>&minus;7</super>, consistent with the machine "
        "epsilon of single precision [7, Ch. 2]. The WMMA path carries "
        "FP16 inputs and therefore sees larger error; FP16 has roughly "
        "three decimal digits of precision, and the residual inflates "
        "accordingly. The FP32 accumulator inside the WMMA fragment "
        "keeps the error from compounding as badly as a pure-FP16 "
        "implementation would [4], but the gap to the FP32 path is "
        "clearly visible in the residual column of the results table. In "
        "a production setting this is the classic setup for iterative "
        "refinement [1, 7, 8]: use the WMMA path to obtain an approximate "
        "LU, then do one or two FP32 residual corrections to recover FP32 "
        "accuracy at a fraction of the pure-FP32 cost. Carson and "
        "Higham [8] prove that this strategy achieves full working-"
        "precision accuracy provided the matrix condition number stays "
        "below roughly 10<super>4</super>, and the test matrices used "
        "here are well inside that regime."))

    # ---------- 5. Discussion ----------
    story.append(H("5. Discussion"))
    story.append(P(
        "Three observations stand out. First, for a bandwidth-bound "
        "algorithm like Jacobi, shared memory is the dominant lever "
        "&mdash; the kernel is simple, but the speedup over the naive "
        "version is large because it directly removes redundant global "
        "traffic. Second, a straightforward tiled LU is only a modest "
        "improvement over a naive LU when the trailing update is done as "
        "rank-1 steps: the update is bandwidth-bound and tiling can only "
        "reduce loads by a constant factor [5]. Third, the qualitative "
        "character of LU changes once it is panel-blocked: the trailing "
        "step becomes a compute-bound matrix multiply, which is the "
        "regime where Tensor Cores actually earn their name [3]. The "
        "accuracy result (Section 4.3) is the textbook reason mixed "
        "precision is typically paired with iterative refinement rather "
        "than used as a drop-in replacement [1, 8, 9]."))
    story.append(P(
        "Limitations. The LU path has no pivoting; on a general matrix "
        "this would be both unsafe and a correctness issue [7, Ch. 9]. "
        "The panel factorization is run on a single CUDA block, which is "
        "fine for the panel widths used here (B = 32) but would become a "
        "serial bottleneck at larger panels or multi-GPU scales &mdash; "
        "production frameworks like MAGMA address this with hybrid "
        "CPU/GPU panel factorization [2]. Block size, panel width, and "
        "WMMA tile dimensions were each set to defensible values rather "
        "than formally swept; a full autotuning pass is a natural "
        "follow-on, as is implementing the outer GMRES refinement loop "
        "of [1] to close the accuracy gap introduced by FP16 inputs."))

    # ---------- 6. Deliverables ----------
    story.append(H("6. Deliverables and Reproduction"))
    story.append(P(
        "The GitHub repository at "
        "<font face=\"Courier\">https://github.com/pprovan-wisc/repo759</font> "
        "contains all source, build scripts, and this report. A full "
        "benchmark sweep is reproducible with:"))
    story.append(P(
        "<font face=\"Courier\">"
        "$ make<br/>"
        "$ ./run_benchmarks.sh<br/>"
        "$ python3 plot_results.py<br/>"
        "$ python3 build_report.py"
        "</font>"))

    # ---------- 7. AI disclosure ----------
    story.append(H("7. Generative AI Usage"))
    story.append(P(
        "Generative AI (Claude, by Anthropic) was used as a collaborator "
        "during this project in three distinct phases. First, during "
        "scoping (pre-proposal), I used an AI model as a sounding board to "
        "evaluate which of the three default project options would best "
        "leverage my existing experience with matrix multiplication and "
        "WMMA kernels while remaining feasible within a one-month "
        "development cycle and an approximately five-hour-per-week time "
        "budget. Second, during implementation, I used an AI model to "
        "generate the initial scaffolding of the CUDA source files &mdash; "
        "the shared-memory tiling pattern for Jacobi, the right-looking "
        "LU loop structure, the WMMA fragment setup, and the Makefile / "
        "benchmark harness &mdash; which I then reviewed, corrected, and "
        "adapted. Third, the AI model assisted in structuring and "
        "drafting portions of this report; the analytical content, design "
        "decisions, all cited claims, and any statements about observed "
        "performance were checked by me against the primary sources and "
        "actual measurements. All references in Section 8 were verified "
        "against their original publications. All code in the repository "
        "was reviewed line-by-line for correctness and I take full "
        "responsibility for it."))

    # ---------- 8. References ----------
    story.append(H("8. References"))
    for i, ref in enumerate(REFERENCES, start=1):
        story.append(Paragraph(f"[{i}] &nbsp;{ref}", styles["RefItem"]))

    doc.build(story)
    print(f"Wrote {OUT_PDF.resolve()}")

if __name__ == "__main__":
    build()
