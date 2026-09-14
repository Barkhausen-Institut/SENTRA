# SENTRA — Evaluation (Dissertation Draft)

**Profile:** `with-client` — 10,000 train samples, 10 epochs, batch size 64, learning rate 0.002, seed 2026, 100 test samples for final accuracy.  
**Model:** MNIST MLP 784→128→10 (~0.10M parameters).  
**Data sources:** `benchmarking/BENCHMARK_TIMING_COMPARISON.md`, `benchmarking/ml-benchmark/configs/with-client.yaml`.

---

## Table of contents

1. [System implementation](#1-system-implementation)
2. [TEE deployment scope and system comparison](#2-tee-deployment-scope-and-system-comparison)
3. [Performance results](#3-performance-results)
4. [Performance analysis](#4-performance-analysis)
5. [Scalability analysis](#5-scalability-analysis)
6. [Limitations](#6-limitations)
7. [LaTeX source (copy into thesis)](#7-latex-source-copy-into-thesis)

---

## 1. System implementation

### 1.1 Software

SENTRA is implemented as a multi-party secure training stack in which a **data-owning client** secret-shares MNIST features and labels to **n = 3** compute nodes using packed Shamir secret sharing with reconstruction threshold **t = 1**. Training is orchestrated by `start_all_nodes.py` and executed in batched form via `run_mnist_batched_secure.py`, with inter-node coordination over a versioned key–value store and synchronised barriers in `secure_comm.py`.

| Component | Role |
|-----------|------|
| Linear layers | Beaver-triple-based multiplication on secret-shared tensors |
| ReLU | Secure comparison |
| Softmax / loss | Fixed-point secure softmax; Padé(2,2) exponential approximation (`exp-approx: pade22`) |
| Gradients | Protocol-compliant `secure_approx` (default); `opened_exact` for ablations only |
| Arithmetic | 61-bit prime field (`field-size = 2^61 − 1`), scale factor `2^16` |
| Prover (node 3) | Selective openings and re-sharing for division, comparison, softmax |

**Deployment variants evaluated**

| Variant | Description |
|---------|-------------|
| SENTRA (WSL) | Native multi-process on WSL development host |
| SENTRA (Docker) | Non-SGX worker container (`sentra-worker-3`) |
| SENTRA (SGX) | Intel SGX via Occlum inside Docker |
| CrypTen | MPC baseline, aligned config (`sentra-with-client.yaml`) |

**Optional extensions:** per-epoch weight versioning (`with-client-versioned.yaml`), failure detection / dropout recovery (`with-client-fault-5node.yaml`), runtime metrics (`SENTRA_RUNTIME_METRICS=1`).

### 1.2 Models and workloads

**Primary model:** two-layer MLP on MNIST:

```text
784 → 128 → ReLU → 10 → softmax
```

**Parameters:** 784×128 + 128 + 128×10 + 10 = **101,770** (~**0.10M**).

**Primary workload (`with-client` profile)**

| Setting | Value |
|---------|-------|
| Training samples | 10,000 |
| Test samples (client eval) | 100 |
| Batch size | 64 (157 batches/epoch) |
| Epochs | 10 |
| Learning rate | 0.002 |
| Seed | 2026 |
| Loss mode | softmax, temperature 2.0 |

**Additional profiles**

| Profile | Train samples | Epochs | Purpose |
|---------|---------------|--------|---------|
| Quick | 256 | 1 | Fast node-owned-data smoke test |
| Fault 5-node | 512 | 5 | Dropout recovery and runtime metrics |
| Sweep default | 10,000 | 8 | Cross-framework sweep (lr 0.003) |

Shared train/test row indices: `benchmarking/ml-benchmark/assets/train_indices.npy` and `benchmarking/ml-benchmark/assets/test_indices.npy`.

### 1.3 Hardware

| Environment | Host | Notes |
|-------------|------|-------|
| Docker non-SGX, SGX, CrypTen | `sentra-worker-3` (Linux) | Primary thesis benchmarks |
| SENTRA WSL | WSL development host | Native multi-process runs |
| SGX | Intel SGX + Occlum in Docker | ~251 s enclave cold start per node |

*Fill in exact CPU model and RAM for the thesis (e.g. 12th Gen Intel Core i7-12650H, 64 GB RAM) to match your hardware subsection.*

Implementation uses one Python process per party; workloads are not heavily multi-threaded within a single node.

---

## 2. TEE deployment scope and system comparison

The committee selection protocol assumes diversity across hardware platforms, operators, and physical machines. Our implementation focuses only on **Intel SGX** deployed on our own infrastructure, without extracting physical machine identifiers from attestation reports. This simplified model demonstrates feasibility on Intel SGX; extending it to other TEEs such as AMD SEV-SNP is outside the scope of this work. Operator and machine identifiers are assumed to come from an **external manifest layer**.

SENTRA uses TEEs to secure both storage and computation. To the best of our knowledge, it is the first distributed training system combining **collective TEE attestation**, **epoch-bound membership**, and **DPSS resharing** to tolerate node churn under packed-MPC constraints.

**Table 2** compares SENTRA with CrypTen and TruShare. Unlike CrypTen and TruShare, SENTRA integrates TEE isolation, packed MPC, DPSS resharing, and versioned storage semantics to provide rollback protection, dynamic membership, and resilient privacy-preserving distributed training. Compared to CrypTen [27], which assumes static MPC participants, SENTRA adds infrastructure-aware protections against rollback attacks, stale membership reuse, and node churn failures, extending secure MPC training into a resilient cloud-native confidential computing service.

**Table 3** compares the performance of SENTRA and CrypTen, including SENTRA **without** confidential computing enabled (Docker non-SGX).

### Table 2 — Feature comparison (SENTRA vs CrypTen vs TruShare)

| Capability | SENTRA | CrypTen | TruShare |
|------------|:------:|:-------:|:--------:|
| TEE isolation (Intel SGX) | ✓ | ✗ | ✗ |
| Packed / SIMD-style MPC | ✓ | Partial | ✓ |
| DPSS resharing (node churn) | ✓ | ✗ | ✗ |
| Versioned storage (rollback protection) | ✓ | ✗ | ✗ |
| Epoch-bound membership | ✓ | ✗ | ✗ |
| Collective TEE attestation | ✓† | ✗ | ✗ |
| Client-side data ownership (labels local) | ✓ | ✗ | Varies |
| Dynamic membership / fault recovery | ✓ | ✗ | ✗ |
| Static MPC participant set | ✗ | ✓ | ✓ |

† Evaluated on homogeneous Intel SGX; operator/machine IDs from external manifest (not parsed from attestation reports in this prototype).

### Table 3 — Performance comparison (`with-client` profile)

| System | Confidential computing | Accuracy (%) | Training (h) | Total (h) | Throughput (samples/s) |
|--------|------------------------|--------------|--------------|-----------|------------------------|
| SENTRA (Docker) | **Off** (non-SGX) | 76.0 | **3.15** | 3.23 | **8.81** |
| SENTRA (WSL) | Off | 75.0 | 4.10 | 4.27 | 6.78 |
| SENTRA (SGX/Occlum) | **On** | 76.0 | 4.40 | 4.59 | 6.32 |
| CrypTen | Off | **83.0** | 4.06 | 4.07 | 6.83 |

Same hyperparameters: 10,000 train samples, 10 epochs, batch 64, LR 0.002, 100 test samples. See [§3.1](#31-main-results-with-client-profile) for distribution time, prover cost, and run IDs.

---

## 3. Performance results

### 3.1 Main results (`with-client` profile)

| System | Parties | Accuracy (%) | Distribution (s) | Training (h) | Total (h) | Throughput (samples/s) | Prover (s)† |
|--------|---------|--------------|------------------|--------------|-----------|------------------------|-------------|
| SENTRA (WSL) | 3 + client | 75.0 | 128 | 4.10 | 4.27 | 6.78 | 1,447 |
| SENTRA (Docker) | 3 + client | 76.0 | 129 | 3.15 | 3.23 | 8.81 | 1,658 |
| SENTRA (SGX/Occlum) | 3 + client | 76.0 | 220 | 4.40 | 4.59 | 6.32 | 3,191 |
| CrypTen | 4 ranks | 83.0 | 0.9‡ | 4.06 | 4.07 | 6.83 | — |

† Attributed opener work on node 3 **inside** training wall time (not additive on all nodes).  
‡ CrypTen rank-0 `data_load` + `encrypt_model`; no separate client distributor.

**Run IDs (logs)**

| System | Run ID |
|--------|--------|
| SENTRA WSL | `run_20260524_150754` |
| SENTRA Docker | `run_20260526_191039` |
| SENTRA SGX | `run_20260527_031949` (superseded: `run_20260525_204422`) |
| CrypTen | Terminal run (stdout / Run summary) |

### 3.2 Per-batch stage times (epoch 10, node 1)

| Env | fwd | dz2 | dw2 | da1/dz1 | dw1 | upd | **total/batch** |
|-----|-----|-----|-----|---------|-----|-----|-----------------|
| WSL | 2.25 s | 0.21 s | 0.05 s | 0.31 s | 1.94 s | 0.48 s | **5.23 s** |
| SGX | 8.30 s | 0.65 s | 0.08 s | 0.53 s | 3.53 s | 5.29 s | **18.36 s** |

157 batches per epoch. Forward pass and `dw1` dominate on WSL; SGX additionally inflates `upd`.

### 3.3 Training-time ratios (vs SENTRA Docker training = 11,348 s, `run_20260526_191039`)

| Comparison | Ratio |
|------------|-------|
| WSL / Docker | 1.30× (Docker faster) |
| SGX / Docker | 1.40× (Docker faster) |
| CrypTen / Docker | 1.29× (Docker faster) |
| SGX / WSL | 1.07× |
| CrypTen / WSL | 0.99× |

### 2.4 Accuracy deltas (100 test samples)

| Comparison | Δ |
|------------|---|
| CrypTen − SENTRA SGX | +7.0 pp (83% vs 76%) |
| CrypTen − SENTRA Docker | +7.0 pp |
| CrypTen − SENTRA WSL | +8.0 pp |
| SENTRA Docker − SENTRA WSL | +1.0 pp |

### 3.5 Prover breakdown (WSL, node 3)

| Sub-protocol | Time (s) |
|--------------|----------|
| `opened_divide_and_reshare_vector` | 936.6 |
| `reshare_vector` | 276.5 |
| `secure_divide_shares_batch_opened` | 71.4 |
| `softmax_opened_clip` | 68.8 |
| `secure_compare_batch_opened` | 60.3 |
| `softmax_opened_group_max` | 33.5 |
| **Total prover** | **1,447.2** |

---

## 4. Performance analysis

We evaluate SENTRA along four axes: (1) end-to-end training time and accuracy versus CrypTen under matched data splits, (2) cost of the client-centric secret-sharing pipeline, (3) overhead of TEE execution (SGX), and (4) per-batch breakdown of secure forward/backward/update stages. **Table 3** highlights that SENTRA without confidential computing (Docker non-SGX) is fastest on training wall clock; enabling SGX adds ~1.41× training time vs Docker on the same worker with unchanged accuracy (76%).

**Training time and deployment.** Dockerised non-SGX SENTRA achieves the fastest secure training among SENTRA variants (~3.15 h, `run_20260526_191039`), roughly 1.30× faster than WSL (4.10 h) and 1.40× faster than SGX (4.40 h, `run_20260527_031949`). SGX incurs large one-time enclave bootstrap (~251 s per node) and is within ~7% of WSL training wall time on the same protocol. CrypTen training (4.06 h) is within 1% of WSL SENTRA but ~29% slower than Docker SENTRA on the same worker. Superseded SGX: `run_20260525_204422` (7.90 h, per-batch inset only).

**Accuracy.** Final client-reconstructed accuracy is 76% for Docker and SGX SENTRA vs 75% on WSL, compared with 83% for CrypTen on the same 100-sample test subset. The gap reflects different loss paths (SENTRA secure fixed-point softmax vs CrypTen encrypted cross-entropy); nodes never receive plaintext labels in with-client mode.

**Client distribution.** SENTRA pays ~2.1–3.7 minutes upfront to secret-share 10,000 training and 100 test samples. CrypTen amortises data placement into sub-second rank-0 load/encrypt; SENTRA distribution should be reported as additional pipeline cost in cross-system comparisons.

**Post-training evaluation.** Share-wait time varies by environment (~484 s WSL vs ~9.7 s Docker vs ~3.9 s SGX); reconstruction remains sub-millisecond. Client `eval` wall time includes waiting for training to finish and must not be interpreted as inference latency alone.

---

## 5. Scalability analysis

**Sample and epoch scaling.** A historical 3-node recovery smoke workload (512 samples, 2 epochs) completed training in ~229 s (~4.5 samples/s), but its configuration is no longer included. The current `quick.yaml` profile uses 256 samples for 1 epoch; the current `with-client-fault-5node.yaml` profile uses 512 samples for 5 epochs.

**Party geometry.** SENTRA uses 3 MPC nodes plus an out-of-band client; CrypTen uses 4 world-size ranks with MNIST on rank 0. Party counts are comparable, but trust and data placement differ: SENTRA never ships plaintext training data to nodes.

**Memory and communication.** Peak RSS and per-iteration communication are exported via `runtime_metrics` (`rss_mb_peak`, `comm_mb_per_iter`). Lazy packed-secret unpacking on WSL adds measurable host-side overhead (~786 s cumulative unpack wait on node 1 for 10,000 training rows).

**TEE trade-off.** SGX provides memory-isolated execution at ~1.41× training time vs Docker non-SGX and ~4.2 minutes cold start per node—dominant when attested compute is required.

---

## 6. Limitations

1. **Training time** — 3–8 hours for 10 epochs on 10,000 samples; prover-centric openings are costly, especially in SGX.
3. **Accuracy vs CrypTen** — 7–8 percentage points lower on the same test slice; comparisons must separate protocol fidelity from framework performance.
4. **Model scale** — Single MLP (~0.10M parameters); deeper models and larger batches need further study.
5. **Statistical reporting** — Primary rows are single successful runs per configuration; repeat runs with confidence intervals are recommended before final submission.
6. **Fault tolerance** — Recovery and versioning are instrumented; systematic churn studies (multiple dropouts, stragglers) remain future work.

---

## 7. LaTeX source (copy into thesis)

Paste the block below into your dissertation `.tex` file. Use `\usepackage{amssymb}` or `pifont` for `\checkmark`. Map `\cite{knott2021crypten}` to your CrypTen reference (e.g. `[27]`). Add TruShare to the bibliography for Table 2.

```latex
\subsection{System Implementation}
\subsubsection{Software}
We implemented SENTRA as a multi-party secure training stack in which a data-owning \textbf{client} secret-shares MNIST features and labels to $n{=}3$ compute nodes using packed Shamir secret sharing with reconstruction threshold $t{=}1$. Training is orchestrated by \texttt{start\_all\_nodes.py} and executed in batched form via \texttt{run\_mnist\_batched\_secure.py}, with inter-node coordination over a versioned key--value store and synchronised barriers in \texttt{secure\_comm.py}.

Linear layers use Beaver-triple-based multiplication on secret-shared tensors; non-linearities employ secure comparison (ReLU) and a fixed-point secure softmax path. Exponential terms in softmax are approximated with a Pad\'{e}$(2,2)$ expansion (\texttt{exp-approx: pade22}), and gradients follow the protocol-compliant \texttt{secure\_approx} mode unless an ablation explicitly enables \texttt{opened\_exact}. All tensor values are mapped to a $61$-bit prime field (\texttt{field-size} $= 2^{61}-1$) with fixed-point scaling (\texttt{scale-factor} $= 2^{16}$), softmax temperature $2.0$, and gradient/logit clipping for numerical stability. One designated \textbf{prover} node (node~3) performs selective openings and re-sharing required by division, comparison, and softmax sub-protocols; prover time is logged separately from wall-clock training time.

For deployment studies we compare: (i)~native multi-process runs under WSL, (ii)~Dockerised non-SGX workers, (iii)~Intel SGX enclaves via Occlum, and (iv)~CrypTen as an MPC baseline on the same MNIST row indices and hyperparameters. Optional extensions include per-epoch weight versioning, failure detection with dropout re-share recovery, and runtime instrumentation reporting communication volume, CPU utilisation, and peak RSS.

\subsubsection{Models and workloads}
Our primary evaluation uses a two-layer MLP on MNIST:
\[
784 \rightarrow 128 \rightarrow \mathrm{ReLU} \rightarrow 10 \rightarrow \mathrm{softmax},
\]
with approximately \textbf{0.10M} trainable parameters ($784{\times}128 + 128 + 128{\times}10 + 10$). Unless stated otherwise, experiments use the \textbf{with-client} profile: $10{,}000$ training samples, batch size $64$ (157 mini-batches per epoch), learning rate $0.002$, $10$ epochs, seed $2026$, and client-side evaluation on $100$ held-out test samples after training. Train/test indices are fixed and shared with the CrypTen baseline.

We additionally report:
\begin{enumerate}[leftmargin = *]
    \item \textbf{Quick smoke:} $256$ samples, $1$ epoch (node-owned-data sanity check).
    \item \textbf{Fault recovery:} $512$ samples, $5$ epochs, $5$ nodes (dropout and runtime-metrics experiment).
    \item \textbf{Sweep default:} $10{,}000$ samples, $8$ epochs, learning rate $0.003$ (historical cross-framework comparison on the worker host).
\end{enumerate}

\subsubsection{Hardware}
End-to-end benchmarks were executed on a Linux worker (\texttt{sentra-worker-3}) for Docker non-SGX, SGX (Occlum), and CrypTen, and on a WSL development host for native SENTRA runs. SGX experiments require Intel SGX with the Occlum runtime inside Docker. Our Python implementation is predominantly single-threaded per process; each party runs one worker process. We report wall-clock times from \texttt{[BENCHMARK]} log lines.


\subsection{Performance}

We evaluate SENTRA along four axes: (1)~end-to-end training time and accuracy versus CrypTen under matched data splits, (2)~cost of the client-centric secret-sharing pipeline, (3)~overhead of TEE execution (SGX), and (4)~per-batch breakdown of secure forward/backward/update stages.

\subsubsection{TEE deployment scope and positioning}
The committee selection protocol assumes diversity across hardware platforms, operators, and physical machines. Our implementation focuses only on Intel SGX deployed on our own infrastructure, without extracting physical machine identifiers from attestation reports. This simplified model demonstrates feasibility on Intel SGX; extending it to other TEEs such as AMD SEV-SNP is outside the scope of this work. Operator and machine identifiers are assumed to come from an external manifest layer. SENTRA uses TEEs to secure both storage and computation. To the best of our knowledge, it is the first distributed training system combining collective TEE attestation, epoch-bound membership, and DPSS resharing to tolerate node churn under packed-MPC constraints. Table~\ref{tab:sentra-comparison} compares SENTRA with CrypTen and TruShare. Unlike CrypTen and TruShare, SENTRA integrates TEE isolation, packed MPC, DPSS resharing, and versioned storage semantics to provide rollback protection, dynamic membership, and resilient privacy-preserving distributed training. Compared to CrypTen~\cite{knott2021crypten}, which assumes static MPC participants, SENTRA adds infrastructure-aware protections against rollback attacks, stale membership reuse, and node churn failures, extending secure MPC training into a resilient cloud-native confidential computing service. Table~\ref{tab:sentra-perf} compares the performance of SENTRA and CrypTen, including SENTRA without confidential computing enabled.

\begin{table}[t]
\centering
\caption{Comparison of SENTRA with CrypTen and TruShare.}
\label{tab:sentra-comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Capability} & \textbf{SENTRA} & \textbf{CrypTen} & \textbf{TruShare} \\
\midrule
TEE isolation (Intel SGX) & \checkmark & -- & -- \\
Packed / SIMD-style MPC & \checkmark & Partial & \checkmark \\
DPSS resharing (node churn) & \checkmark & -- & -- \\
Versioned storage (rollback protection) & \checkmark & -- & -- \\
Epoch-bound membership & \checkmark & -- & -- \\
Collective TEE attestation & \checkmark$^\dagger$ & -- & -- \\
Client-side data ownership & \checkmark & -- & Varies \\
Dynamic membership / fault recovery & \checkmark & -- & -- \\
Static MPC participant set & -- & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
\vspace{0.5em}
{\footnotesize $^\dagger$Homogeneous Intel SGX in our evaluation; operator/machine IDs from external manifest.}
\end{table}

\subsubsection{Performance Analysis}
Table~\ref{tab:sentra-perf} summarises the primary \textbf{with-client} thesis profile. \emph{Training} is secure MPC wall time on nodes; \emph{Distribution} is client PSS upload plus node dataset receive; \emph{Total} is client-logged end-to-end time. Throughput is $(\text{train samples} \times \text{epochs}) / \text{training time}$.

\begin{table*}[t]
\centering
\caption{Performance of SENTRA and CrypTen (\textbf{with-client} profile: $10{,}000$ train samples, $10$ epochs, batch $64$, LR $0.002$, $100$ test samples). SENTRA (Docker) runs without confidential computing; SENTRA (SGX) enables Intel SGX via Occlum.}
\label{tab:sentra-perf}
\begin{tabular}{lccccccc}
\toprule
\textbf{System} & \textbf{CC} & \textbf{Parties} & \textbf{Acc.\ (\%)} & \textbf{Distrib.\ (s)} & \textbf{Train (h)} & \textbf{Total (h)} & \textbf{Throughput} \\
\midrule
SENTRA (Docker)     & Off & $3{+}$client & 76.0 & 129  & \textbf{3.15} & \textbf{3.23} & \textbf{8.81} \\
SENTRA (WSL)        & Off & $3{+}$client & 75.0 & 128  & 4.10 & 4.27 & 6.78 \\
SENTRA (SGX/Occlum) & On  & $3{+}$client & 76.0 & 220  & 4.40 & 4.59 & 6.32 \\
\midrule
CrypTen             & Off & $4$ ranks    & \textbf{83.0} & 0.9$^\ddagger$ & 4.06 & 4.07 & 6.83 \\
\bottomrule
\end{tabular}
\vspace{0.5em}

{\footnotesize CC = confidential computing. Throughput in samples/s. \\
$^\ddagger$CrypTen rank-0 \texttt{data\_load} + \texttt{encrypt\_model}; no separate client distributor.}
\end{table*}

Dockerised non-SGX SENTRA achieves the fastest secure training among SENTRA variants ($\approx 3.15$\,h, \texttt{run\_20260526\_191039}), roughly $1.30\times$ faster than WSL ($4.10$\,h) and $1.40\times$ faster than SGX ($4.40$\,h, run \texttt{run\_20260527\_031949}). SGX incurs a large one-time enclave bootstrap ($\approx 251$\,s per node) and is within $\approx 7\%$ of WSL training wall time. CrypTen training ($4.06$\,h) is within $1\%$ of WSL SENTRA training time but $\approx 29\%$ slower than Docker SENTRA on the same worker hardware.

Final client-reconstructed accuracy is \textbf{76\%} for Docker and SGX SENTRA vs.\ \textbf{75\%} on WSL, compared with \textbf{83\%} for CrypTen on the same $100$-sample test subset. The gap reflects different loss paths (SENTRA secure fixed-point softmax vs.\ CrypTen encrypted cross-entropy) rather than label leakage: nodes never receive plaintext labels in with-client mode.

SENTRA pays an explicit upfront cost of $\approx 2.1$--$3.7$\,minutes to secret-share $10{,}000$ training and $100$ test samples to three nodes. CrypTen amortises data placement into sub-second rank-0 load/encrypt; for fair reporting, SENTRA distribution should be counted as additional pipeline cost in cross-system comparisons.

\begin{table}[t]
\centering
\caption{Per-batch secure stage times at epoch 10 (node 1, batch size 64).}
\label{tab:sentra-stages}
\begin{tabular}{lcccccc|c}
\toprule
\textbf{Env} & \textbf{fwd} & \textbf{dz2} & \textbf{dw2} & \textbf{da1/dz1} & \textbf{dw1} & \textbf{upd} & \textbf{total} \\
\midrule
WSL  & 2.25\,s & 0.21\,s & 0.05\,s & 0.31\,s & 1.94\,s & 0.48\,s & 5.23\,s \\
SGX  & 8.30\,s & 0.65\,s & 0.08\,s & 0.53\,s & 3.53\,s & 5.29\,s & 18.36\,s \\
\bottomrule
\end{tabular}
\end{table}

On WSL, node~3 records $\approx 1{,}447$\,s prover work ($\approx 9.8\%$ of training wall time), dominated by \texttt{opened\_divide\_and\_reshare\_vector} and \texttt{reshare\_vector}. Under Docker (\texttt{run\_20260526\_191039}), prover time is $\approx 1{,}658$\,s ($\approx 14.6\%$ of training). Under SGX (\texttt{run\_20260527\_031949}), prover time is $\approx 3{,}191$\,s ($\approx 20.2\%$ of training). After training, share-wait time is environment-dependent ($\approx 484$\,s WSL vs.\ $\approx 9.4$\,s Docker vs.\ $\approx 3.9$\,s SGX); reconstruction itself remains sub-millisecond.


\subsubsection{Scalability Analysis}
The current quick-benchmark profile uses $256$ samples for $1$ epoch and is intended only as a node-owned-data sanity check. The current fault-recovery profile uses $512$ samples for $5$ epochs across $5$ nodes and is likewise not representative of full MNIST subset learning. SENTRA uses $3$ MPC nodes plus an out-of-band client; CrypTen runs $4$ world-size ranks with MNIST resident on rank~0. Peak RSS and per-iteration communication are exported via \texttt{runtime\_metrics}. SGX provides memory-isolated execution at the cost of $\approx 1.41\times$ training time vs.\ Docker non-SGX and $\approx 4.2$\,minutes cold start per node.


\subsubsection{Limitations}
Our evaluation demonstrates that a client-driven, secret-shared training pipeline can reach $\approx 76\%$ MNIST accuracy under a full secure softmax protocol, with reproducible benchmarking against CrypTen on identical sample indices. Several limitations remain. First, committee diversity and physical-machine binding are not fully realised in the prototype: we deploy homogeneous Intel SGX without parsing machine identifiers from attestation quotes. Second, absolute training time is high ($3$--$8$\,hours for $10$ epochs on $10{,}000$ samples), and prover-centric openings constitute a substantial fraction of cost, especially in SGX. Third, CrypTen achieves higher accuracy ($83\%$) on the same test slice, so cross-framework comparisons must separate protocol fidelity from raw MPC framework performance. Fourth, we primarily study a single MLP scale ($\approx 0.10$M parameters). Fifth, variance across repeated runs is not yet fully tabulated with confidence intervals on all platforms. Finally, while failure recovery and versioning hooks are instrumented, a systematic study of degradation under sustained churn is left to future work.
```

---

## 5-node dropout recovery (Docker, `with-client-fault-5node.yaml`)

Controlled fault experiment: kill MPC node 3 mid-epoch-1 training (`t=1`, `failure_timeout=20s`, batched Lagrange chunks of 8192).

**Primary successful run:** `run_20260527_042846` (image `5bcc4e91…`, `--no-cache` build with eval-barrier fix).

| Step | Evidence (all survivors 1, 2, 4, 5) |
|------|-------------------------------------|
| Failure detection | `Detected 1 failed node(s): {3}` |
| Lagrange reshare | `101770 scalars in 13 chunk(s)`; progress 13/13 |
| Resume | `Dropout weight reshare complete`; `Resumed after dropout recovery (n_active=4, MPC peers=[1, 2, 4, 5])` |
| Post-recovery training | Epoch 1 batches 5–8; epoch-1 eval; **Epoch 2 batches 1+** (no `No connection to node 3`) |

**Earlier partial run:** `run_20260527_041355` — recovery + batches 5–8 only; epoch-1 eval failed on SYNC to dead node 3 (fixed by `SecureMPCNetwork.barrier()` and `MPCReconstructionManager._required_peer_ids()` honouring `set_active_peers()`).

**Reproduce:**

```bash
cd benchmarking/ml-benchmark
docker compose build --no-cache ml-benchmark
make run-with-client-fault-5node
# Second terminal: kill node 3 after "Epoch 1 Batch 2 completed"
export CID=$(docker ps -q --filter "ancestor=sentra-ml-benchmark:latest" | head -1)
docker exec "$CID" kill -9 $(docker exec "$CID" pgrep -f "run_mnist_batched_secure.py --node-id 3 " | head -1)
```

**SGX (same YAML, Occlum enclave):**

```bash
cd benchmarking/ml-benchmark
make build-ml-sgx    # rebuild if recovery/barrier fixes changed
make sgx-run-with-client-fault-5node
# Second terminal (SGX image):
export CID=$(docker ps -q --filter "ancestor=sentra-ml-benchmark-sgx:latest" | head -1)
docker exec "$CID" kill -9 $(docker exec "$CID" pgrep -f "run_mnist_batched_secure.py --node-id 3 " | head -1)
```

**Measured timings (node 1, `sentra-worker-3`):**

| Run | Training (s) | Total (s) | Recovery (s) | `success` | Throughput |
|-----|-------------:|----------:|-------------:|:---------:|-----------:|
| `run_20260527_042846` | 430.0 | 521.9 | 8.09 | 1 | 5.95 samples/s |
| `run_20260527_033821` (5-node, no kill) | 399.6 | 484.6 | — | — | 6.41 samples/s |
| `run_20260527_022635` (SGX) | 63.4 | — | 0.22 | 0 | (7 iters only) |

Prior SGX attempt `run_20260527_022635` failed recovery (`success=0`); re-run with `make sgx-run-with-client-fault-5node` after `make build-ml-sgx`. Full detail: `benchmarking/BENCHMARK_TIMING_COMPARISON.md` §14.

---

## Metrics 3 & 5 — dynamic join time and versioning overhead

## LaTeX table — measured systems metrics (CrypTen vs SENTRA)

Copy-paste into your thesis (requires `\usepackage{booktabs}` and either `\usepackage{tabularx}` or replace `tabularx` with `tabular`).

```latex
\begin{table*}[t]
\centering
\caption{Measured systems metrics for CrypTen vs SENTRA (\texttt{with-client}: 10{,}000 train samples, 10 epochs, batch 64). Primary runs: CrypTen terminal; SGX \texttt{run\_20260527\_031949}; Docker \texttt{run\_20260526\_191039}. $^\dagger$ fault-5node; $^\ddagger$ comm smoke proxy; $^{\ddagger\ddagger}$ versioned-run peak RSS. See \texttt{BENCHMARK\_TIMING\_COMPARISON.md} \S15.}
\label{tab:sentra-systems-metrics}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric (measured)} & \textbf{CrypTen} & \textbf{SENTRA SGX} & \textbf{SENTRA non-SGX} \\
\midrule
Training throughput (samples/sec)      & 6.83  & 6.32  & 8.81 \\
Per-iteration latency (s)              & 9.32  & 10.08 & 7.23 \\
Per-iteration latency (ms)             & 9321  & 10081 & 7228 \\
Communication volume / iteration (MB)  & N/A   & --    & 24.00$^\ddagger$ \\
Peak memory usage (GB)                 & --    & --    & 3.26$^{\ddagger\ddagger}$ \\
Failure recovery time (sec)            & --    & 0.22$^\dagger$ & 8.09$^\dagger$ \\
Training progress loss after failure   & --    & --    & -- \\
Dynamic node join time (sec)           & --    & --    & -- \\
Version enforcement overhead (\%)      & N/A   & N/A   & 0.0004 \\
Performance degradation under dropout (\%) & Abort & 20$^\dagger$ & 20$^\dagger$ \\
Final accuracy (\%)                    & 83    & 76    & 76 \\
Client / node cold-start (s)           & N/A   & 251   & 1.04 \\
Client PSS distribution (s)            & N/A   & 220   & 129 \\
Secure training (h)                    & 4.06  & 4.40  & 3.15 \\
Logged end-to-end (h)                  & 4.07  & 4.59  & 3.23 \\
\bottomrule
\end{tabular}
\end{table*}
```

### 5 — Version enforcement overhead (%)

**Not in the main Table 3 runs** — requires a paired comparison on the same worker:

| Run | Config | Run ID (example) |
|-----|--------|-------------------|
| Baseline | `with-client.yaml` | `run_20260526_191039` (Docker, no versioning) |
| Versioned | `with-client-versioned.yaml` | Worker terminal 2026-05-27 (sync run ID to volume when available) |

**Measured (worker, 2026-05-27):**

| Metric | Baseline `run_20260526_191039` | Versioned `run_20260527_053839` |
|--------|-------------------------------:|--------------:|
| Training wall (node 1) | 11,348 s (~3.15 h) | **12,331 s (~3.43 h)** |
| `versioning_sec_total` | 0 | **0.046 s** (10 calls) |
| Instrumented overhead | — | **0.046 / 12,331 ≈ 0.0004%** |
| Training wall vs baseline | — | **+9.6%** (unpaired runs; re-run both for strict A/B) |
| `rss_mb_peak` | — | **3,342 MiB** |
| `comm_mb_per_iter` | *(from baseline run)* | **0** (wire bytes not recorded) |

```bash
cd benchmarking/ml-benchmark
make run-with-client              # baseline (~3.15 h training)
make run-with-client-versioned    # versioned (~3.43 h training on worker 2026-05-27)

make compare-versioning \
  BASELINE_RUN=/workspace/node/logs/run_20260526_191039 \
  VERSIONED_RUN=/workspace/node/logs/run_YYYYMMDD_HHMMSS
```

Report **`versioning_sec_total / training_wall_sec × 100`** for instrumented KVS persist cost; report **paired training wall** only when both runs use the same image on the same host. CrypTen: **N/A**. See `BENCHMARK_TIMING_COMPARISON.md` §14.5.

### 3 — Dynamic node join time (sec)

Dynamic join remains unmeasured. The repository currently contains neither a join benchmark profile nor a join-specific reproduction procedure. The dropout evidence in `run_20260527_042846` must not be reported as a join result; add a maintained join configuration and capture a successful `recovery_join` run before filling this metric.

---

## Reproducing log extraction

```bash
# WSL
grep '\[BENCHMARK\]' sentra-node/python/node/logs/run_20260524_150754/*.log

# Docker worker (SGX example)
RUN=run_20260527_031949
docker compose -f benchmarking/ml-benchmark/docker-compose.yml run --rm --entrypoint bash ml-benchmark -c \
  "grep '\[BENCHMARK\]' /workspace/node/logs/${RUN}/*.log"

# CrypTen
cd benchmarking/crypten-benchmark && make run 2>&1 | tee crypten_run.log
grep -E '\[BENCHMARK\]|Run summary' crypten_run.log
```

**Config paths**

- SENTRA: `benchmarking/ml-benchmark/configs/with-client.yaml`
- Versioned: `benchmarking/ml-benchmark/configs/with-client-versioned.yaml`
- Fault 5-node recovery: `benchmarking/ml-benchmark/configs/with-client-fault-5node.yaml`
- CrypTen: `benchmarking/crypten-benchmark/configs/sentra-with-client.yaml`

---

*Generated for dissertation reporting. Update hardware subsection and add $\pm$ standard deviations after additional repeated runs.*
