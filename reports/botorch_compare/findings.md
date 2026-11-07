# skopt gp_minimize vs BoTorch: findings

Run on 2026-11-07. Four problems from the initial-design study (Rastrigin 2D/4D,
pressure vessel, welded beam), budget 20d evaluations, 20 seeds. Every method
starts from the same Latin-hypercube design of 2d points, so only the model, the
acquisition function and its optimiser differ. Full tables: [summary.md](summary.md).
Setup: [README.md](README.md).

## Final best feasible value (median over 20 seeds, relative regret to best known)

| problem | skopt gp_minimize | in-house sklearn loop | BoTorch |
| --- | --- | --- | --- |
| Rastrigin 2D (opt 0) | 3.9 | 1.5 | 3.1 |
| Rastrigin 4D (opt 0) | 23.0 | 15.1 | 19.8 |
| Pressure vessel | 38,400 (+552 %) | 6,460 (+9.7 %) | **6,080 (+3.3 %)** |
| Welded beam | 7.29 (+323 %), feasible in only 11/20 runs | **1.96 (+13.5 %)** | 2.24 (+30 %) |
| Time per run (4D constrained) | ~16 s | ~5-9 s | ~150-230 s |

Batch of 4 BoTorch (10 seeds): Rastrigin 2D 3.5, Rastrigin 4D 13.5, pressure vessel 7,360 (+25 %), welded beam 3.94 (+129 %, 9/10 feasible).

## What it means

1. **Unconstrained problems: no real difference between skopt and BoTorch.**
   On Rastrigin the two are statistically indistinguishable (Mann-Whitney
   p = 0.78 in 2D, 0.23 in 4D). Both are a Matern-5/2 GP with EI, so this is
   expected. The in-house loop does somewhat better on Rastrigin 4D (p = 0.004),
   most likely because its cruder acquisition optimiser explores more on this
   very multimodal function.
2. **Constrained engineering problems: the big gap is constraint handling, not
   the library's GP.** gp_minimize has no constraint support, so the only option
   is a penalty, and it fails: about 6.5x the optimum on the pressure vessel, and
   no feasible design at all in 9 of 20 welded-beam runs. Both methods that model
   each constraint with its own GP (EI x probability of feasibility) get within
   3-30 % of the best known design in 80 evaluations. BoTorch does that in a few
   lines with `LogConstrainedExpectedImprovement`.
3. **BoTorch vs the hand-written loop is a split decision.** BoTorch wins on the
   pressure vessel and the hand-written loop wins on the welded beam (6
   constraints). Given 20 seeds and the tuning freedom in both, treat them as
   comparable. The point is that BoTorch gives you this without writing and
   maintaining your own loop.
4. **BoTorch is 10-30x slower per run on CPU** (about 2-3 s per iteration with 7
   GPs, mostly hyperparameter fitting). That is irrelevant when each evaluation
   is a CFD run taking minutes to hours, but noticeable on cheap test functions.
5. **Batch of 4 (`botorch_q4`, 10 seeds) trades sample efficiency for
   wall-clock time.** It proposes 4 designs per GP fit, so 80 evaluations take
   18 rounds instead of 72, which is about 4x less wall-clock time if you can run 4
   CFD cases in parallel. On the same 80 evaluations it ends worse on the
   constrained problems: pressure vessel +25 % (vs +3.3 % sequential), welded
   beam +129 % with 1 of 10 runs never feasible (vs +30 %). It still beats skopt
   on both. On Rastrigin 4D it did as well as or better than sequential
   (13.5 vs 19.8) in the cloud run, but not in the MacBook run (16.8 vs 17.5),
   so treat that as noise. If evaluations are hours long and you have the cores, q=4 is usually
   the right call. If evaluations are the bottleneck and not wall-clock, stay
   sequential.

6. **The MacBook reproduces every conclusion.** Daniel's M4 Max re-ran the full
   sweep (same seeds, 10 performance cores). Individual medians move a little
   because of floating-point differences between the platforms (for example
   skopt on Rastrigin 2D: 3.9 in the cloud vs 4.5 on the Mac), but the ranking
   and the significance tests are the same: skopt ~ BoTorch without
   constraints; skopt fails with constraints (pressure vessel +527 %, welded
   beam feasible in only 12/20 runs); BoTorch +3.5 % / +27 %; batch-of-4
   +18 % / +102 %.
7. **The M4 Max CPU is about 2.5x faster per core than the cloud sandbox.**
   BoTorch per run: 60 s vs 151 s (pressure vessel), 101 s vs 230 s (welded
   beam). The whole sweep (320 runs) takes under 10 minutes on 10 cores.
8. **The Apple GPU (MPS) is 3-10x slower than one CPU core here.** Median
   seconds per BoTorch run, CPU vs GPU (5 GPU seeds): Rastrigin 2D 2.2 vs 20.1,
   Rastrigin 4D 6.2 vs 53.6, pressure vessel 60 vs 176, welded beam 101 vs 294.
   With at most 80 training points every GPU kernel launch costs more than the
   arithmetic it saves, and MPS is limited to float32. The float32 runs also
   ended somewhat worse (welded beam +56 % vs +27 %, from 5 seeds). GPUs start to pay
   off with thousands of data points, many parallel models, or neural-network
   surrogates. That's the regime of data-driven CFD surrogates, not a
   100-evaluation GP loop.

## Mac run (M4 Max, 10 workers): final medians

| problem | skopt | sklearn loop | BoTorch | BoTorch q=4 | BoTorch MPS (5 seeds) |
| --- | --- | --- | --- | --- | --- |
| Rastrigin 2D | 4.53 | 1.45 | 4.58 | 4.76 | 7.82 |
| Rastrigin 4D | 17.5 | 15.1 | 17.5 | 16.8 | 13.9 |
| Pressure vessel | 36,900 (+527 %) | 6,460 (+9.7 %) | 6,090 (+3.5 %) | 6,930 (+18 %) | 6,330 (+7.6 %) |
| Welded beam | 6.53 (+279 %), 12/20 feasible | 1.97 (+14 %) | 2.20 (+27 %) | 3.48 (+102 %) | 2.69 (+56 %) |

The cloud tables and plots are in [summary.md](summary.md). The Mac's copy is in
the repo folder on Daniel's MacBook, under reports/botorch_compare/.

## Caveat

The skopt penalty (worst-seen value + 10 x violation) is one reasonable choice
among many. A better-tuned penalty would close some of the gap, but a penalty
can never use the constraint information the way a constraint model does.
