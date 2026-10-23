# Lean proofs for the paper

This directory accompanies *Suboptimal Frequentist Performance of the Flat-Prior Generalized Bayes Optimal Algorithm in Best Arm Identification*. All project definitions, proofs, and the axiom audit are in [LeanProofs.lean](LeanProofs.lean). The remaining files make that source reproducible and connect it to the paper.

The formalization includes counterparts of the paper's 18 numbered proved results: Theorems 1–2 and Lemmas 3–8 and 10–19. It also proves the corrected Gaussian-tail equivalence in Footnote 14. Conjecture 9 remains open. The [paper-to-proof map](PAPER_MAP.md) records the exact scope and the manuscript arguments that are not separately certified.

## Run the verification

Install Git and Lean's toolchain manager, **elan**, using the [official installation guide](https://lean-lang.org/install/). Then run these commands in a terminal; they work in PowerShell and in a Unix shell:

```sh
git clone https://github.com/jkomiyama/bayesoptimalalg.git
cd "bayesoptimalalg/Lean Proofs"
lake exe cache get
lake build
```

If you already cloned this repository, start by entering its `Lean Proofs` directory. Run the commands from the directory containing `lakefile.toml`.

The project selects **Lean 4.24.0** and pins mathlib to commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. Elan obtains the selected toolchain; `lake exe cache get` downloads compiled mathlib dependencies. The first setup needs an internet connection and may take several minutes. No Python script is needed to verify this project.

`lake build` checks the source when it has changed. To check the complete source again even after a successful build, run:

```sh
lake env lean LeanProofs.lean
```

This command reprocesses every project proof in the single source file, using the compiled mathlib dependencies. A successful run exits with code **0** and includes this final audit output:

```text
'BayesOptimal.allPublicTheorems' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The source's guarded audit references all **1,517 public theorem/lemma declarations**. It makes verification fail if the aggregate axiom report differs from the three standard Lean axioms shown above. It therefore also rejects an admitted proof dependency such as `sorryAx`. The source contains no `sorry`, `admit`, or additional unproved mathematical axiom. Lean performs the mathematical proof checking.

If `lake` is not found, reopen the terminal after installing elan. If dependency downloads fail, check the connection and rerun `lake exe cache get`. Keep the supplied toolchain and dependency pins when reproducing this version.

## Find a result from the paper

The links below open the principal statements in the source. Each name has the common prefix `BayesOptimal.`; for example, the full name of Theorem 1 is `BayesOptimal.GeneralOptimality.FiniteHorizon.theorem1`. Supporting declarations and the relevant assumptions are listed in [PAPER_MAP.md](PAPER_MAP.md).

| Paper result | Topic | Principal Lean declaration |
|---|---|---|
| Theorem 1 | Optimal actions, recommendations, and posterior risk | [GeneralOptimality.FiniteHorizon.theorem1](LeanProofs.lean#L22955) |
| Theorem 2 | Polynomial frequentist regret and zero exponential-rate limit inferior | [RealPrecisionMain.theorem2](LeanProofs.lean#L24729), [theorem2_paper_exponent](LeanProofs.lean#L24740), [theorem2_exponential_rate](LeanProofs.lean#L24751) |
| Lemma 3 | Severe underestimation of arm 3 | [Probability.lemma3](LeanProofs.lean#L19148) |
| Lemma 4 | Small-ball probability | [Probability.lemma4](LeanProofs.lean#L17215) |
| Lemma 5 | Drift-event failure bound | [Probability.lemma5](LeanProofs.lean#L17208) |
| Lemma 6 | Upper bound for arm 3's EBI | [PolicyChoice.lemma6_upper](LeanProofs.lean#L14268) |
| Lemma 7 | Lower bounds and exclusion of saturated actions | [PolicyChoice.lemma7_individual_lower](LeanProofs.lean#L14295) |
| Lemma 8 | Final-round exclusion of arm 3 | [PolicyChoice.lemma8_individual_lower](LeanProofs.lean#L14345) |
| Conjecture 9 | Almost-every-instance exponential convergence | Open; no proof is claimed |
| Lemma 10 | Mills inequalities | [TailRatio.mills_lower](LeanProofs.lean#L19112), [mills_upper](LeanProofs.lean#L19087) |
| Lemma 11 | Gaussian bound for exclusion loss | [ExclusionBounds.lemma11](LeanProofs.lean#L13318) |
| Lemma 12 | EBI bounded by exclusion loss | [FiniteModelExclusion.lemma12_any_finite_arm](LeanProofs.lean#L8864) |
| Lemma 13 | Full/reduced policy comparison | [PolicyEmbedding.lemma13_loss_by_policy_embeddings](LeanProofs.lean#L16930) |
| Lemma 14 | Two-arm terminal-regret integral | [Marginal.lemma14_normalized](LeanProofs.lean#L13657) |
| Lemma 15 | One-step improvement and its lower bound | [Marginal.lemma15_g1_lower](LeanProofs.lean#L13701) |
| Lemma 16 | Strict smaller-count advantage and its lower bound | [Marginal.lemma16_g2_lower](LeanProofs.lean#L13749) |
| Lemma 17 | Optimal smaller-count policy for two arms | [TwoArmOptimal.lemma17_every_posterior](LeanProofs.lean#L14621) |
| Lemma 18 | Forecast probability of a small gap | [Forecast.lemma18](LeanProofs.lean#L14958) |
| Lemma 19 | Shared-sample coupling and EBI gains | [Coupling19.lemma19_individual_gains_via_coupling](LeanProofs.lean#L15760) |
| Corrected Footnote 14 | Polynomial Gaussian tails iff the logarithmic gap scale holds | [Footnote14Characterization.corrected_footnote14_iff](LeanProofs.lean#L21945) |
| Equation (26) | Actual regret as gap times error probability | [Main.regret_eq](LeanProofs.lean#L19390) |

Theorem 1 covers arbitrary finite arm sets and positive real precisions for sequentially posterior-optimal policies. Theorem 2 uses the three-arm instance `(0, 0, Δ)`, with `Δ > 0`, and initialization-only posterior-risk minimization against history-dependent competitors. Its polynomial bound holds for sufficiently large odd horizons; its exponential-rate limit inferior is over all horizons. The Footnote 14 equivalence concerns the actual Gaussian tail measure and assumes only that the deterministic gap sequence is nonnegative.

## Read the proof alongside the paper

For the main result, start at `RealPrecisionMain.theorem2` and follow the lemmas it invokes. The source is arranged so definitions precede their uses, rather than repeating the manuscript's exposition order. Use the paper map to navigate in paper order:

1. **Section 2:** Gaussian posterior states, value functions, EBI, and the optimal-policy characterization in Theorem 1.
2. **Sections 3–4:** the frequentist regret definition, initialization-only optimality, and the proof of Theorem 2.
3. **Appendices A–D:** Gaussian tail bounds, event probabilities, and the upper bound for arm 3.
4. **Appendix E.1–E.3:** policy embeddings, two-arm losses, smaller-count optimality, forecast probabilities, and shared-sample coupling.
5. **Appendix G.4 and corrected Footnote 14:** Gaussian standardization and both directions of the asymptotic equivalence.

In a declaration, the parameters before the colon are assumptions, the expression after the colon is the proposition, and the block after `:= by` is the proof. A `#check` command displays a statement's type; the preceding proof is checked when Lean processes its declaration.

| Lean notation | Meaning in the paper |
|---|---|
| `∀ᵐ x ∂μ, ...` | For almost every `x` under measure `μ` |
| `∀ᶠ T in Filter.atTop, ...` | For all sufficiently large natural horizons |
| `d =O[Filter.atTop] f` | The sequence `d` is big-O of `f` |
| `Fin 3` | Three arms, indexed `0, 1, 2`; index `2` is paper arm 3 |

In VS Code, open this `Lean Proofs` directory as the project folder and install the **Lean 4** extension. Open `LeanProofs.lean`, search for a declaration name, and use **Go to Definition** to follow its dependencies. The Lean information panel displays theorem types and intermediate proof goals. See the [Lean community project guide](https://leanprover-community.github.io/install/project.html) for editor setup.

### Appendix G and this repository

The named mathematical statements described in Appendix G.1–G.4 are retained, including their assumptions and the stated exclusions. Appendix G's file paths and G.5 reproduction commands describe the earlier modular supplement. They do not literally apply to this single-source layout; use these replacements:

| Reference in Appendix G | Location or command in this repository |
|---|---|
| Individual files such as `RealPrecisionMain.lean` | The corresponding named section of [LeanProofs.lean](LeanProofs.lean) |
| `docs/PROOF_MAP.html` | [PAPER_MAP.md](PAPER_MAP.md) |
| `BayesOptimal.optimal_action_iff_EBI_max` and `BayesOptimal.terminal_recommendation_iff` on page 42 | [BayesOptimal.DecisionRules.optimal_action_iff_EBI_max](LeanProofs.lean#L12401) and [BayesOptimal.DecisionRules.terminal_recommendation_iff](LeanProofs.lean#L12416) |
| `python3 -X utf8 verify.py --fresh` in G.5 | Run `lake exe cache get`, then `lake env lean LeanProofs.lean`; `lake build` is also available |
| `EndToEndAudit.lean` and `checks/*Audit.lean` | The paper-result signature checks and guarded aggregate axiom audit at the end of [LeanProofs.lean](LeanProofs.lean#L25581) |
| `REPORT.txt`, `verification_results.json`, and `logs/` | Read the current command's exit code and terminal output; these historical record files are not distributed in this layout |

Appendix G omits the former common outer namespace `Junpei`. The current full names begin with `BayesOptimal.`. The two decision-rule names in the table also use the inner namespace `DecisionRules`. The corrected Footnote 14 theorem and the historical smaller-scale counterexample both remain in the source. This documentation update does not change the PDF or the formal proofs.

## Files and verification scope

| File | Purpose |
|---|---|
| `LeanProofs.lean` | All project definitions, proofs, paper-result checks, and the guarded axiom audit |
| `README.md` | Setup, verification, and reading instructions |
| `PAPER_MAP.md` | Detailed correspondence with numbered results, equations, remarks, and scope limits |
| `lean-toolchain` | Exact Lean version |
| `lakefile.toml` | Build target and pinned mathlib requirement |
| `lake-manifest.json` | Locked dependency revisions |
| `.gitignore` | Excludes generated build files |

Lake creates a local `.lake/` directory for downloaded dependencies and compiled files. It is generated automatically and is not part of the proof source.

Successful verification certifies the formal statements under their written assumptions. It does not certify every sentence or displayed calculation in the manuscript. In particular, the exact triple-integral display in Equation (18), the broader arbitrary-arm initialization-repair paragraph, and the sharper explicit Gaussian constants introduced in Appendix F are not separately certified by this source. The historical counterexample to the old Footnote 14 is retained alongside the proof of its corrected statement. See [PAPER_MAP.md](PAPER_MAP.md) for these distinctions and the remaining exclusions.
