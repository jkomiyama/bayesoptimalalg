import Junpei.InitialOptimalPaths

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory

namespace Junpei.HistoryPolicy

open Junpei.FiniteGaussian Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open Junpei.PolicyContinuation

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

/-- The observable history records every chosen arm, its actual reward, and
the randomizer used for that decision. Past randomization is available to
later decisions; no unrevealed reward or latent mean is available. -/
abbrev History (A : Type*) (n : ℕ) := Fin n → A × (ℝ × ℝ)

/-- A general Borel, finite-history adaptive sampling algorithm. Its rules
can use the complete observation and randomization history, not merely a
posterior state. The eventual horizon is fixed by the interpreter. -/
structure Policy (A : Type*) [MeasurableSpace A] where
  draw : (n : ℕ) → History A n → ℝ → A
  recommend : (n : ℕ) → History A n → ℝ → A
  draw_measurable : ∀ n, Measurable (fun p : History A n × ℝ => draw n p.1 p.2)
  recommend_measurable : ∀ n,
    Measurable (fun p : History A n × ℝ => recommend n p.1 p.2)

def append {n : ℕ} (h : History A n) (a : A) (y u : ℝ) : History A (n + 1) :=
  Fin.snoc h (a, (y, u))

theorem append_measurable (n : ℕ) :
    Measurable (fun p : History A n × (A × (ℝ × ℝ)) =>
      append p.1 p.2.1 p.2.2.1 p.2.2.2) := by
  apply measurable_pi_lambda
  intro j
  refine Fin.lastCases ?_ (fun i => ?_) j
  · simpa only [append, Fin.snoc_last] using
      (measurable_snd : Measurable (fun p : History A n × (A × (ℝ × ℝ)) => p.2))
  · simpa only [append, Fin.snoc_castSucc] using
      (measurable_pi_apply i).comp
        (measurable_fst : Measurable (fun p : History A n × (A × (ℝ × ℝ)) => p.1))

/-- Explicit compilation of a history algorithm into a raw-observation
strategy. Exactly one selected raw reward is supplied at each sample node. -/
def toRaw (π : Policy A) (n : ℕ) : (H : ℕ) → History A n → Seeds H → Strategy (A := A) H
  | 0, h, u => .stop (π.recommend n h (u ⟨0, by omega⟩))
  | H + 1, h, u =>
      let a := π.draw n h (u ⟨0, by omega⟩)
      .sample a (fun y => toRaw π (n + 1) H
        (append h a y (u ⟨0, by omega⟩)) (seedTail u))

abbrev Input (A : Type*) (n H : ℕ) := History A n × (Seeds H × RawStream A H)

/-- Compilation is jointly Borel in the initial history, all fresh seeds,
and the entire raw calendar array. This is proved from the policy rules. -/
theorem toRaw_run_measurable (π : Policy A) (n H : ℕ) :
    Measurable (fun p : Input A n H => run (toRaw π n H p.1 p.2.1) p.2.2) := by
  induction H generalizing n with
  | zero =>
      exact (π.recommend_measurable n).comp (measurable_fst.prodMk
        ((measurable_pi_apply (⟨0, by omega⟩ : Fin 1)).comp
          (measurable_fst.comp measurable_snd)))
  | succ H ih =>
      let g : Input A n (H + 1) × A → A := fun p =>
        run (toRaw π (n + 1) H
          (append p.1.1 p.2 (p.1.2.2 p.2 ⟨0, by omega⟩)
            (p.1.2.1 ⟨0, by omega⟩)) (seedTail p.1.2.1)) (tail p.1.2.2)
      have hg : Measurable g := by
        apply measurable_from_prod_countable_left
        intro a
        have hh : Measurable (fun p : Input A n (H + 1) =>
            append p.1 a (p.2.2 a ⟨0, by omega⟩) (p.2.1 ⟨0, by omega⟩)) :=
          (append_measurable n).comp (show Measurable (fun p : Input A n (H + 1) =>
            (p.1, (a, (p.2.2 a ⟨0, by omega⟩, p.2.1 ⟨0, by omega⟩)))) by fun_prop)
        have ht : Measurable (fun p : Input A n (H + 1) =>
            (append p.1 a (p.2.2 a ⟨0, by omega⟩) (p.2.1 ⟨0, by omega⟩),
              (seedTail p.2.1, tail p.2.2))) := by
          apply hh.prodMk
          unfold seedTail tail
          fun_prop
        exact (ih (n + 1)).comp ht
      have ha : Measurable (fun p : Input A n (H + 1) =>
          π.draw n p.1 (p.2.1 ⟨0, by omega⟩)) :=
        (π.draw_measurable n).comp (measurable_fst.prodMk
          ((measurable_pi_apply (⟨0, by omega⟩ : Fin (H + 2))).comp
            (measurable_fst.comp measurable_snd)))
      exact hg.comp (measurable_id.prodMk ha)

/-- Every fixed-seed compiled policy satisfies the raw strategy's recursive
regularity requirements. No regularity assumption is added to the policy. -/
theorem toRaw_regular (π : Policy A) (n H : ℕ) (h : History A n) (u : Seeds H) :
    Regular (toRaw π n H h u) := by
  induction H generalizing n with
  | zero => trivial
  | succ H ih =>
      constructor
      · intro y
        exact ih (n + 1) _ (seedTail u)
      · let a := π.draw n h (u ⟨0, by omega⟩)
        have hh : Measurable (fun p : ℝ × RawStream A H =>
            append h a p.1 (u ⟨0, by omega⟩)) :=
          (append_measurable n).comp (show Measurable (fun p : ℝ × RawStream A H =>
            (h, (a, (p.1, u ⟨0, by omega⟩)))) by fun_prop)
        exact (toRaw_run_measurable π (n + 1) H).comp
          (hh.prodMk (measurable_const.prodMk measurable_snd))

theorem toRaw_budget (π : Policy A) (n H : ℕ) (h : History A n)
    (u : Seeds H) (r : RawStream A H) :
    (trace (toRaw π n H h u) r).length = H := trace_budget _ _

/-- Direct history semantics, independently of the raw-strategy datatype. -/
def recommendation (π : Policy A) (n : ℕ) : (H : ℕ) →
    History A n → Seeds H → RawStream A H → A
  | 0, h, u, _ => π.recommend n h (u ⟨0, by omega⟩)
  | H + 1, h, u, r =>
      let a := π.draw n h (u ⟨0, by omega⟩)
      recommendation π (n + 1) H
        (append h a (r a ⟨0, by omega⟩) (u ⟨0, by omega⟩)) (seedTail u) (tail r)

def sampledArms (π : Policy A) (n : ℕ) : (H : ℕ) →
    History A n → Seeds H → RawStream A H → List A
  | 0, _, _, _ => []
  | H + 1, h, u, r =>
      let a := π.draw n h (u ⟨0, by omega⟩)
      a :: sampledArms π (n + 1) H
        (append h a (r a ⟨0, by omega⟩) (u ⟨0, by omega⟩)) (seedTail u) (tail r)

/-- Pointwise semantic equality, with no probabilistic equivalence premise. -/
theorem toRaw_run_eq (π : Policy A) (n H : ℕ) (h : History A n)
    (u : Seeds H) (r : RawStream A H) :
    run (toRaw π n H h u) r = recommendation π n H h u r := by
  induction H generalizing n with
  | zero => rfl
  | succ H ih =>
      exact ih (n + 1) _ (seedTail u) (tail r)

theorem toRaw_trace_eq (π : Policy A) (n H : ℕ) (h : History A n)
    (u : Seeds H) (r : RawStream A H) :
    trace (toRaw π n H h u) r = sampledArms π n H h u r := by
  induction H generalizing n with
  | zero => rfl
  | succ H ih =>
      exact congrArg (List.cons (π.draw n h (u ⟨0, by omega⟩)))
        (ih (n + 1) _ (seedTail u) (tail r))

/-- Actual latent Gaussian/noise execution of every Borel history algorithm
therefore has the proved Bellman lower bound. This includes arbitrary use
of observed histories and past randomizers. -/
theorem history_risk_ge_loss (π : Policy A) (n H : ℕ) (h : History A n)
    (u : Seeds H) (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a) :
    Junpei.FinitePosterior.loss H c m ≤
      ∫ w, maxAll w.1 - w.1 (recommendation π n H h u (observations w))
        ∂worldLaw H c m := by
  simpa only [rawRisk, toRaw_run_eq] using
    rawRisk_ge_loss (toRaw π n H h u) c m hc (toRaw_regular π n H h u)

theorem toRaw_family_measurable (π : Policy A) (n H : ℕ) (h : History A n) :
    Measurable (fun p : Seeds H × RawStream A H => run (toRaw π n H h p.1) p.2) :=
  (toRaw_run_measurable π n H).comp (measurable_const.prodMk measurable_id)

/-- Independent fresh randomization does not change the exact encoding:
the history algorithm is a feasible randomized raw-strategy family. -/
theorem toRaw_family_feasible (π : Policy Draft0.BayesianModel.Arm)
    (n H : ℕ) (h : History Draft0.BayesianModel.Arm n) :
    Junpei.SequentialRisk.Feasible (fun u => toRaw π n H h u) := by
  exact ⟨fun u => toRaw_regular π n H h u, toRaw_family_measurable π n H h⟩

def actualRisk (π : Policy A) (n H : ℕ) (h : History A n)
    (c : Precisions A) (m : Means A) : ℝ :=
  ∫ u, ∫ w, maxAll w.1 - w.1 (recommendation π n H h u (observations w))
    ∂worldLaw H c m ∂Junpei.Policy.seedLaw (H + 1)

theorem randomized_history_risk_ge_loss (π : Policy A) (n H : ℕ) (h : History A n)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a) :
    Junpei.FinitePosterior.loss H c m ≤ actualRisk π n H h c m := by
  simpa only [actualRisk, rawRisk, toRaw_run_eq] using
    Junpei.EmbeddingRandomized.randomized_rawRisk_ge_loss
      (Junpei.Policy.seedLaw (H + 1)) (fun u => toRaw π n H h u) c m hc
      (fun u => toRaw_regular π n H h u) (toRaw_family_measurable π n H h)

theorem familyRisk_eq_actualRisk (π : Policy Draft0.BayesianModel.Arm)
    (n H : ℕ) (h : History Draft0.BayesianModel.Arm n)
    (c : Draft0.BayesianModel.Counts) (m : Draft0.BayesianModel.Means) :
    Junpei.SequentialRisk.familyRisk (fun u => toRaw π n H h u) c m =
      actualRisk π n H h (Junpei.CountModelBridge.natPrecisions c) m := by
  simp only [Junpei.SequentialRisk.familyRisk, rawRisk, toRaw_run_eq, actualRisk]

/-- The initialization-only optimality premise competes against every
Borel full-history algorithm above, rather than just posterior-state rules.
This is an actual-risk inequality, using the explicit encoding theorem. -/
theorem minimizing_actualRisk_le_history (π : Junpei.Policy.StatePolicy)
    (t H : ℕ) (c : Draft0.BayesianModel.Counts) (m : Draft0.BayesianModel.Means)
    (hπ : Junpei.SequentialRisk.MinimizesActualRisk π t H c m)
    (ρ : Policy Draft0.BayesianModel.Arm) (n : ℕ)
    (h : History Draft0.BayesianModel.Arm n) :
    Junpei.SequentialRisk.continuationRisk π t H c m ≤
      actualRisk ρ n H h (Junpei.CountModelBridge.natPrecisions c) m := by
  have hi := hπ (fun u => toRaw ρ n H h u) (toRaw_family_feasible ρ n H h)
  rwa [familyRisk_eq_actualRisk] at hi

end
end Junpei.HistoryPolicy
