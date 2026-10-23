import Junpei.RealPrecisionPolicy

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSectionVars false
set_option maxHeartbeats 1500000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.GeneralOptimality

open Junpei.FiniteGaussian Junpei.FinitePosterior
open Junpei.PolicyContinuation Junpei.EmbeddingRaw Junpei.EmbeddingExecution

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

/-- The conjugate update on the full real-precision Gaussian state space,
using only the selected arm and its observed reward. -/
def updateState (a : A) (p : (Precisions A × Means A) × ℝ) : Precisions A × Means A :=
  (advance p.1.1 a, shift p.1.2 a ((p.2 - p.1.2 a) / ((p.1.1 a : ℝ) + 1)))

theorem updateState_measurable (a : A) : Measurable (updateState a) := by
  apply Measurable.prodMk
  · exact (JointGaussianMeasurable.advance_measurable a).comp
      (measurable_fst.comp measurable_fst)
  · exact (JointGaussianMeasurable.shift_joint_measurable a).comp
      ((measurable_snd.comp measurable_fst).prodMk (by fun_prop))

theorem updateState_joint_measurable :
    Measurable (fun p : A × ((Precisions A × Means A) × ℝ) => updateState p.1 p.2) :=
  measurable_from_prod_countable_right updateState_measurable

/-- Reconstruct the posterior from the observed finite history. Past
randomizers are observable but do not enter the Gaussian update. -/
def foldState (c0 : Precisions A) (m0 : Means A) : (n : ℕ) →
    HistoryPolicy.History A n → Precisions A × Means A
  | 0, _ => (c0, m0)
  | n + 1, h =>
      let p := foldState c0 m0 n (Fin.init h)
      let z := h (Fin.last n)
      updateState z.1 (p, z.2.1)

theorem foldState_append (c0 : Precisions A) (m0 : Means A) (n : ℕ)
    (h : HistoryPolicy.History A n) (a : A) (y u : ℝ) :
    foldState c0 m0 (n + 1) (HistoryPolicy.append h a y u) =
      updateState a (foldState c0 m0 n h, y) := by
  simp only [foldState, HistoryPolicy.append, Fin.init_snoc, Fin.snoc_last]

theorem foldState_measurable (c0 : Precisions A) (m0 : Means A) (n : ℕ) :
    Measurable (foldState c0 m0 n) := by
  induction n with
  | zero => exact measurable_const
  | succ n ih =>
    have hh : Measurable (fun h : HistoryPolicy.History A (n + 1) => Fin.init h) := by
      unfold Fin.init
      fun_prop
    have hp := ih.comp hh
    have hz : Measurable (fun h : HistoryPolicy.History A (n + 1) => h (Fin.last n)) :=
      measurable_pi_apply _
    exact updateState_joint_measurable.comp (hz.fst.prodMk (hp.prodMk hz.snd.fst))

/-- A real-precision posterior-state policy is a concrete Borel history
policy. The compiler sees no latent means, future seeds, or unobserved rewards.
The argument N is the fixed total number of draws after the starting state. -/
def compile (π : RealPrecisionPolicy.Policy A) (N : ℕ)
    (c0 : Precisions A) (m0 : Means A) : HistoryPolicy.Policy A where
  draw := fun n h u =>
    let p := foldState c0 m0 n h
    π.draw (N - (n + 1)) p.1 p.2 u
  recommend := fun n h u =>
    let p := foldState c0 m0 n h
    π.recommend p.1 p.2 u
  draw_measurable := by
    intro n
    have hp := (foldState_measurable c0 m0 n).comp
      (measurable_fst : Measurable (fun p : HistoryPolicy.History A n × ℝ => p.1))
    exact (π.draw_measurable (N - (n + 1))).comp (hp.prodMk measurable_snd)
  recommend_measurable := by
    intro n
    have hp := (foldState_measurable c0 m0 n).comp
      (measurable_fst : Measurable (fun p : HistoryPolicy.History A n × ℝ => p.1))
    exact π.recommend_measurable.comp (hp.prodMk measurable_snd)

/-- Exact equality of compiled raw strategies at every reachable stage,
including every selected reward, posterior update, and fresh randomizer. -/
theorem toRaw_compile (π : RealPrecisionPolicy.Policy A) (N n H : ℕ)
    (c0 : Precisions A) (m0 : Means A) (h : HistoryPolicy.History A n)
    (u : Seeds H) (horizon : n + H = N) :
    HistoryPolicy.toRaw (compile π N c0 m0) n H h u =
      RealPrecisionPolicy.rawTree π H
        (foldState c0 m0 n h).1 (foldState c0 m0 n h).2 u := by
  induction H generalizing n with
  | zero => rfl
  | succ H ih =>
    have hindex : N - (n + 1) = H := by omega
    let p := foldState c0 m0 n h
    let a := π.draw H p.1 p.2 (u ⟨0, by omega⟩)
    change Strategy.sample (π.draw (N - (n + 1)) p.1 p.2 (u ⟨0, by omega⟩))
      (fun y => HistoryPolicy.toRaw (compile π N c0 m0) (n + 1) H
        (HistoryPolicy.append h
          (π.draw (N - (n + 1)) p.1 p.2 (u ⟨0, by omega⟩))
          y (u ⟨0, by omega⟩)) (seedTail u)) =
      Strategy.sample a (fun y => RealPrecisionPolicy.rawTree π H (advance p.1 a)
        (shift p.2 a ((y - p.2 a) / ((p.1 a : ℝ) + 1))) (seedTail u))
    simp only [hindex]
    congr 1
    funext y
    rw [ih (n + 1) _ (seedTail u) (by omega), foldState_append]
    rfl

def emptyHistory : HistoryPolicy.History A 0 := Fin.elim0

/-- Exact identity of actual posterior risks, obtained by compilation into
the same latent-Gaussian and raw-observation execution. -/
theorem actualRisk_compile (π : RealPrecisionPolicy.Policy A) (H : ℕ)
    (c : Precisions A) (m : Means A) :
    HistoryPolicy.actualRisk (compile π H c m) 0 H emptyHistory c m =
      RealPrecisionPolicy.actualRisk π H c m := by
  simp only [HistoryPolicy.actualRisk, RealPrecisionPolicy.actualRisk, rawRisk,
    ← HistoryPolicy.toRaw_run_eq]
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro u
  have he := toRaw_compile π H 0 H c m emptyHistory u (by omega)
  simpa only [foldState] using congrArg (fun s : Strategy (A := A) H =>
    ∫ w, maxAll w.1 - w.1 (run s (observations w)) ∂worldLaw H c m) he

end
end Junpei.GeneralOptimality
