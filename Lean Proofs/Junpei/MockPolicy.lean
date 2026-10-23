import Junpei.MockSample
import Junpei.EmbeddingRaw

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockPolicy

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.FiniteModelExclusion
open Junpei.FiniteModelMeasurable
open Junpei.EmbeddingRaw

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

def retainedBest (m : Means A) : A :=
  ((firstMax (A := {a : A // a ≠ i}) (fun a => m a) : {a : A // a ≠ i}) : A)

theorem retainedBest_excludes (m : Means A) : retainedBest i m ≠ i :=
  (firstMax (A := {a : A // a ≠ i}) (fun a => m a)).property

theorem retainedBest_attains (m : Means A) :
    m (retainedBest i m) = otherMaximum i m :=
  firstMax_attains (A := {a : A // a ≠ i}) (fun a => m a)

theorem retainedBest_measurable : Measurable (retainedBest i) := by
  exact measurable_subtype_coe.comp (firstMax_measurable.comp
    (measurable_pi_lambda _ (fun a => measurable_pi_apply (a : A))))

def terminalChoice (m : Means A) (a : A) : A :=
  if a = i then retainedBest i m else a

theorem terminalChoice_excludes (m : Means A) (a : A) : terminalChoice i m a ≠ i := by
  unfold terminalChoice
  split_ifs with h
  · exact retainedBest_excludes i m
  · exact h

theorem terminalChoice_preserves (m : Means A) (a : A) (ha : a ≠ i) :
    terminalChoice i m a = a := by simp [terminalChoice, ha]

theorem terminalChoice_measurable :
    Measurable (fun p : Means A × A => terminalChoice i p.1 p.2) :=
  ((retainedBest_measurable i).comp measurable_fst).ite
    (measurable_snd (measurableSet_singleton i)) measurable_snd

theorem terminalChoice_posterior_best (m : Means A) (a : A) (ha : m a = maxAll m) :
    m (terminalChoice i m a) = otherMaximum i m := by
  unfold terminalChoice
  split_ifs with h
  · exact retainedBest_attains i m
  · apply le_antisymm
    · exact coordinate_le_maxAll (fun b : {a : A // a ≠ i} => m b) ⟨a, h⟩
    · apply maxAll_le
      intro b
      rw [ha]
      exact coordinate_le_maxAll m b

def terminalModified : {H : ℕ} → Strategy (A := A) H → Precisions A → Means A →
    Strategy (A := A) H
  | 0, .stop a, _, m => .stop (terminalChoice i m a)
  | _ + 1, .sample a next, c, m =>
      .sample a (fun y =>
        let z := (y - m a) / ((c a : ℝ) + 1)
        terminalModified (next y) (advance c a) (shift m a z))

/-- Replay the internally requested sequence using the same actual arm requests.
If that arm is i, ignore the real reward and supply the next auxiliary reward.
Every other real reward is passed unchanged to the Bayes continuation. -/
def replay : {H : ℕ} → Strategy (A := A) H → Precisions A → Means A →
    (Fin H → ℝ) → Strategy (A := A) H
  | 0, .stop a, _, m, _ => .stop (terminalChoice i m a)
  | H + 1, .sample a next, c, m, imaginary =>
      if a = i then
        let y := imaginary ⟨0, Nat.zero_lt_succ H⟩
        let z := (y - m a) / ((c a : ℝ) + 1)
        .sample a (fun _ => replay (next y) (advance c a) (shift m a z)
          (fun t => imaginary t.succ))
      else
        .sample a (fun y =>
          let z := (y - m a) / ((c a : ℝ) + 1)
          replay (next y) (advance c a) (shift m a z) (fun t => imaginary t.succ))

def substitute {H : ℕ} (real : RawStream A H) (imaginary : Fin H → ℝ) : RawStream A H :=
  fun a t => if a = i then imaginary t else real a t

theorem replay_discards_real {H : ℕ} (next : ℝ → Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (imaginary : Fin (H + 1) → ℝ) :
    replay i (.sample i next) c m imaginary =
      .sample i (fun _ => replay i (next (imaginary ⟨0, Nat.zero_lt_succ H⟩))
        (advance c i) (shift m i ((imaginary ⟨0, Nat.zero_lt_succ H⟩ - m i) /
          ((c i : ℝ) + 1))) (fun t => imaginary t.succ)) := by
  simp [replay]

theorem replay_supplies_real {H : ℕ} (a : A) (ha : a ≠ i)
    (next : ℝ → Strategy (A := A) H) (c : Precisions A) (m : Means A)
    (imaginary : Fin (H + 1) → ℝ) :
    replay i (.sample a next) c m imaginary =
      .sample a (fun y => replay i (next y) (advance c a)
        (shift m a ((y - m a) / ((c a : ℝ) + 1))) (fun t => imaginary t.succ)) := by
  simp [replay, ha]

theorem replay_run_eq {H : ℕ} (s : Strategy (A := A) H) (c : Precisions A) (m : Means A)
    (imaginary : Fin H → ℝ) (real : RawStream A H) :
    run (replay i s c m imaginary) real =
      run (terminalModified i s c m) (substitute i real imaginary) := by
  induction H generalizing c m with
  | zero => cases s; rfl
  | succ H ih =>
    cases s with
    | sample a next =>
      by_cases ha : a = i
      · simp only [replay, run, terminalModified, substitute, ha, if_true]
        exact ih (next _) _ _ _ _
      · simp only [replay, run, terminalModified, substitute, if_neg ha]
        exact ih (next _) _ _ _ _

theorem terminalModified_excludes {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (real : RawStream A H) :
    run (terminalModified i s c m) real ≠ i := by
  induction H generalizing c m with
  | zero => cases s with | stop a => exact terminalChoice_excludes i m a
  | succ H ih => cases s with | sample a next => exact ih _ _ _ _

theorem terminalModified_preserves {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (real : RawStream A H) (hne : run s real ≠ i) :
    run (terminalModified i s c m) real = run s real := by
  induction H generalizing c m with
  | zero => cases s with | stop a => exact terminalChoice_preserves i m a hne
  | succ H ih => cases s with | sample a next => exact ih _ _ _ _ hne

theorem terminalModified_regret_bound {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (real : RawStream A H) (u : Means A) :
    (maxAll u - u (run (terminalModified i s c m) real)) - (maxAll u - u (run s real)) ≤
      max (u i - otherMinimum i u) 0 := by
  by_cases h : run s real = i
  · rw [h]
    have hmin := Junpei.MockSample.otherMinimum_le i u _
      (terminalModified_excludes i s c m real)
    have hp := le_max_left (u i - otherMinimum i u) (0 : ℝ)
    linarith
  · rw [terminalModified_preserves i s c m real h]
    simp only [sub_self]
    exact le_max_right _ _

theorem replay_excludes {H : ℕ} (s : Strategy (A := A) H) (c : Precisions A)
    (m : Means A) (imaginary : Fin H → ℝ) (real : RawStream A H) :
    run (replay i s c m imaginary) real ≠ i := by
  rw [replay_run_eq]
  exact terminalModified_excludes i s c m _

theorem replay_budget {H : ℕ} (s : Strategy (A := A) H) (c : Precisions A)
    (m : Means A) (imaginary : Fin H → ℝ) (real : RawStream A H) :
    (trace (replay i s c m imaginary) real).length = H := trace_budget _ _

/-- The first auxiliary observation updates the internal Gaussian posterior
but consumes no actual sampling opportunity. -/
def continuation (H : ℕ) (c : Precisions A) (m : Means A) (y : ℝ) : Strategy (A := A) H :=
  let z := (y - m i) / ((c i : ℝ) + 1)
  rawOfInnovation (optimalFirst H (advance c i) (shift m i z))
    (advance c i) (shift m i z)

def forcedPolicy (H : ℕ) (c : Precisions A) (m : Means A) : Strategy (A := A) (H + 1) :=
  .sample i (continuation i H c m)

def mockPolicy (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) : Strategy (A := A) H :=
  let y := auxiliary ⟨0, Nat.zero_lt_succ H⟩
  let z := (y - m i) / ((c i : ℝ) + 1)
  replay i (continuation i H c m y) (advance c i) (shift m i z)
    (fun t => auxiliary t.succ)

/-- This is the paper's T-t actual budget, despite the initial mock observation.
The theorem also returns a complete future arm sequence with exactly that length. -/
theorem mockPolicy_budget (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A H) :
    (trace (mockPolicy i H c m auxiliary) real).length = H := trace_budget _ _

theorem mockPolicy_recommends_retained (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A H) :
    run (mockPolicy i H c m auxiliary) real ≠ i := replay_excludes i _ _ _ _ _

/-- The complete mock execution agrees pathwise with the real forced-i
Bayes continuation, supplied the substituted array and terminal restriction. -/
theorem mockPolicy_run_eq_full (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A (H + 1)) :
    run (mockPolicy i H c m auxiliary) (tail real) =
      run (terminalModified i (forcedPolicy i H c m) c m) (substitute i real auxiliary) := by
  simp only [mockPolicy, forcedPolicy, terminalModified, run, substitute, if_true]
  exact replay_run_eq i _ _ _ _ _

end Removed
end
end Junpei.MockPolicy
