import Junpei.MockPolicy
import Junpei.EmbeddingExecution

set_option linter.unusedSectionVars false
set_option maxHeartbeats 1000000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockMeasurable

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.FiniteModelExclusion
open Junpei.FiniteModelMeasurable Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open Junpei.MockPolicy

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

/-- The full raw Bayes continuation with its posterior-best retained
terminal recommendation is jointly measurable in its means and reward array. -/
theorem modifiedOptimal_run_measurable (H : ℕ) (c : Precisions A) :
    Measurable (fun p : Means A × RawStream A H =>
      run (terminalModified i (rawOfInnovation (optimalFirst H c p.1) c p.1) c p.1) p.2) := by
  induction H generalizing c with
  | zero =>
    exact (terminalChoice_measurable i).comp
      (measurable_fst.prodMk (firstMax_measurable.comp measurable_fst))
  | succ H ih =>
    let g : (Means A × RawStream A (H + 1)) × A → A := fun p =>
      run (terminalModified i (rawOfInnovation (optimalFirst H (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2 p.2 ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1)))) (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2 p.2 ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1)))) (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2 p.2 ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1)))) (tail p.1.2)
    have hg : Measurable g := by
      apply measurable_from_prod_countable_left
      intro a
      have ht : Measurable (fun p : Means A × RawStream A (H + 1) =>
          (shift p.1 a ((p.2 a ⟨0, Nat.zero_lt_succ H⟩ - p.1 a) / ((c a : ℝ) + 1)),
            tail p.2)) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro b
          simp only [shift]
          split_ifs <;> fun_prop
        · unfold tail; fun_prop
      exact (ih (advance c a)).comp ht
    exact hg.comp (measurable_id.prodMk ((chooseFirst_measurable H c).comp measurable_fst))

/-- Measurability remains true after real arm-i observations are ignored and
the externally generated pseudo-stream is supplied instead. -/
theorem replayOptimal_run_measurable (H : ℕ) (c : Precisions A) (imaginary : Fin H → ℝ) :
    Measurable (fun p : Means A × RawStream A H =>
      run (replay i (rawOfInnovation (optimalFirst H c p.1) c p.1) c p.1 imaginary) p.2) := by
  have ht : Measurable (fun p : Means A × RawStream A H =>
      (p.1, substitute i p.2 imaginary)) := by
    apply measurable_fst.prodMk
    apply measurable_pi_lambda
    intro a
    apply measurable_pi_lambda
    intro t
    simp only [substitute]
    split_ifs <;> fun_prop
  convert (modifiedOptimal_run_measurable i H c).comp ht using 1
  funext p
  exact replay_run_eq i _ _ _ _ _

/-- The mock policy is a regular measurable adaptive tree for each fixed
independent auxiliary stream; no real arm-i observation enters its continuation. -/
theorem replayOptimal_regular (H : ℕ) (c : Precisions A) (m : Means A)
    (imaginary : Fin H → ℝ) :
    Regular (replay i (rawOfInnovation (optimalFirst H c m) c m) c m imaginary) := by
  induction H generalizing c m with
  | zero => trivial
  | succ H ih =>
    let a := chooseFirst H c m
    by_cases ha : a = i
    · simp only [optimalFirst, rawOfInnovation, replay]
      rw [if_pos ha]
      change Regular (.sample a (fun _ =>
        replay i (rawOfInnovation (optimalFirst H (advance c a)
          (shift m a ((imaginary ⟨0, Nat.zero_lt_succ H⟩ - m a) / ((c a : ℝ) + 1))))
          (advance c a) (shift m a ((imaginary ⟨0, Nat.zero_lt_succ H⟩ - m a) /
            ((c a : ℝ) + 1)))) (advance c a)
          (shift m a ((imaginary ⟨0, Nat.zero_lt_succ H⟩ - m a) / ((c a : ℝ) + 1)))
          (fun t => imaginary t.succ)))
      constructor
      · intro _; exact ih _ _ _
      · exact (replayOptimal_run_measurable i H (advance c a) (fun t => imaginary t.succ)).comp
          (measurable_const.prodMk measurable_snd)
    · simp only [optimalFirst, rawOfInnovation, replay]
      rw [if_neg ha]
      change Regular (.sample a (fun y =>
        replay i (rawOfInnovation (optimalFirst H (advance c a)
          (shift m a ((y - m a) / ((c a : ℝ) + 1)))) (advance c a)
          (shift m a ((y - m a) / ((c a : ℝ) + 1)))) (advance c a)
          (shift m a ((y - m a) / ((c a : ℝ) + 1))) (fun t => imaginary t.succ)))
      constructor
      · intro _; exact ih _ _ _
      · have ht : Measurable (fun p : ℝ × RawStream A H =>
            (shift m a ((p.1 - m a) / ((c a : ℝ) + 1)), p.2)) := by
          apply Measurable.prodMk
          · apply measurable_pi_lambda
            intro b
            simp only [shift]
            split_ifs <;> fun_prop
          · exact measurable_snd
        exact (replayOptimal_run_measurable i H (advance c a)
          (fun t => imaginary t.succ)).comp ht

theorem mockPolicy_regular (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) : Regular (mockPolicy i H c m auxiliary) :=
  replayOptimal_regular i H _ _ _

theorem forcedPolicy_run_measurable (H : ℕ) (c : Precisions A) (m : Means A) :
    Measurable (run (forcedPolicy i H c m)) := by
  have ht : Measurable (fun p : RawStream A (H + 1) =>
      (shift m i ((p i ⟨0, Nat.zero_lt_succ H⟩ - m i) / ((c i : ℝ) + 1)), tail p)) := by
    apply Measurable.prodMk
    · apply measurable_pi_lambda
      intro b
      simp only [shift]
      split_ifs <;> fun_prop
    · unfold tail; fun_prop
  exact (optimalFirst_raw_run_measurable H (advance c i)).comp ht

theorem forcedPolicy_regular (H : ℕ) (c : Precisions A) (m : Means A) :
    Regular (forcedPolicy i H c m) := by
  constructor
  · intro y
    exact optimalFirst_raw_regular H _ _
  · have ht : Measurable (fun p : ℝ × RawStream A H =>
        (shift m i ((p.1 - m i) / ((c i : ℝ) + 1)), p.2)) := by
      apply Measurable.prodMk
      · apply measurable_pi_lambda
        intro b
        simp only [shift]
        split_ifs <;> fun_prop
      · exact measurable_snd
    exact (optimalFirst_raw_run_measurable H (advance c i)).comp ht

theorem forcedPolicy_modified_run_measurable (H : ℕ) (c : Precisions A) (m : Means A) :
    Measurable (run (terminalModified i (forcedPolicy i H c m) c m)) := by
  have ht : Measurable (fun p : RawStream A (H + 1) =>
      (shift m i ((p i ⟨0, Nat.zero_lt_succ H⟩ - m i) / ((c i : ℝ) + 1)), tail p)) := by
    apply Measurable.prodMk
    · apply measurable_pi_lambda
      intro b
      simp only [shift]
      split_ifs <;> fun_prop
    · unfold tail; fun_prop
  exact (modifiedOptimal_run_measurable i H (advance c i)).comp ht

def pad {H : ℕ} (real : RawStream A H) : RawStream A (H + 1) :=
  fun a => Fin.cases 0 (real a)

theorem tail_pad {H : ℕ} (real : RawStream A H) : tail (pad real) = real := rfl

/-- Joint Borel dependence on the entire independently generated auxiliary
stream and the actual adaptive observations. -/
theorem mockPolicy_run_measurable (H : ℕ) (c : Precisions A) (m : Means A) :
    Measurable (fun p : (Fin (H + 1) → ℝ) × RawStream A H =>
      run (mockPolicy i H c m p.1) p.2) := by
  have ht : Measurable (fun p : (Fin (H + 1) → ℝ) × RawStream A H =>
      substitute i (pad p.2) p.1) := by
    apply measurable_pi_lambda
    intro a
    apply measurable_pi_lambda
    intro t
    by_cases ha : a = i
    · simp only [substitute, ha, if_true]
      fun_prop
    · simp only [substitute, if_neg ha, pad]
      refine Fin.cases ?_ (fun r => ?_) t
      · exact measurable_const
      · simp only [Fin.cases_succ]
        fun_prop
  convert (forcedPolicy_modified_run_measurable i H c m).comp ht using 1
  funext p
  have h := mockPolicy_run_eq_full i H c m p.1 (pad p.2)
  simpa only [tail_pad] using h

def auxiliaryOfBlock (H : ℕ) (m : Means A) (b : MockStream.Block (H + 1)) : Fin (H + 1) → ℝ :=
  fun t => m i + b.1 + b.2 t

theorem auxiliaryOfBlock_measurable (H : ℕ) (m : Means A) :
    Measurable (auxiliaryOfBlock i H m) := by
  unfold auxiliaryOfBlock
  fun_prop

theorem mockPolicy_block_run_measurable (H : ℕ) (c : Precisions A) (m : Means A) :
    Measurable (fun p : MockStream.Block (H + 1) × RawStream A H =>
      run (mockPolicy i H c m (auxiliaryOfBlock i H m p.1)) p.2) := by
  have ht : Measurable (fun p : MockStream.Block (H + 1) × RawStream A H =>
      (auxiliaryOfBlock i H m p.1, p.2)) :=
    ((auxiliaryOfBlock_measurable i H m).comp measurable_fst).prodMk measurable_snd
  exact (mockPolicy_run_measurable i H c m).comp ht

end Removed
end
end Junpei.MockMeasurable
