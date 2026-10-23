import Junpei.EmbeddingCoupling
import Junpei.EmbeddingMeasurable

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.PolicyEmbedding

open FiniteGaussian FinitePosterior FiniteModelExclusion FiniteModelMeasurable
open EmbeddingDirect EmbeddingRaw EmbeddingExecution EmbeddingCoupling EmbeddingMeasurable

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]
variable (i : A) [Nonempty (Retained i)]

noncomputable section

theorem simulated_optimal_joint_measurable (dummy : Retained i) (H : ℕ)
    (c : Precisions A) (m : Means A) :
    Measurable (fun p : MockStream.Block H × RawStream (Retained i) H =>
      run (A := Retained i)
        (simulated i dummy (rawOfInnovation (optimalFirst H c m) c m) c m p.1) p.2) := by
  have ht : Measurable (fun p : MockStream.Block H × RawStream (Retained i) H =>
      ((m, restrictMeans i m), (copiedObservations i m p.1, p.2))) := by
    unfold copiedObservations
    fun_prop
  exact (result_measurable i dummy H c (restrictPrecisions i c)).comp ht

/-- Both literal policy embeddings of Lemma 13, for every finite arm set,
every removed arm, every finite horizon, and arbitrary positive real
Gaussian precisions. All measurability, independence, risk representation,
and budget properties of the constructed policies are proved in Lean. -/
theorem lemma13_loss_by_policy_embeddings (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) :
    |loss H c m - loss H (restrictPrecisions i c) (restrictMeans i m)| ≤ exclusion i c m := by
  obtain ⟨dummy⟩ := (inferInstance : Nonempty (Retained i))
  have hforward := lemma13_first_embedding i H c m hc
  have hreverse := lemma13_second_embedding i dummy
    (rawOfInnovation (optimalFirst H c m) c m) c m hc
    (regular_run_measurable _ (optimalFirst_raw_regular H c m))
    (fun b => simulated_optimal_regular i dummy H c m (restrictPrecisions i c)
      (restrictMeans i m) (copiedObservations i m b))
    (simulated_optimal_joint_measurable i dummy H c m)
  rw [optimalFirst_rawRisk H c m hc] at hreverse
  exact abs_le.mpr ⟨by linarith, hforward⟩

def forcedInnovation (H : ℕ) (c : Precisions A) (m : Means A) (a : A) : Strategy (A := A) (H + 1) :=
  .sample a (fun z => optimalFirst H (advance c a) (shift m a z))

def forcedRaw (H : ℕ) (c : Precisions A) (m : Means A) (a : A) : Strategy (A := A) (H + 1) :=
  rawOfInnovation (forcedInnovation H c m a) c m

theorem forcedRaw_regular (H : ℕ) (c : Precisions A) (m : Means A) (a : A) :
    Regular (forcedRaw H c m a) := by
  constructor
  · intro y
    exact optimalFirst_raw_regular H _ _
  · have ht : Measurable (fun p : ℝ × RawStream A H =>
        (shift m a ((p.1 - m a) / ((c a : ℝ) + 1)), p.2)) := by
      apply Measurable.prodMk
      · exact shift_param_measurable _ _ a measurable_const (by fun_prop)
      · exact measurable_snd
    exact (optimalFirst_raw_run_measurable H (advance c a)).comp ht

theorem forcedRaw_risk (H : ℕ) (c : Precisions A) (m : Means A) (a : A)
    (hc : ∀ b, 0 < c b) : rawRisk (forcedRaw H c m a) c m = actionLoss H c m a := by
  rw [forcedRaw, rawOfInnovation_risk _ _ _ hc (forcedRaw_regular H c m a)]
  change (∫ z, expectedRegret (optimalFirst H (advance c a) (shift m a z))
    (advance c a) (shift m a z) ∂gaussianReal 0 (innovation c a)) = actionLoss H c m a
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro z
  dsimp only
  rw [expectedRegret_eq _ _ _ (optimalFirst_admissible H _ _) (advance_positive c a hc),
    optimalFirst_utility]
  rfl

def forcedResult (dummy a : Retained i) (H : ℕ) (c : Precisions A)
    (d : Precisions (Retained i)) (p : Input i (H + 1)) : Retained i :=
  run (A := Retained i) (simulate i dummy (forcedRaw H c p.1.1 a)
    d p.1.2 p.2.1) p.2.2

theorem forcedResult_measurable (dummy a : Retained i) (H : ℕ) (c : Precisions A)
    (d : Precisions (Retained i)) : Measurable (forcedResult i dummy a H c d) := by
  have ht : Measurable (fun p : Input i (H + 1) =>
      ((shift p.1.1 (a : A)
          ((p.2.2 a ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 a) / ((c a : ℝ) + 1)),
        shift p.1.2 a
          ((p.2.2 a ⟨0, Nat.zero_lt_succ H⟩ - p.1.2 a) / ((d a : ℝ) + 1))),
        ((fun t : Fin H => p.2.1 t.succ), tail p.2.2))) := by
    unfold tail
    fun_prop
  have h := (result_measurable i dummy H (advance c a) (advance d a)).comp ht
  convert h using 1
  funext p
  simp only [forcedResult, forcedRaw, forcedInnovation, rawOfInnovation, simulate,
    a.property, dite_false, run, result, Function.comp_def]

theorem forced_simulated_regular (dummy a : Retained i) (H : ℕ) (c : Precisions A)
    (m : Means A) (d : Precisions (Retained i)) (n : Means (Retained i))
    (imaginary : Fin (H + 1) → ℝ) :
    Regular (simulate i dummy (forcedRaw H c m a) d n imaginary) := by
  simp only [forcedRaw, forcedInnovation, rawOfInnovation, simulate, a.property]
  constructor
  · intro y
    exact simulated_optimal_regular i dummy H _ _ _ _ _
  · have ht : Measurable (fun p : ℝ × RawStream (Retained i) H =>
        ((shift m (a : A) ((p.1 - m a) / ((c a : ℝ) + 1)),
          shift n a ((p.1 - n a) / ((d a : ℝ) + 1))),
          ((fun t : Fin H => imaginary t.succ), p.2))) := by
      fun_prop
    exact (result_measurable i dummy H (advance c a) (advance d a)).comp ht

theorem forced_simulated_joint_measurable (dummy a : Retained i) (H : ℕ)
    (c : Precisions A) (m : Means A) :
    Measurable (fun p : MockStream.Block (H + 1) × RawStream (Retained i) (H + 1) =>
      run (A := Retained i)
        (simulated i dummy (forcedRaw H c m a) c m p.1) p.2) := by
  have ht : Measurable (fun p : MockStream.Block (H + 1) × RawStream (Retained i) (H + 1) =>
      ((m, restrictMeans i m), (copiedObservations i m p.1, p.2))) := by
    unfold copiedObservations
    fun_prop
  exact (forcedResult_measurable i dummy a H c (restrictPrecisions i c)).comp ht

/-- Eq. (14) by the same two genuine policy embeddings, with the first
action constrained to either retained arm. -/
theorem lemma13_actionLoss_by_policy_embeddings (H : ℕ) (c : Precisions A) (m : Means A)
    (a : Retained i) (hc : ∀ b, 0 < c b) :
    |actionLoss H c m a - actionLoss H (restrictPrecisions i c) (restrictMeans i m) a| ≤
      exclusion i c m := by
  obtain ⟨dummy⟩ := (inferInstance : Nonempty (Retained i))
  have hforward := lemma13_first_embedding_forced i H c m a hc
  have hreg := forcedRaw_regular H c m a
  have hregular : ∀ b, Regular (simulated i dummy (forcedRaw H c m a) c m b) :=
    fun b => forced_simulated_regular i dummy a H c m (restrictPrecisions i c)
      (restrictMeans i m) (copiedObservations i m b)
  have hfull := regular_run_measurable _ hreg
  have hmeas := forced_simulated_joint_measurable i dummy a H c m
  have hreverse := lemma13_second_embedding_forced i dummy a
    (fun y => let z := (y - m a) / ((c a : ℝ) + 1)
      rawOfInnovation (optimalFirst H (advance c a) (shift m a z))
        (advance c a) (shift m a z)) c m hc hfull hregular hmeas
  change actionLoss H (restrictPrecisions i c) (restrictMeans i m) a ≤
    rawRisk (forcedRaw H c m a) c m + exclusion i c m at hreverse
  rw [forcedRaw_risk H c m a hc] at hreverse
  exact abs_le.mpr ⟨by linarith, hforward⟩

end
end Junpei.PolicyEmbedding
