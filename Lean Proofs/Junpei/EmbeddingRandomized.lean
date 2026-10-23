import Junpei.EmbeddingExecution

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.EmbeddingRandomized

open FiniteGaussian FinitePosterior EmbeddingRaw EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

theorem rawPayoff_measurable {H : ℕ} (s : Strategy (A := A) H) (hs : Measurable (run s)) :
    Measurable (fun w : Means A × RawStream A H => maxAll w.1 - w.1 (run s (observations w))) := by
  have hcoord : Measurable (fun p : (Means A × RawStream A H) × A => p.1.1 p.2) :=
    measurable_from_prod_countable_left (fun a => by
      change Measurable (fun w : Means A × RawStream A H => w.1 a)
      fun_prop)
  exact ((maxAll_lipschitz (A := A)).continuous.measurable.comp measurable_fst).sub
    (hcoord.comp (measurable_id.prodMk (hs.comp (observations_measurable H))))

theorem regret_norm_le_sum (u : Means A) (a : A) :
    ‖maxAll u - u a‖ ≤ 2 * ∑ b, |u b| := by
  obtain ⟨b, hb⟩ := maxAll_attained u
  have h1 := Finset.single_le_sum (fun b _ => abs_nonneg (u b)) (Finset.mem_univ b)
  have h2 := Finset.single_le_sum (fun b _ => abs_nonneg (u b)) (Finset.mem_univ a)
  rw [Real.norm_eq_abs, ← hb]
  have h := abs_add_le (u b) (-u a)
  rw [abs_neg, ← sub_eq_add_neg] at h
  linarith

variable {S : Type*} [MeasurableSpace S]

def randomizedPayoff {H : ℕ} (s : S → Strategy (A := A) H)
    (p : S × (Means A × RawStream A H)) : ℝ :=
  maxAll p.2.1 - p.2.1 (run (s p.1) (observations p.2))

theorem randomizedPayoff_measurable {H : ℕ} (s : S → Strategy (A := A) H)
    (hs : Measurable (fun p : S × RawStream A H => run (s p.1) p.2)) :
    Measurable (randomizedPayoff s) := by
  have ht : Measurable (fun p : S × (Means A × RawStream A H) =>
      (p.1, observations p.2)) := measurable_fst.prodMk ((observations_measurable H).comp measurable_snd)
  have hcoord : Measurable (fun p : (S × (Means A × RawStream A H)) × A => p.1.2.1 p.2) :=
    measurable_from_prod_countable_left (fun a => by
      change Measurable (fun p : S × (Means A × RawStream A H) => p.2.1 a)
      fun_prop)
  exact ((maxAll_lipschitz (A := A)).continuous.measurable.comp
    (measurable_fst.comp measurable_snd)).sub
      (hcoord.comp (measurable_id.prodMk (hs.comp ht)))

theorem randomizedPayoff_integrable {H : ℕ} (seedLaw : Measure S) [IsProbabilityMeasure seedLaw]
    (s : S → Strategy (A := A) H) (c : Precisions A) (m : Means A)
    (hs : Measurable (fun p : S × RawStream A H => run (s p.1) p.2)) :
    Integrable (randomizedPayoff s) (seedLaw.prod (worldLaw H c m)) := by
  have hi : Integrable (fun p : S × (Means A × RawStream A H) => 2 * ∑ a, |p.2.1 a|)
      (seedLaw.prod (worldLaw H c m)) := by
    apply Integrable.const_mul
    apply integrable_finset_sum
    intro a _
    simpa only [Real.norm_eq_abs] using
      (((posterior_coordinate_integrable c m a).norm.comp_fst (arrayLaw H)).comp_snd seedLaw)
  apply hi.mono' (randomizedPayoff_measurable s hs).aestronglyMeasurable
  exact Filter.Eventually.of_forall (fun p => regret_norm_le_sum p.2.1 _)

theorem randomized_rawRisk_integrable {H : ℕ} (seedLaw : Measure S) [IsProbabilityMeasure seedLaw]
    (s : S → Strategy (A := A) H) (c : Precisions A) (m : Means A)
    (hs : Measurable (fun p : S × RawStream A H => run (s p.1) p.2)) :
    Integrable (fun r => rawRisk (s r) c m) seedLaw :=
  (randomizedPayoff_integrable seedLaw s c m hs).integral_prod_left

/-- Independent external randomization, including an entire imaginary
posterior-predictive stream, is a feasible raw-history strategy and cannot
beat the deterministic Gaussian Bellman optimum. -/
theorem randomized_rawRisk_ge_loss {H : ℕ} (seedLaw : Measure S) [IsProbabilityMeasure seedLaw]
    (s : S → Strategy (A := A) H) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) (hregular : ∀ r, Regular (s r))
    (hs : Measurable (fun p : S × RawStream A H => run (s p.1) p.2)) :
    loss H c m ≤ ∫ r, rawRisk (s r) c m ∂seedLaw := by
  have h := integral_mono (integrable_const (loss H c m))
    (randomized_rawRisk_integrable seedLaw s c m hs)
    (fun r => rawRisk_ge_loss (s r) c m hc (hregular r))
  simpa using h

theorem randomized_rawRisk_ge_actionLoss {H : ℕ} (seedLaw : Measure S)
    [IsProbabilityMeasure seedLaw] (a : A) (next : S → ℝ → Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hc : ∀ b, 0 < c b)
    (hregular : ∀ r, Regular (.sample a (next r)))
    (hs : Measurable (fun p : S × RawStream A (H + 1) => run (.sample a (next p.1)) p.2)) :
    actionLoss H c m a ≤ ∫ r, rawRisk (.sample a (next r)) c m ∂seedLaw := by
  have h := integral_mono (integrable_const (actionLoss H c m a))
    (randomized_rawRisk_integrable seedLaw (fun r => .sample a (next r)) c m hs)
    (fun r => rawRisk_ge_actionLoss a (next r) c m hc (hregular r))
  simpa using h

end
end Junpei.EmbeddingRandomized
