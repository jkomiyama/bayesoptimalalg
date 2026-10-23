import Junpei.UniformSampling

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

open Junpei.FiniteGaussian Junpei.FiniteModelMeasurable

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

/-- The actual reward array. The centered array is a probability-space
representation, while an implemented algorithm sees these shifted rewards. -/
def observedRewards {T : ℕ} (μ : Means A) (x : Arrays A T) : Arrays A T :=
  fun a j => μ a + x a j

def observedEmpirical {T : ℕ} (n : A → ℕ) (r : Arrays A T) : Means A :=
  fun a => Draft0.BadEvent.sampleMean (n a) (r a)

/-- This decision rule has no true-mean argument: it takes averages of the
observed rewards and returns the fixed Borel empirical maximizer. -/
def observedRecommendation {T : ℕ} (n : A → ℕ) (r : Arrays A T) : A :=
  firstMax (observedEmpirical n r)

theorem observedRewards_measurable (T : ℕ) (μ : Means A) :
    Measurable (observedRewards (T := T) μ) := by
  unfold observedRewards
  fun_prop

theorem observedEmpirical_measurable (T : ℕ) (n : A → ℕ) :
    Measurable (observedEmpirical (T := T) n) := by
  apply measurable_pi_lambda
  intro a
  exact (Draft0.FrequentistModel.sampleMean_continuous T (n a)).measurable.comp
    (measurable_pi_apply a)

theorem observedRecommendation_measurable (T : ℕ) (n : A → ℕ) :
    Measurable (observedRecommendation (T := T) n) :=
  firstMax_measurable.comp (observedEmpirical_measurable T n)

theorem meanCoefficient_sum (T n : ℕ) (hn : 0 < n) (hnT : n ≤ T) :
    (∑ j : Fin T, Draft0.BadEvent.meanCoefficient n j) = 1 := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt hn)
  unfold Draft0.BadEvent.meanCoefficient
  rw [Draft0.BadEvent.fin_sum_if_lt n T hnT]
  simp only [mul_zero, add_zero, mul_inv_cancel₀ hnR]

/-- The empirical statistic used in the exponent proof is literally the
average of the observed rewards, whenever the allocated count is positive. -/
theorem observedEmpirical_eq_empirical (T : ℕ) (μ : Means A) (n : A → ℕ)
    (hn : ∀ a, 0 < n a) (hnT : ∀ a, n a ≤ T) (x : Arrays A T) :
    observedEmpirical n (observedRewards μ x) = empirical μ n x := by
  funext a
  unfold observedEmpirical observedRewards empirical Draft0.BadEvent.sampleMean
    Draft0.BadEvent.weightedSum
  simp only [mul_add, Finset.sum_add_distrib, ← Finset.sum_mul,
    meanCoefficient_sum T (n a) (hn a) (hnT a), one_mul]

/-- Pointwise equality connects the parameter-independent observed-data
algorithm to the actual Gaussian-world recommendation used by every limit. -/
theorem observedRecommendation_eq (T : ℕ) (μ : Means A) (n : A → ℕ)
    (hn : ∀ a, 0 < n a) (hnT : ∀ a, n a ≤ T) (x : Arrays A T) :
    observedRecommendation n (observedRewards μ x) = recommendation μ n x := by
  unfold observedRecommendation recommendation
  rw [observedEmpirical_eq_empirical T μ n hn hnT x]

def observedLaw (T : ℕ) (μ : Means A) : Measure (Arrays A T) :=
  (law T).map (observedRewards μ)

instance observedLaw_probability (T : ℕ) (μ : Means A) :
    IsProbabilityMeasure (observedLaw T μ) :=
  MeasureTheory.Measure.isProbabilityMeasure_map (observedRewards_measurable T μ).aemeasurable

def observedRegret (T : ℕ) (μ : Means A) (best : A) (n : A → ℕ) : ℝ :=
  ∫ r, μ best - μ (observedRecommendation n r) ∂observedLaw T μ

/-- Exact equality of actual observed-data risk and the centered-array
representation; no distribution or model-equivalence hypothesis is assumed. -/
theorem observedRegret_eq (T : ℕ) (μ : Means A) (best : A) (n : A → ℕ)
    (hn : ∀ a, 0 < n a) (hnT : ∀ a, n a ≤ T) :
    observedRegret T μ best n = regret T μ best n := by
  have hm : Measurable (fun r : Arrays A T => μ best - μ (observedRecommendation n r)) :=
    measurable_const.sub ((measurable_of_countable μ).comp
      (observedRecommendation_measurable T n))
  unfold observedRegret observedLaw
  rw [integral_map (observedRewards_measurable T μ).aemeasurable hm.aestronglyMeasurable]
  simp only [observedRecommendation_eq T μ n hn hnT, regret]

/-- The uniform rate statement for a rule whose computational inputs are
only raw observed rewards and the deterministic rounded allocation. -/
theorem observed_uniform_frequentist_rate_limit (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) :
    Tendsto (Junpei.RegretExponent.rate
      (fun T => observedRegret T μ best (allocation T))) atTop
      (𝓝 (((Junpei.ProseCorollaries.minimumGap μ best)^2 /
        (4 * (Fintype.card A : ℝ)) : ℝ) : EReal)) := by
  apply (uniform_frequentist_rate_limit μ best hbest).congr'
  filter_upwards [allocation_eventually_proper (A := A)] with T hT
  simp only [Junpei.RegretExponent.rate, observedRegret_eq T μ best (allocation T) hT.1 hT.2]

theorem observed_uniform_frequentist_rate (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) :
    liminf (Junpei.RegretExponent.rate
      (fun T => observedRegret T μ best (allocation T))) atTop =
      (((Junpei.ProseCorollaries.minimumGap μ best)^2 /
        (4 * (Fintype.card A : ℝ)) : ℝ) : EReal) :=
  (observed_uniform_frequentist_rate_limit μ best hbest).liminf_eq

end
end Junpei.UniformSampling
