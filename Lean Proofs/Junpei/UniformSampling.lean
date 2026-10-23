import Junpei.UniformSamplingRounding

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

open Junpei.FiniteGaussian Junpei.ProseCorollaries

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

omit [DecidableEq A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem allocation_eventually_proper :
    ∀ᶠ T : ℕ in atTop,
      (∀ a : A, 0 < allocation T a) ∧ (∀ a : A, allocation T a ≤ T) := by
  filter_upwards [eventually_ge_atTop (Fintype.card A)] with T hT
  exact ⟨allocation_positive T hT, allocation_le_budget T⟩

/-- The manuscript's uniform-sampling frequentist exponent, with the
allocation rounding made explicit and the actual empirical recommendation.
This is a full limit, which in particular certifies the paper's liminf. -/
theorem uniform_frequentist_rate_limit (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) :
    Tendsto (Junpei.RegretExponent.rate (fun T => regret T μ best (allocation T))) atTop
      (𝓝 (((minimumGap μ best)^2 / (4 * (Fintype.card A : ℝ)) : ℝ) : EReal)) := by
  exact equal_allocation_extended_rate_limit μ best hbest allocation (Fintype.card A)
    (by exact_mod_cast (Fintype.card_pos : 0 < Fintype.card A))
    allocation_proportion allocation_eventually_proper

theorem uniform_frequentist_rate (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) :
    liminf (Junpei.RegretExponent.rate (fun T => regret T μ best (allocation T))) atTop =
      (((minimumGap μ best)^2 / (4 * (Fintype.card A : ℝ)) : ℝ) : EReal) :=
  (uniform_frequentist_rate_limit μ best hbest).liminf_eq

theorem uniform_error_log_limit (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) :
    Tendsto (fun T => -Real.log (error T μ best (allocation T)) / (T : ℝ)) atTop
      (𝓝 ((minimumGap μ best)^2 / (4 * (Fintype.card A : ℝ)))) := by
  exact equal_allocation_error_limit μ best hbest allocation (Fintype.card A)
    (by exact_mod_cast (Fintype.card_pos : 0 < Fintype.card A))
    allocation_proportion allocation_eventually_proper

end
end Junpei.UniformSampling
