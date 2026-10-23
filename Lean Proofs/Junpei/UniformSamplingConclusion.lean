import Junpei.UniformSamplingRate

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

open Junpei.FiniteGaussian Junpei.FiniteModelExclusion Junpei.ProseCorollaries Junpei.TailRatio

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

theorem minimum_gap_squared_identity (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}] (K : ℝ) (hK : 0 < K)
    (hbest : ∀ a, a ≠ best → μ a < μ best) :
    -maxAll (fun a : {a : A // a ≠ best} => -((μ best - μ a)^2 / (4*K))) =
      (minimumGap μ best)^2 / (4*K) := by
  obtain ⟨a0, ha0⟩ := maxAll_attained (fun a : {a : A // a ≠ best} => μ a)
  have haGap : μ best - μ a0 = minimumGap μ best := by
    unfold minimumGap otherMaximum
    rw [← ha0]
  have hm : maxAll (fun a : {a : A // a ≠ best} => -((μ best - μ a)^2 / (4*K))) =
      -((minimumGap μ best)^2 / (4*K)) := by
    apply le_antisymm
    · apply maxAll_le
      intro a
      apply neg_le_neg
      apply div_le_div_of_nonneg_right _ (by positivity)
      have hg := (gap_bounds μ best a a.2).1
      exact (sq_le_sq₀ (minimumGap_pos μ best hbest).le
        (sub_pos.mpr (hbest a a.2)).le).mpr hg
    · simpa only [haGap] using coordinate_le_maxAll
        (fun a : {a : A // a ≠ best} => -((μ best - μ a)^2 / (4*K))) a0
  rw [hm, neg_neg]

/-- Exact frequentist error exponent for any deterministic allocation with
asymptotically equal sampling proportions. The probability is derived from
the actual independent Gaussian observations and empirical-max selector. -/
theorem equal_allocation_error_limit (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best)
    (n : ℕ → A → ℕ) (K : ℝ) (hK : 0 < K)
    (hn : ∀ a, Tendsto (fun T : ℕ => (n T a : ℝ) / (T : ℝ)) atTop (𝓝 (1/K)))
    (hproper : ∀ᶠ T in atTop, (∀ a, 0 < n T a) ∧ (∀ a, n T a ≤ T)) :
    Tendsto (fun T => -Real.log (error T μ best (n T)) / (T : ℝ)) atTop
      (𝓝 ((minimumGap μ best)^2 / (4*K))) := by
  let p : ℕ → {a : A // a ≠ best} → ℝ := fun T a => normalTail (threshold μ (n T) best a)
  have h := finite_union_rate_limit p (fun T => error T μ best (n T))
    (fun a => (μ best - μ a)^2 / (4*K))
    (Filter.Eventually.of_forall (fun T a => normalTail_pos _))
    (by
      filter_upwards [hproper] with T hT
      exact (error_tail_sandwich T μ (n T) best hT.1 hT.2).1)
    (by
      filter_upwards [hproper] with T hT
      exact (error_tail_sandwich T μ (n T) best hT.1 hT.2).2)
    (fun a => threshold_rate_limit μ n K hK hn best a (sub_pos.mpr (hbest a a.2)))
  rwa [minimum_gap_squared_identity μ best K hK hbest] at h

theorem equal_allocation_error_eventually_pos (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}] (n : ℕ → A → ℕ)
    (hproper : ∀ᶠ T in atTop, (∀ a, 0 < n T a) ∧ (∀ a, n T a ≤ T)) :
    ∀ᶠ T in atTop, 0 < error T μ best (n T) := by
  obtain ⟨a⟩ := (inferInstance : Nonempty {a : A // a ≠ best})
  filter_upwards [hproper] with T hT
  exact (normalTail_pos _).trans_le
    ((error_tail_sandwich T μ (n T) best hT.1 hT.2).1 a)

theorem regret_error_bounds (T : ℕ) (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) (n : A → ℕ) :
    minimumGap μ best * error T μ best n ≤ regret T μ best n ∧
      regret T μ best n ≤ maximumGap μ best * error T μ best n :=
  error_regret_sandwich (law T) μ best hbest (recommendation μ n)
    (recommendation_measurable T μ n)

/-- The exact same exponent holds for actual frequentist simple regret. -/
theorem equal_allocation_regret_limit (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best)
    (n : ℕ → A → ℕ) (K : ℝ) (hK : 0 < K)
    (hn : ∀ a, Tendsto (fun T : ℕ => (n T a : ℝ) / (T : ℝ)) atTop (𝓝 (1/K)))
    (hproper : ∀ᶠ T in atTop, (∀ a, 0 < n T a) ∧ (∀ a, n T a ≤ T)) :
    Tendsto (fun T => -Real.log (regret T μ best (n T)) / (T : ℝ)) atTop
      (𝓝 ((minimumGap μ best)^2 / (4*K))) := by
  have he := equal_allocation_error_limit μ best hbest n K hK hn hproper
  have hl := he.sub (tendsto_const_div_atTop_nhds_zero_nat (Real.log (maximumGap μ best)))
  have hu := he.sub (tendsto_const_div_atTop_nhds_zero_nat (Real.log (minimumGap μ best)))
  have hamin := minimumGap_pos μ best hbest
  have hamax := maximumGap_pos μ best hbest
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le'
    (by simpa only [sub_zero] using hl) (by simpa only [sub_zero] using hu)
  · filter_upwards [equal_allocation_error_eventually_pos μ best n hproper,
      eventually_ge_atTop (1 : ℕ)] with T hp hT
    have hb := regret_error_bounds T μ best hbest (n T)
    have hrp := (mul_pos hamin hp).trans_le hb.1
    have hh := Real.log_le_log hrp hb.2
    rw [Real.log_mul hamax.ne' hp.ne'] at hh
    have hd := div_le_div_of_nonneg_right (neg_le_neg hh)
      (by positivity : 0 ≤ (T : ℝ))
    simpa only [neg_add, add_div, sub_eq_add_neg, neg_div, add_comm] using hd
  · filter_upwards [equal_allocation_error_eventually_pos μ best n hproper,
      eventually_ge_atTop (1 : ℕ)] with T hp hT
    have hb := regret_error_bounds T μ best hbest (n T)
    have hh := Real.log_le_log (mul_pos hamin hp) hb.1
    rw [Real.log_mul hamin.ne' hp.ne'] at hh
    have hd := div_le_div_of_nonneg_right (neg_le_neg hh)
      (by positivity : 0 ≤ (T : ℝ))
    simpa only [neg_add, add_div, sub_eq_add_neg, neg_div, add_comm] using hd

theorem equal_allocation_extended_rate_limit (μ : Means A) (best : A)
    [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best)
    (n : ℕ → A → ℕ) (K : ℝ) (hK : 0 < K)
    (hn : ∀ a, Tendsto (fun T : ℕ => (n T a : ℝ) / (T : ℝ)) atTop (𝓝 (1/K)))
    (hproper : ∀ᶠ T in atTop, (∀ a, 0 < n T a) ∧ (∀ a, n T a ≤ T)) :
    Tendsto (Junpei.RegretExponent.rate (fun T => regret T μ best (n T))) atTop
      (𝓝 (((minimumGap μ best)^2 / (4*K) : ℝ) : EReal)) := by
  have h := EReal.tendsto_coe.mpr (equal_allocation_regret_limit μ best hbest n K hK hn hproper)
  apply h.congr'
  filter_upwards [equal_allocation_error_eventually_pos μ best n hproper] with T hp
  have hrp := (mul_pos (minimumGap_pos μ best hbest) hp).trans_le
    (regret_error_bounds T μ best hbest (n T)).1
  simp only [Junpei.RegretExponent.rate, if_neg hrp.ne']

end
end Junpei.UniformSampling
