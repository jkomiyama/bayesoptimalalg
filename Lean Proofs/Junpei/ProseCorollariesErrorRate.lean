import Junpei.FiniteModelExclusion
import Junpei.RegretExponent

set_option autoImplicit false
set_option warningAsError true

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.ProseCorollaries

open Junpei.FiniteGaussian Junpei.FiniteModelExclusion

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]
variable {Ω : Type*} [MeasurableSpace Ω]

noncomputable section

def minimumGap (μ : Means A) (best : A) [Nonempty {a : A // a ≠ best}] : ℝ :=
  μ best - otherMaximum best μ

def maximumGap (μ : Means A) (best : A) : ℝ := μ best + maxAll (-μ)

omit [Nonempty A] [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem minimumGap_pos (μ : Means A) (best : A) [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) : 0 < minimumGap μ best := by
  obtain ⟨a, ha⟩ := maxAll_attained (fun a : {a : A // a ≠ best} => μ a)
  unfold minimumGap otherMaximum
  rw [← ha]
  exact sub_pos.mpr (hbest a a.2)

omit [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem gap_bounds (μ : Means A) (best a : A) [Nonempty {a : A // a ≠ best}]
    (ha : a ≠ best) : minimumGap μ best ≤ μ best - μ a ∧
      μ best - μ a ≤ maximumGap μ best := by
  have hl := coordinate_le_maxAll (fun b : {b : A // b ≠ best} => μ b) ⟨a, ha⟩
  have hu := coordinate_le_maxAll (-μ) a
  constructor
  · change μ best - maxAll (fun b : {b : A // b ≠ best} => μ b) ≤ μ best - μ a
    linarith
  · change μ best - μ a ≤ μ best + maxAll (-μ)
    change -μ a ≤ maxAll (-μ) at hu
    linarith

omit [MeasurableSpace A] [MeasurableSingletonClass A] in
theorem maximumGap_pos (μ : Means A) (best : A) [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best) : 0 < maximumGap μ best := by
  obtain ⟨a, ha⟩ := (inferInstance : Nonempty {a : A // a ≠ best})
  exact (minimumGap_pos μ best hbest).trans_le
    ((gap_bounds μ best a ha).1.trans (gap_bounds μ best a ha).2)

omit [Nonempty A] [DecidableEq A] in
theorem finite_value_integrable (ν : Measure Ω) [IsFiniteMeasure ν]
    (f : A → ℝ) (J : Ω → A) (hJ : Measurable J) :
    Integrable (fun w => f (J w)) ν := by
  have hm := (measurable_of_countable f).comp hJ
  apply (integrable_const (∑ a, |f a|)).mono' hm.aestronglyMeasurable
  exact Filter.Eventually.of_forall (fun w => by
    rw [Real.norm_eq_abs]
    exact Finset.single_le_sum (fun a _ => abs_nonneg (f a)) (Finset.mem_univ (J w)))

def recommendationError (ν : Measure Ω) (best : A) (J : Ω → A) : ℝ :=
  ν.real {w | J w ≠ best}

def frequentistRegret (ν : Measure Ω) (μ : Means A) (best : A) (J : Ω → A) : ℝ :=
  ∫ w, μ best - μ (J w) ∂ν

/-- At every finite unique-best instance, simple regret is sandwiched
between the minimum and maximum gaps times the actual error probability. -/
theorem error_regret_sandwich (ν : Measure Ω) [IsProbabilityMeasure ν]
    (μ : Means A) (best : A) [Nonempty {a : A // a ≠ best}]
    (_hbest : ∀ a, a ≠ best → μ a < μ best) (J : Ω → A) (hJ : Measurable J) :
    minimumGap μ best * recommendationError ν best J ≤ frequentistRegret ν μ best J ∧
      frequentistRegret ν μ best J ≤ maximumGap μ best * recommendationError ν best J := by
  let E : Set Ω := {w | J w ≠ best}
  have hE : MeasurableSet E := (hJ (measurableSet_singleton best)).compl
  have hi := finite_value_integrable ν (fun a => μ best - μ a) J hJ
  have hlo := integral_mono ((integrable_const (minimumGap μ best)).indicator hE) hi
    (fun w => by
      by_cases hw : w ∈ E
      · rw [indicator_of_mem hw]
        exact (gap_bounds μ best (J w) hw).1
      · have he : J w = best := not_not.mp hw
        simp only [indicator_of_notMem hw, he, sub_self, le_refl])
  have hhi := integral_mono hi ((integrable_const (maximumGap μ best)).indicator hE)
    (fun w => by
      by_cases hw : w ∈ E
      · rw [indicator_of_mem hw]
        exact (gap_bounds μ best (J w) hw).2
      · have he : J w = best := not_not.mp hw
        simp only [indicator_of_notMem hw, he, sub_self, le_refl])
  rw [integral_indicator_const _ hE, smul_eq_mul, mul_comm] at hlo hhi
  exact ⟨hlo, hhi⟩

theorem liminf_add_vanishing_coe (u : ℕ → EReal) (v : ℕ → ℝ)
    (hv : Tendsto v atTop (𝓝 0)) :
    liminf (fun T => u T + (v T : EReal)) atTop = liminf u atTop := by
  have hvc : Tendsto (fun T => (v T : EReal)) atTop (𝓝 0) := by
    simpa only [EReal.coe_zero] using EReal.tendsto_coe.mpr hv
  have hlo := EReal.le_liminf_add (u := u) (v := fun T => (v T : EReal)) (f := atTop)
  rw [hvc.liminf_eq, add_zero] at hlo
  have hhi := EReal.liminf_add_le (u := fun T => (v T : EReal)) (v := u) (f := atTop)
    (Or.inl (by rw [hvc.limsup_eq]; exact EReal.coe_ne_bot 0))
    (Or.inl (by rw [hvc.limsup_eq]; exact EReal.coe_ne_top 0))
  rw [hvc.limsup_eq, zero_add] at hhi
  have hhi' : liminf (fun T => u T + (v T : EReal)) atTop ≤ liminf u atTop := by
    simpa only [Pi.add_apply, add_comm] using hhi
  exact le_antisymm hhi' hlo

theorem limsup_add_vanishing_coe (u : ℕ → EReal) (v : ℕ → ℝ)
    (hv : Tendsto v atTop (𝓝 0)) :
    limsup (fun T => u T + (v T : EReal)) atTop = limsup u atTop := by
  have hvc : Tendsto (fun T => (v T : EReal)) atTop (𝓝 0) := by
    simpa only [EReal.coe_zero] using EReal.tendsto_coe.mpr hv
  have hlo := EReal.le_limsup_add (u := u) (v := fun T => (v T : EReal)) (f := atTop)
  rw [hvc.liminf_eq, add_zero] at hlo
  have hhi := EReal.limsup_add_le (u := u) (v := fun T => (v T : EReal)) (f := atTop)
    (Or.inr (by rw [hvc.limsup_eq]; exact EReal.coe_ne_top 0))
    (Or.inr (by rw [hvc.limsup_eq]; exact EReal.coe_ne_bot 0))
  rw [hvc.limsup_eq, add_zero] at hhi
  exact le_antisymm hhi hlo

theorem rate_le_of_scaled_lower (p r : ℕ → ℝ) (a : ℝ) (ha : 0 < a)
    (hp : ∀ T, 0 ≤ p T) (_hr : ∀ T, 0 ≤ r T)
    (hl : ∀ T, a * p T ≤ r T) (T : ℕ) (hT : 0 < T) :
    Junpei.RegretExponent.rate r T ≤ Junpei.RegretExponent.rate p T +
      (((-Real.log a) / (T : ℝ) : ℝ) : EReal) := by
  by_cases hpz : p T = 0
  · simp [Junpei.RegretExponent.rate, hpz]
  · have hpp : 0 < p T := lt_of_le_of_ne (hp T) (Ne.symm hpz)
    have hrp : 0 < r T := (mul_pos ha hpp).trans_le (hl T)
    have hlog := Real.log_le_log (mul_pos ha hpp) (hl T)
    rw [Real.log_mul ha.ne' hpp.ne'] at hlog
    rw [Junpei.RegretExponent.rate, if_neg hrp.ne',
      Junpei.RegretExponent.rate, if_neg hpz, ← EReal.coe_add]
    apply EReal.coe_le_coe
    rw [← add_div]
    apply div_le_div_of_nonneg_right _ (by exact_mod_cast hT.le)
    linarith

/-- Multiplication by fixed positive gap constants preserves the full
extended-real exponential rate, including horizons with zero error. -/
theorem liminf_rate_eq_of_gap_sandwich (p r : ℕ → ℝ) (a b : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hp : ∀ T, 0 ≤ p T) (hr : ∀ T, 0 ≤ r T)
    (hl : ∀ T, a * p T ≤ r T) (hu : ∀ T, r T ≤ b * p T) :
    liminf (Junpei.RegretExponent.rate p) atTop =
      liminf (Junpei.RegretExponent.rate r) atTop := by
  have hlow : ∀ T, b⁻¹ * r T ≤ p T := by
    intro T
    exact (inv_mul_le_iff₀ hb).mpr (hu T)
  have hpr := liminf_le_liminf (show ∀ᶠ T : ℕ in atTop,
      Junpei.RegretExponent.rate p T ≤ Junpei.RegretExponent.rate r T +
        (((-Real.log b⁻¹) / (T : ℝ) : ℝ) : EReal) by
    filter_upwards [eventually_ge_atTop (1 : ℕ)] with T hT
    exact rate_le_of_scaled_lower r p b⁻¹ (inv_pos.mpr hb) hr hp hlow T (by omega))
  have hrp := liminf_le_liminf (show ∀ᶠ T : ℕ in atTop,
      Junpei.RegretExponent.rate r T ≤ Junpei.RegretExponent.rate p T +
        (((-Real.log a) / (T : ℝ) : ℝ) : EReal) by
    filter_upwards [eventually_ge_atTop (1 : ℕ)] with T hT
    exact rate_le_of_scaled_lower p r a ha hp hr hl T (by omega))
  rw [liminf_add_vanishing_coe _ _ (tendsto_const_div_atTop_nhds_zero_nat _)] at hpr hrp
  exact le_antisymm hpr hrp

theorem limsup_rate_eq_of_gap_sandwich (p r : ℕ → ℝ) (a b : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hp : ∀ T, 0 ≤ p T) (hr : ∀ T, 0 ≤ r T)
    (hl : ∀ T, a * p T ≤ r T) (hu : ∀ T, r T ≤ b * p T) :
    limsup (Junpei.RegretExponent.rate p) atTop =
      limsup (Junpei.RegretExponent.rate r) atTop := by
  have hlow : ∀ T, b⁻¹ * r T ≤ p T := by
    intro T
    exact (inv_mul_le_iff₀ hb).mpr (hu T)
  have hpr := limsup_le_limsup (show ∀ᶠ T : ℕ in atTop,
      Junpei.RegretExponent.rate p T ≤ Junpei.RegretExponent.rate r T +
        (((-Real.log b⁻¹) / (T : ℝ) : ℝ) : EReal) by
    filter_upwards [eventually_ge_atTop (1 : ℕ)] with T hT
    exact rate_le_of_scaled_lower r p b⁻¹ (inv_pos.mpr hb) hr hp hlow T (by omega))
  have hrp := limsup_le_limsup (show ∀ᶠ T : ℕ in atTop,
      Junpei.RegretExponent.rate r T ≤ Junpei.RegretExponent.rate p T +
        (((-Real.log a) / (T : ℝ) : ℝ) : EReal) by
    filter_upwards [eventually_ge_atTop (1 : ℕ)] with T hT
    exact rate_le_of_scaled_lower p r a ha hp hr hl T (by omega))
  rw [limsup_add_vanishing_coe _ _ (tendsto_const_div_atTop_nhds_zero_nat _)] at hpr hrp
  exact le_antisymm hpr hrp

/-- The unnumbered exponential-rate equivalence in Section 6 for any
measurable horizon-indexed family of recommendations at a unique-best
finite-arm instance. No positive-error assumption excludes zero horizons. -/
theorem error_regret_same_exponential_rate (ν : Measure Ω) [IsProbabilityMeasure ν]
    (μ : Means A) (best : A) [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best)
    (J : ℕ → Ω → A) (hJ : ∀ T, Measurable (J T)) :
    liminf (Junpei.RegretExponent.rate (fun T => recommendationError ν best (J T))) atTop =
      liminf (Junpei.RegretExponent.rate (fun T => frequentistRegret ν μ best (J T))) atTop := by
  apply liminf_rate_eq_of_gap_sandwich _ _ (minimumGap μ best) (maximumGap μ best)
    (minimumGap_pos μ best hbest) (maximumGap_pos μ best hbest)
    (fun _ => measureReal_nonneg)
  · intro T
    exact ((mul_nonneg (minimumGap_pos μ best hbest).le measureReal_nonneg).trans
      (error_regret_sandwich ν μ best hbest (J T) (hJ T)).1)
  · intro T
    exact (error_regret_sandwich ν μ best hbest (J T) (hJ T)).1
  · intro T
    exact (error_regret_sandwich ν μ best hbest (J T) (hJ T)).2

/-- The upper exponential rate agrees as well; together with the preceding
theorem this covers existence and equality of an exponential-rate limit. -/
theorem error_regret_same_upper_exponential_rate (ν : Measure Ω) [IsProbabilityMeasure ν]
    (μ : Means A) (best : A) [Nonempty {a : A // a ≠ best}]
    (hbest : ∀ a, a ≠ best → μ a < μ best)
    (J : ℕ → Ω → A) (hJ : ∀ T, Measurable (J T)) :
    limsup (Junpei.RegretExponent.rate (fun T => recommendationError ν best (J T))) atTop =
      limsup (Junpei.RegretExponent.rate (fun T => frequentistRegret ν μ best (J T))) atTop := by
  apply limsup_rate_eq_of_gap_sandwich _ _ (minimumGap μ best) (maximumGap μ best)
    (minimumGap_pos μ best hbest) (maximumGap_pos μ best hbest)
    (fun _ => measureReal_nonneg)
  · intro T
    exact ((mul_nonneg (minimumGap_pos μ best hbest).le measureReal_nonneg).trans
      (error_regret_sandwich ν μ best hbest (J T) (hJ T)).1)
  · intro T
    exact (error_regret_sandwich ν μ best hbest (J T) (hJ T)).1
  · intro T
    exact (error_regret_sandwich ν μ best hbest (J T) (hJ T)).2

end
end Junpei.ProseCorollaries
