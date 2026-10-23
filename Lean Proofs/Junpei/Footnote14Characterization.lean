import Junpei.Footnote14Scale
import Junpei.UniformSamplingTail

set_option autoImplicit false
set_option warningAsError true

open MeasureTheory ProbabilityTheory Set Filter Asymptotics
open scoped NNReal ENNReal

namespace Junpei.Footnote14Characterization

noncomputable section

open Junpei.ProseCorollaries.Footnote14 Junpei.TailRatio

/-- Polynomial lower bound at all sufficiently large integer horizons. -/
def EventuallyPolynomialTail (gap : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, 0 < c ∧ ∃ p : ℕ, ∀ᶠ T : ℕ in atTop,
    c * Draft0.Rates.invPower p T ≤
      (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (gap T))

theorem tail_standardize (T : ℕ) (hT : 0 < T) (g : ℝ) :
    (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici g) =
      normalTail (g / standardDeviation T) := by
  have hσ := standardDeviation_pos T hT
  have hv : (⟨standardDeviation T ^ 2, sq_nonneg _⟩ : ℝ≥0) * 1 =
      ((T : ℝ≥0) ^ 3)⁻¹ := by
    apply NNReal.coe_injective
    simp only [NNReal.coe_mk, mul_one, standardDeviation_sq,
      NNReal.coe_inv, NNReal.coe_pow, NNReal.coe_natCast]
  have hm := gaussianReal_map_const_mul (μ := (0 : ℝ)) (v := (1 : ℝ≥0))
    (standardDeviation T)
  rw [mul_zero, hv] at hm
  have hp : (fun x : ℝ => standardDeviation T * x) ⁻¹' Ici g =
      Ici (g / standardDeviation T) := by
    ext x
    change g ≤ standardDeviation T * x ↔ g / standardDeviation T ≤ x
    rw [div_le_iff₀ hσ]
    rw [mul_comm]
  rw [measureReal_def, ← hm, Measure.map_apply (by fun_prop) measurableSet_Ici, hp]
  haveI : NoAtoms (gaussianReal 0 1) := noAtoms_gaussianReal (by norm_num : (1 : ℝ≥0) ≠ 0)
  exact congrArg ENNReal.toReal
    (measure_congr (Ioi_ae_eq_Ici (μ := gaussianReal 0 1)
      (a := g / standardDeviation T))).symm

theorem scaled_logarithmic_tail_lower (T B : ℕ) (C : ℝ) (hT : 5 ≤ T)
    (hC : 0 ≤ C) (hB : C + 1 ≤ (B : ℝ)) :
    Draft0.BadEvent.underestimationConstant * Draft0.Rates.invPower (B ^ 2) T ≤
      (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (C * logarithmicGap T)) := by
  rw [tail_standardize T (by omega)]
  have he : C * logarithmicGap T / standardDeviation T =
      C * Real.sqrt (Real.log (T : ℝ)) := by
    unfold logarithmicGap
    field_simp [(standardDeviation_pos T (by omega)).ne']
  rw [he]
  have hl := Junpei.Probability.underestimation_probability_polynomial_lower
    T B 0 C hT (by norm_num) hC (by simpa using hB)
  simpa only [Junpei.Probability.underestimationSet, neg_mul,
    Junpei.TailRatio.shifted_left_tail, add_zero] using hl

theorem eventuallyPolynomialTail_of_isBigO (gap : ℕ → ℝ)
    (hgap : gap =O[atTop] logarithmicGap) : EventuallyPolynomialTail gap := by
  obtain ⟨C, hC, hbound⟩ := Asymptotics.isBigO_iff'.mp hgap
  obtain ⟨B, hB⟩ := exists_nat_ge (C + 1)
  refine ⟨Draft0.BadEvent.underestimationConstant,
    Draft0.BadEvent.underestimationConstant_positive, B ^ 2, ?_⟩
  filter_upwards [hbound, eventually_ge_atTop (5 : ℕ)] with T hbound hT
  have hg0 : 0 ≤ logarithmicGap T := by
    unfold logarithmicGap standardDeviation
    positivity
  have hg : gap T ≤ C * logarithmicGap T := by
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg hg0] at hbound
    exact (le_abs_self _).trans hbound
  exact (scaled_logarithmic_tail_lower T B C hT hC.le hB).trans
    (measureReal_mono (fun _ hx => hg.trans hx))

theorem standardized_threshold_sq_bound (T p : ℕ) (c g : ℝ)
    (hT : 5 ≤ T) (hc : 0 < c) (hg : 0 ≤ g)
    (hbound : c * Draft0.Rates.invPower p T ≤
      (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici g)) :
    (g / standardDeviation T) ^ 2 ≤
      (4 + 2 * ((p : ℝ) + |Real.log normalization| + |Real.log c|)) *
        Real.log (T : ℝ) := by
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hlog := Draft0.Drift.log_horizon_ge_one T hT
  have hx : 0 ≤ g / standardDeviation T :=
    div_nonneg hg (standardDeviation_pos T (by omega)).le
  rw [tail_standardize T (by omega)] at hbound
  by_cases hlarge : 2 ≤ g / standardDeviation T
  · have hpoly : 0 < c * Draft0.Rates.invPower p T :=
      mul_pos hc (Draft0.Rates.invPower_pos p T (by omega))
    have hlogs := Real.log_le_log hpoly hbound
    rw [Draft0.Rates.invPower, Real.log_mul hc.ne' (by positivity),
      Real.log_inv, Real.log_pow] at hlogs
    have htail := (Junpei.UniformSampling.tail_log_bounds
      (g / standardDeviation T) hlarge).1
    have hn := le_abs_self (Real.log normalization)
    have hc' := neg_le_abs (Real.log c)
    have hconst : |Real.log normalization| + |Real.log c| ≤
        (|Real.log normalization| + |Real.log c|) * Real.log (T : ℝ) :=
      le_mul_of_one_le_right (by positivity) hlog
    nlinarith
  · have hsmall : g / standardDeviation T < 2 := lt_of_not_ge hlarge
    have hcoef : 0 ≤ (p : ℝ) + |Real.log normalization| + |Real.log c| := by positivity
    nlinarith [mul_nonneg hcoef (show 0 ≤ Real.log (T : ℝ) by linarith)]

theorem isBigO_of_eventuallyPolynomialTail (gap : ℕ → ℝ)
    (hgap : ∀ T, 0 ≤ gap T) (hpoly : EventuallyPolynomialTail gap) :
    gap =O[atTop] logarithmicGap := by
  obtain ⟨c, hc, p, hbound⟩ := hpoly
  let K : ℝ := 4 + 2 * ((p : ℝ) + |Real.log normalization| + |Real.log c|)
  have hK : 0 ≤ K := by dsimp [K]; positivity
  apply Asymptotics.IsBigO.of_bound (Real.sqrt K)
  filter_upwards [hbound, eventually_ge_atTop (5 : ℕ)] with T hbound hT
  have hσ := standardDeviation_pos T (by omega)
  have hlog : 0 ≤ Real.log (T : ℝ) :=
    (by norm_num : (0 : ℝ) ≤ 1).trans (Draft0.Drift.log_horizon_ge_one T hT)
  have hsq := standardized_threshold_sq_bound T p c (gap T) hT hc (hgap T) hbound
  have hroot : gap T / standardDeviation T ≤
      Real.sqrt K * Real.sqrt (Real.log (T : ℝ)) := by
    have he : (Real.sqrt K * Real.sqrt (Real.log (T : ℝ))) ^ 2 =
        K * Real.log (T : ℝ) := by rw [mul_pow, Real.sq_sqrt hK, Real.sq_sqrt hlog]
    have hn : 0 ≤ Real.sqrt K * Real.sqrt (Real.log (T : ℝ)) := by positivity
    nlinarith
  have hmul := (div_le_iff₀ hσ).mp hroot
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (hgap T),
    abs_of_nonneg (show 0 ≤ logarithmicGap T by unfold logarithmicGap; positivity)]
  unfold logarithmicGap
  nlinarith only [hmul]

/-- Exact asymptotic characterization for nonnegative thresholds under the actual
Gaussian law with variance T^-3. Polynomial means a positive polynomial lower
bound, with unrestricted constants and exponent at all sufficiently large T. -/
theorem eventuallyPolynomialTail_iff_isBigO (gap : ℕ → ℝ)
    (hgap : ∀ T, 0 ≤ gap T) :
    EventuallyPolynomialTail gap ↔
      gap =O[atTop] (fun T : ℕ => (T : ℝ) ^ (-(3 : ℝ) / 2) *
        Real.sqrt (Real.log (T : ℝ))) := by
  have he : logarithmicGap = (fun T : ℕ => (T : ℝ) ^ (-(3 : ℝ) / 2) *
        Real.sqrt (Real.log (T : ℝ))) := by
    funext T
    rw [logarithmicGap, Junpei.Footnote14Scale.standardDeviation_eq_paper_scale]
  rw [← he]
  exact ⟨isBigO_of_eventuallyPolynomialTail gap hgap,
    eventuallyPolynomialTail_of_isBigO gap⟩

theorem finite_positive_lower_bound (s : Finset ℕ) (f : ℕ → ℝ)
    (hf : ∀ n ∈ s, 0 < f n) : ∃ c : ℝ, 0 < c ∧ ∀ n ∈ s, c ≤ f n := by
  induction s using Finset.induction_on with
  | empty => exact ⟨1, by norm_num, by simp⟩
  | @insert a s _ ih =>
    obtain ⟨c, hc, hb⟩ := ih (fun n hn => hf n (Finset.mem_insert_of_mem hn))
    refine ⟨min c (f a), lt_min hc (hf a (Finset.mem_insert_self a s)), ?_⟩
    intro n hn
    rcases Finset.mem_insert.mp hn with rfl | hn
    · exact min_le_right _ _
    · exact (min_le_left _ _).trans (hb n hn)

/-- Since every finite threshold has positive Gaussian tail probability, the
finite prefix can be absorbed into the positive polynomial constant. -/
theorem polynomialTailLowerBound_iff_eventually (gap : ℕ → ℝ) :
    Junpei.Footnote14Scale.PolynomialTailLowerBound gap ↔ EventuallyPolynomialTail gap := by
  constructor
  · rintro ⟨c, hc, p, hb⟩
    exact ⟨c, hc, p, (eventually_ge_atTop 5).mono (fun T hT => hb T hT)⟩
  · rintro ⟨c, hc, p, hb⟩
    obtain ⟨N, hN⟩ := eventually_atTop.mp hb
    let f : ℕ → ℝ := fun T => if 5 ≤ T then
      (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (gap T)) * (T : ℝ) ^ p else 1
    have hf : ∀ T ∈ Finset.range N, 0 < f T := by
      intro T _hT
      dsimp [f]
      split_ifs with hT
      · rw [tail_standardize T (by omega)]
        exact mul_pos (Junpei.UniformSampling.normalTail_pos _)
          (pow_pos (by exact_mod_cast (show 0 < T by omega)) _)
      · norm_num
    obtain ⟨d, hd, hsmall⟩ := finite_positive_lower_bound (Finset.range N) f hf
    refine ⟨min c d, lt_min hc hd, p, ?_⟩
    intro T hT
    by_cases hTN : N ≤ T
    · exact (mul_le_mul_of_nonneg_right (min_le_left c d)
        (Draft0.Rates.invPower_nonneg p T)).trans (hN T hTN)
    · have hfT := hsmall T (Finset.mem_range.mpr (by omega))
      dsimp [f] at hfT
      rw [if_pos hT] at hfT
      have hTp : 0 < (T : ℝ) ^ p := pow_pos
        (by exact_mod_cast (show 0 < T by omega)) _
      unfold Draft0.Rates.invPower
      rw [← div_eq_mul_inv]
      exact (div_le_iff₀ hTp).mpr ((min_le_right c d).trans hfT)

/-- The same exact characterization using the original definition whose lower
bound is required for every integer T ≥ 5, not just eventually. -/
theorem polynomialTailLowerBound_iff_isBigO (gap : ℕ → ℝ)
    (hgap : ∀ T, 0 ≤ gap T) :
    Junpei.Footnote14Scale.PolynomialTailLowerBound gap ↔
      gap =O[atTop] (fun T : ℕ => (T : ℝ) ^ (-(3 : ℝ) / 2) *
        Real.sqrt (Real.log (T : ℝ))) := by
  rw [polynomialTailLowerBound_iff_eventually, eventuallyPolynomialTail_iff_isBigO gap hgap]

/-- The smaller O(T^-3/2) scale is sufficient, although the separately proved
counterexample shows it is not necessary for a polynomial tail lower bound. -/
theorem polynomialTailLowerBound_of_original_scale (gap : ℕ → ℝ)
    (hgap : gap =O[atTop] (fun T : ℕ => (T : ℝ) ^ (-(3 : ℝ) / 2))) :
    Junpei.Footnote14Scale.PolynomialTailLowerBound gap := by
  apply (polynomialTailLowerBound_iff_eventually gap).mpr
  apply eventuallyPolynomialTail_of_isBigO gap
  apply hgap.trans
  apply Asymptotics.IsBigO.of_bound 1
  filter_upwards [eventually_ge_atTop (5 : ℕ)] with T hT
  rw [← Junpei.Footnote14Scale.standardDeviation_eq_paper_scale,
    Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_pos (standardDeviation_pos T (by omega)),
    abs_of_nonneg (show 0 ≤ logarithmicGap T by
      unfold logarithmicGap standardDeviation
      positivity), one_mul]
  exact le_mul_of_one_le_right (standardDeviation_pos T (by omega)).le
    (Real.one_le_sqrt.mpr (Draft0.Drift.log_horizon_ge_one T hT))

/-- A positive real exponent and an explicit starting horizon give the same
polynomial lower-bound condition as an eventual bound with a natural exponent.
Increasing the exponent handles a natural exponent equal to zero. -/
theorem realExponentPolynomialTail_iff_eventually (gap : ℕ → ℝ) :
    (∃ c : ℝ, 0 < c ∧ ∃ p : ℝ, 0 < p ∧ ∃ T0 : ℕ, ∀ T : ℕ, T0 ≤ T →
      c * (T : ℝ) ^ (-p) ≤
        (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (gap T))) ↔
      EventuallyPolynomialTail gap := by
  constructor
  · rintro ⟨c, hc, p, _hp, T0, hbound⟩
    obtain ⟨n, hn⟩ := exists_nat_ge p
    refine ⟨c, hc, n, ?_⟩
    filter_upwards [eventually_ge_atTop T0, eventually_ge_atTop (1 : ℕ)] with T hT hT1
    have hpow : Draft0.Rates.invPower n T ≤ (T : ℝ) ^ (-p) := by
      unfold Draft0.Rates.invPower
      rw [← Real.rpow_natCast, ← Real.rpow_neg (Nat.cast_nonneg T)]
      exact Real.rpow_le_rpow_of_exponent_le (by exact_mod_cast hT1) (neg_le_neg hn)
    exact (mul_le_mul_of_nonneg_left hpow hc.le).trans (hbound T hT)
  · rintro ⟨c, hc, n, hbound⟩
    obtain ⟨N, hN⟩ := eventually_atTop.mp hbound
    refine ⟨c, hc, (n : ℝ) + 1, by positivity, max N 1, ?_⟩
    intro T hT
    have hT1 : 1 ≤ T := (le_max_right N 1).trans hT
    have hpow : (T : ℝ) ^ (-((n : ℝ) + 1)) ≤ Draft0.Rates.invPower n T := by
      unfold Draft0.Rates.invPower
      rw [← Real.rpow_natCast, ← Real.rpow_neg (Nat.cast_nonneg T)]
      exact Real.rpow_le_rpow_of_exponent_le (by exact_mod_cast hT1) (by linarith)
    exact (mul_le_mul_of_nonneg_left hpow hc.le).trans
      (hN T ((le_max_left N 1).trans hT))

/-- Exact corrected footnote statement: for every nonnegative threshold sequence,
a positive polynomial lower bound with real constants c > 0 and p > 0, valid
from an explicit natural horizon T0 onward, is equivalent to the sharp
O(T^(-3/2) sqrt(log T)) scale. The probability is the actual Gaussian law. -/
theorem corrected_footnote14_iff (d : ℕ → ℝ) (hd : ∀ T, 0 ≤ d T) :
    (∃ c : ℝ, 0 < c ∧ ∃ p : ℝ, 0 < p ∧ ∃ T0 : ℕ, ∀ T : ℕ, T0 ≤ T →
      c * (T : ℝ) ^ (-p) ≤
        (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (d T))) ↔
      d =O[atTop] (fun T : ℕ => (T : ℝ) ^ (-(3 : ℝ) / 2) *
        Real.sqrt (Real.log (T : ℝ))) := by
  rw [realExponentPolynomialTail_iff_eventually,
    eventuallyPolynomialTail_iff_isBigO d hd]

end
end Junpei.Footnote14Characterization
