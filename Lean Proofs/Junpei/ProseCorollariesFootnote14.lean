import Junpei.ProbabilityPolynomial

set_option autoImplicit false
set_option warningAsError true

open MeasureTheory ProbabilityTheory Set Filter Asymptotics
open scoped NNReal ENNReal

namespace Junpei.ProseCorollaries.Footnote14

noncomputable section

/-- The standard deviation corresponding to the footnote's variance T^-3. -/
def standardDeviation (T : ℕ) : ℝ := (Real.sqrt ((T : ℝ) ^ 3))⁻¹

/-- A threshold larger than every fixed multiple of that standard deviation. -/
def logarithmicGap (T : ℕ) : ℝ := standardDeviation T * Real.sqrt (Real.log (T : ℝ))

theorem standardDeviation_pos (T : ℕ) (hT : 0 < T) : 0 < standardDeviation T := by
  unfold standardDeviation
  have hp : 0 < (T : ℝ) := by exact_mod_cast hT
  positivity

theorem standardDeviation_sq (T : ℕ) :
    standardDeviation T ^ 2 = ((T : ℝ) ^ 3)⁻¹ := by
  unfold standardDeviation
  simpa only [inv_pow] using congrArg Inv.inv (Real.sq_sqrt
    (show 0 ≤ (T : ℝ) ^ 3 by positivity))

/-- A positive polynomial lower bound still holds for the larger threshold
sqrt(log T) times T^-3/2, under the actual Gaussian law of variance T^-3. -/
theorem logarithmicGap_polynomial_tail (T : ℕ) (hT : 5 ≤ T) :
    Draft0.BadEvent.underestimationConstant * Draft0.Rates.invPower 4 T ≤
      (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (logarithmicGap T)) := by
  have hσ := standardDeviation_pos T (by omega)
  have hv : (⟨(-standardDeviation T) ^ 2, sq_nonneg _⟩ : ℝ≥0) * 1 =
      ((T : ℝ≥0) ^ 3)⁻¹ := by
    apply NNReal.coe_injective
    simp only [NNReal.coe_mk, mul_one,
      neg_sq, standardDeviation_sq, NNReal.coe_inv, NNReal.coe_pow, NNReal.coe_natCast]
  have hm := gaussianReal_map_const_mul (μ := (0 : ℝ)) (v := (1 : ℝ≥0))
    (-standardDeviation T)
  rw [mul_zero, hv] at hm
  have hpre : (fun x : ℝ => -standardDeviation T * x) ⁻¹' Ici (logarithmicGap T) =
      Junpei.Probability.underestimationSet T 1 := by
    ext x
    change standardDeviation T * Real.sqrt (Real.log (T : ℝ)) ≤
      -standardDeviation T * x ↔ x ≤ -1 * Real.sqrt (Real.log (T : ℝ))
    constructor <;> intro hx <;> nlinarith
  rw [measureReal_def, ← hm, Measure.map_apply (by fun_prop) measurableSet_Ici, hpre]
  exact Junpei.Probability.underestimation_probability_polynomial_lower T 2 0 1 hT
    (by norm_num) (by norm_num) (by norm_num)

/-- Thus the footnote's claimed necessary O(T^-3/2) scale is stronger than
polynomial-tail probability requires. This is a kernel-checked counterexample,
not a claim about whether the main theorem's enlarged event would work. -/
theorem logarithmicGap_not_bigO_standardDeviation :
    ¬ (logarithmicGap =O[atTop] standardDeviation) := by
  intro h
  obtain ⟨C, _hC, hb⟩ := Asymptotics.isBigO_iff'.mp h
  have hlog : ∀ᶠ T : ℕ in atTop, (C + 1) ^ 2 ≤ Real.log (T : ℝ) :=
    (Real.tendsto_log_atTop.comp tendsto_natCast_atTop_atTop).eventually_ge_atTop _
  obtain ⟨T, hbound, hlarge, hT⟩ :=
    (hb.and (hlog.and (eventually_ge_atTop (5 : ℕ)))).exists
  have hσ := standardDeviation_pos T (by omega)
  have hroot : C + 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.le_sqrt_of_sq_le hlarge
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (by
    unfold logarithmicGap
    exact mul_nonneg hσ.le (Real.sqrt_nonneg _)), abs_of_pos hσ] at hbound
  unfold logarithmicGap at hbound
  have hsmall : Real.sqrt (Real.log (T : ℝ)) ≤ C := by
    apply (mul_le_mul_iff_right₀ hσ).mp
    simpa only [mul_comm] using hbound
  linarith

/-- The counterexample combines the positive polynomial constant, its
all-large-integer-horizon tail bound, and failure of the alleged necessary
Big-O condition in one explicit formal statement. -/
theorem counterexample_to_necessary_scale :
    (0 < Draft0.BadEvent.underestimationConstant) ∧
      (∀ T : ℕ, 5 ≤ T →
        Draft0.BadEvent.underestimationConstant * Draft0.Rates.invPower 4 T ≤
          (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (logarithmicGap T))) ∧
      ¬ (logarithmicGap =O[atTop] standardDeviation) :=
  ⟨Draft0.BadEvent.underestimationConstant_positive,
    logarithmicGap_polynomial_tail, logarithmicGap_not_bigO_standardDeviation⟩

end
end Junpei.ProseCorollaries.Footnote14
