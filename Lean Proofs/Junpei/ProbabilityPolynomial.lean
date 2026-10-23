import Junpei.ProbabilityModel

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Probability

open Draft0 BadEvent

noncomputable section

def underestimationInterval (T : ℕ) (CU : ℝ) : Set ℝ :=
  Icc (-CU * Real.sqrt (Real.log (T : ℝ)) - 1) (-CU * Real.sqrt (Real.log (T : ℝ)))

theorem underestimationPDF_polynomial_lower (T B : ℕ) (Δ CU : ℝ)
    (hT : 5 ≤ T) (hΔ : 0 ≤ Δ) (hCU : 0 ≤ CU) (hB : CU + Δ + 1 ≤ (B : ℝ))
    (x : ℝ) (hx : x ∈ underestimationInterval T CU) :
    underestimationConstant * Rates.invPower (B ^ 2) T ≤ gaussianPDFReal Δ 1 x := by
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hlog := Drift.log_horizon_ge_one T hT
  have hsqrt : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have hBpos : 0 ≤ (B : ℝ) := by positivity
  have hBscale := mul_le_mul_of_nonneg_right hB (Real.sqrt_nonneg (Real.log (T : ℝ)))
  have hΔscale := mul_le_mul_of_nonneg_left hsqrt (by positivity : 0 ≤ Δ + 1)
  have hCUscale : 0 ≤ CU * Real.sqrt (Real.log (T : ℝ)) := by positivity
  have hxlower : -(B : ℝ) * Real.sqrt (Real.log (T : ℝ)) ≤ x - Δ := by
    change -CU * Real.sqrt (Real.log (T : ℝ)) - 1 ≤ x ∧
      x ≤ -CU * Real.sqrt (Real.log (T : ℝ)) at hx
    nlinarith only [hx.1, hBscale, hΔscale]
  have hxupper : x - Δ ≤ (B : ℝ) * Real.sqrt (Real.log (T : ℝ)) := by
    change _ ≤ x ∧ x ≤ -CU * Real.sqrt (Real.log (T : ℝ)) at hx
    nlinarith only [hx.2, hΔ, hCUscale, mul_nonneg hBpos (Real.sqrt_nonneg (Real.log (T : ℝ)))]
  have habs : |x - Δ| ≤ (B : ℝ) * Real.sqrt (Real.log (T : ℝ)) :=
    abs_le.mpr ⟨by simpa only [neg_mul] using hxlower, hxupper⟩
  have hsq : (x - Δ) ^ 2 ≤ (B : ℝ) ^ 2 * Real.log (T : ℝ) := by
    have h := (sq_le_sq₀ (abs_nonneg (x - Δ)) (by positivity)).mpr habs
    simpa only [sq_abs, mul_pow, Real.sq_sqrt (by linarith : 0 ≤ Real.log (T : ℝ))] using h
  have hlogprod : 0 ≤ (B : ℝ) ^ 2 * Real.log (T : ℝ) := by positivity
  have he : Real.exp (-((B ^ 2 : ℕ) : ℝ) * Real.log (T : ℝ)) ≤
      Real.exp (-(x - Δ) ^ 2 / 2) := by
    apply Real.exp_le_exp.mpr
    norm_num only [Nat.cast_pow]
    nlinarith only [hsq, hlogprod]
  unfold Rates.invPower
  rw [← PredictiveBound.exp_neg_nat_log (B ^ 2) hTp]
  have hc : 0 ≤ underestimationConstant := underestimationConstant_positive.le
  convert mul_le_mul_of_nonneg_left he hc using 1
  simp [gaussianPDFReal, underestimationConstant]

theorem underestimation_probability_polynomial_lower (T B : ℕ) (Δ CU : ℝ)
    (hT : 5 ≤ T) (hΔ : 0 ≤ Δ) (hCU : 0 ≤ CU) (hB : CU + Δ + 1 ≤ (B : ℝ)) :
    underestimationConstant * Rates.invPower (B ^ 2) T ≤
      (gaussianReal Δ 1).real (underestimationSet T CU) := by
  have hmass : (gaussianReal Δ 1).real (underestimationInterval T CU) =
      ∫ x in underestimationInterval T CU, gaussianPDFReal Δ 1 x := by
    rw [Measure.real, gaussianReal_apply_eq_integral Δ (by norm_num), ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg Δ 1)
  have hmono : (∫ _x in underestimationInterval T CU,
      underestimationConstant * Rates.invPower (B ^ 2) T) ≤
      ∫ x in underestimationInterval T CU, gaussianPDFReal Δ 1 x := by
    apply setIntegral_mono_on (integrableOn_const isCompact_Icc.measure_ne_top)
      (integrable_gaussianPDFReal Δ 1).integrableOn measurableSet_Icc
    intro x hx
    exact underestimationPDF_polynomial_lower T B Δ CU hT hΔ hCU hB x hx
  have hconst : (∫ _x in underestimationInterval T CU,
      underestimationConstant * Rates.invPower (B ^ 2) T) =
      underestimationConstant * Rates.invPower (B ^ 2) T := by
    rw [underestimationInterval, setIntegral_const, smul_eq_mul, measureReal_def,
      Real.volume_Icc, ENNReal.toReal_ofReal (by linarith)]
    ring
  rw [hconst, ← hmass] at hmono
  exact hmono.trans (measureReal_mono (by
    intro x hx
    exact hx.2))

theorem goodArmSet3_polynomial_lower (T q : ℕ)
    (hT : 16 ≤ T) (horizon : T = 2 * q + 1) :
    (1 / 8 : ℝ) * Rates.invPower 2 T ≤ (arrayLaw q).real (goodArmSet3 T q) := by
  have hTr : (16 : ℝ) ≤ T := by exact_mod_cast hT
  have hTp : 0 < (T : ℝ) := by linarith
  have hlog : 0 ≤ Real.log (T : ℝ) := Real.log_nonneg (by linarith)
  have hsmall : f3 T ≤ 2 * Rates.invPower 3 T := by
    unfold f3 Rates.invPower
    rw [← PredictiveBound.exp_neg_nat_log 3 hTp]
    apply mul_le_mul_of_nonneg_left _ (by norm_num)
    apply Real.exp_le_exp.mpr
    norm_num
    nlinarith
  have hrate : 2 * Rates.invPower 3 T ≤ (1 / 8 : ℝ) * Rates.invPower 2 T := by
    unfold Rates.invPower
    field_simp [ne_of_gt hTp]
    nlinarith only [hTr]
  have hf2 := f2_polynomial_lower T (by omega)
  have hprob := goodArmSet3_probability_lower T q horizon (by omega)
  linarith

theorem rawBadEvent_polynomial_lower (T q B : ℕ) (Δ CU : ℝ)
    (hT : 16 ≤ T) (horizon : T = 2 * q + 1)
    (hΔ : 0 ≤ Δ) (hCU : 0 ≤ CU) (hB : CU + Δ + 1 ≤ (B : ℝ)) :
    (underestimationConstant / 64) * Rates.invPower (B ^ 2 + 4) T ≤
      (rawArraysLaw T).real (rawBadEvent T q Δ CU (by omega) (by omega)) := by
  rw [rawBadEvent_probability_eq]
  have hX := underestimation_probability_polynomial_lower T B Δ CU (by omega) hΔ hCU hB
  have hY := goodArmSet3_polynomial_lower T q hT horizon
  have hY0 : 0 ≤ (1 / 8 : ℝ) * Rates.invPower 2 T :=
    mul_nonneg (by norm_num) (Rates.invPower_nonneg 2 T)
  have hYs := (sq_le_sq₀ hY0 (measureReal_nonneg)).mpr hY
  have hm := mul_le_mul hX hYs (sq_nonneg _) measureReal_nonneg
  have he : (underestimationConstant / 64) * Rates.invPower (B ^ 2 + 4) T =
      (underestimationConstant * Rates.invPower (B ^ 2) T) *
        ((1 / 8 : ℝ) * Rates.invPower 2 T) ^ 2 := by
    unfold Rates.invPower
    rw [pow_add, mul_inv]
    ring
  rw [he]
  exact hm

end

end Junpei.Probability
