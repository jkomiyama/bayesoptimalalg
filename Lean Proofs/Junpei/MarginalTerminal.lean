import Junpei.Marginal
import Draft0.Drift
import Draft0.PredictiveBound

open MeasureTheory ProbabilityTheory Set Filter Asymptotics
open scoped MeasureTheory NNReal ENNReal

namespace Junpei.Marginal

noncomputable section

open Draft0.Gaussian Draft0.Rates Draft0.TerminalBounds

def terminalConstant : ℝ := (Real.sqrt (4 * Real.pi))⁻¹

theorem terminalConstant_pos : 0 < terminalConstant := by
  unfold terminalConstant
  positivity

theorem updateVariance_half_budget_lower (T q n : ℕ)
    (hT : T = 2 * q + 1) (hn : 1 ≤ n) (hnq : n ≤ q) :
    4 * invPower 2 T ≤ (updateVariance (n : ℝ≥0) : ℝ) := by
  have hnr : 1 ≤ (n : ℝ) := by exact_mod_cast hn
  have hnqr : (n : ℝ) ≤ q := by exact_mod_cast hnq
  have hTr : (T : ℝ) = 2 * (q : ℝ) + 1 := by exact_mod_cast hT
  have hTpos : 0 < (T : ℝ) := by nlinarith only [hnr, hnqr, hTr]
  have hnprod : (n : ℝ) * ((n : ℝ) + 1) ≤ (q : ℝ) * ((q : ℝ) + 1) := by
    nlinarith only [hnr, hnqr]
  simp only [updateVariance, NNReal.coe_inv, NNReal.coe_mul, NNReal.coe_add,
    NNReal.coe_natCast, NNReal.coe_one, invPower]
  rw [← div_eq_mul_inv, inv_eq_one_div]
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith only [hnprod, hTr]

/-- The revised terminal large-gap range has density at least c*T^-18
on [d+T^-2,d+2*T^-2]. -/
theorem terminal_gaussianPDF_lower22 (T : ℕ) (gap x : ℝ) (v : ℝ≥0)
    (hT : 5 ≤ T) (hgap : |gap| ≤ 10 * Real.sqrt (Real.log (T : ℝ)) / T)
    (hv : 0 < v) (hvu : (v : ℝ) ≤ 2) (hvl : 4 * invPower 2 T ≤ (v : ℝ))
    (hx : x ∈ Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T)) :
    terminalConstant * invPower 18 T ≤ gaussianPDFReal 0 v x := by
  have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT
  have hTp : 0 < (T : ℝ) := by linarith only [hTr]
  have hlog : 1 ≤ Real.log (T : ℝ) := Draft0.Drift.log_horizon_ge_one T hT
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith only [hlog]
  have hs : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := by
    simpa using Real.sqrt_le_sqrt hlog
  have hi : 0 ≤ invPower 2 T := invPower_nonneg _ _
  have heps : invPower 2 T ≤ Real.sqrt (Real.log (T : ℝ)) / T := by
    unfold invPower
    rw [inv_eq_one_div]
    apply (div_le_div_iff₀ (by positivity) hTp).mpr
    nlinarith only [hs, hTr]
  have hx0 : 0 ≤ x := by linarith only [hx.1, abs_nonneg gap, hi]
  have hxu : x ≤ 12 * Real.sqrt (Real.log (T : ℝ)) / T := by
    calc
      x ≤ |gap| + 2 * invPower 2 T := hx.2
      _ ≤ 10 * Real.sqrt (Real.log (T : ℝ)) / T +
          2 * (Real.sqrt (Real.log (T : ℝ)) / T) := by gcongr
      _ = _ := by ring
  have hsq : x ^ 2 ≤ (12 * Real.sqrt (Real.log (T : ℝ)) / T) ^ 2 :=
    (sq_le_sq₀ hx0 (by positivity)).mpr hxu
  have heq : (12 * Real.sqrt (Real.log (T : ℝ)) / T) ^ 2 =
      144 * Real.log (T : ℝ) * invPower 2 T := by
    rw [div_pow, mul_pow, Real.sq_sqrt hlog0]
    unfold invPower
    ring
  rw [heq] at hsq
  have hvquad := mul_le_mul_of_nonneg_left hvl (show 0 ≤ 36 * Real.log (T : ℝ) by positivity)
  have hquad : x ^ 2 ≤ 36 * Real.log (T : ℝ) * (v : ℝ) := by
    nlinarith only [hsq, hvquad]
  have hquot : x ^ 2 / (2 * (v : ℝ)) ≤ 18 * Real.log (T : ℝ) := by
    apply (div_le_iff₀ (by positivity)).mpr
    nlinarith only [hquad]
  have hexp : Real.exp (-18 * Real.log (T : ℝ)) ≤
      Real.exp (-x ^ 2 / (2 * (v : ℝ))) := by
    apply Real.exp_le_exp.mpr
    rw [neg_div]
    linarith only [hquot]
  have hexpeq : Real.exp (-18 * Real.log (T : ℝ)) = invPower 18 T := by
    exact Draft0.PredictiveBound.exp_neg_nat_log 18 hTp
  rw [hexpeq] at hexp
  have hnorm : terminalConstant ≤ (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ := by
    unfold terminalConstant
    apply (inv_le_inv₀ (by positivity) (by positivity)).mpr
    apply Real.sqrt_le_sqrt
    nlinarith only [hvu, Real.pi_pos]
  unfold gaussianPDFReal
  simp only [sub_zero]
  exact mul_le_mul hnorm hexp (invPower_nonneg _ _) (by positivity)

theorem terminal_interval_lower22 (T : ℕ) (gap : ℝ) (v : ℝ≥0)
    (hT : 5 ≤ T) (hgap : |gap| ≤ 10 * Real.sqrt (Real.log (T : ℝ)) / T)
    (hv : 0 < v) (hvu : (v : ℝ) ≤ 2) (hvl : 4 * invPower 2 T ≤ (v : ℝ)) :
    terminalConstant * invPower 20 T ≤
      (gaussianReal 0 v (Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T))).toReal := by
  have hi := invPower_nonneg 2 T
  have hmass : (gaussianReal 0 v
      (Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T))).toReal =
      ∫ x in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T), gaussianPDFReal 0 v x := by
    rw [gaussianReal_apply_eq_integral 0 (ne_of_gt hv), ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg _ _)
  have hmono : (∫ _ in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T),
      terminalConstant * invPower 18 T) ≤
      ∫ x in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T), gaussianPDFReal 0 v x := by
    apply setIntegral_mono_on (integrableOn_const isCompact_Icc.measure_ne_top)
      (integrable_gaussianPDFReal 0 v).integrableOn measurableSet_Icc
    intro x hx
    exact terminal_gaussianPDF_lower22 T gap x v hT hgap hv hvu hvl hx
  have hconst : (∫ _ in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T),
      terminalConstant * invPower 18 T) = terminalConstant * invPower 20 T := by
    rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith only [hi])]
    rw [← invPower_product 18 2 T]
    ring
  rw [← hconst, hmass]
  exact hmono

/-- Revised Lemma 15's separate terminal Omega(T^-22) conclusion, including
zero gap, with an explicit uniform positive constant. -/
theorem lemma15_terminal_uniform22 (T q n : ℕ) (gap : ℝ) (vR : ℝ≥0)
    (hT : 5 ≤ T) (hodd : T = 2 * q + 1) (hn : 1 ≤ n) (hnq : n ≤ q)
    (hgap : |gap| ≤ 10 * Real.sqrt (Real.log (T : ℝ)) / T) :
    terminalConstant * invPower 22 T ≤
      oneStepImprovement gap (updateVariance (n : ℝ≥0)) vR := by
  have hnT : n < T := by omega
  have hb := updateVariance_count_bounds n T hn hnT
  have hp : 0 < updateVariance (n : ℝ≥0) := by
    have hnp : 0 < (n : ℝ≥0) := by exact_mod_cast (show 0 < n by omega)
    unfold updateVariance
    positivity
  have hvl := updateVariance_half_budget_lower T q n hodd hn hnq
  have hm := terminal_interval_lower22 T gap (updateVariance (n : ℝ≥0)) hT hgap hp
    (by linarith only [hb.2]) hvl
  have hscale := mul_le_mul_of_nonneg_left hm (invPower_nonneg 2 T)
  have hl := oneStepImprovement_ge_corrected_bound gap (invPower 2 T)
    (updateVariance (n : ℝ≥0)) vR
  unfold correctedImprovementLowerBound at hl
  calc
    terminalConstant * invPower 22 T = invPower 2 T * (terminalConstant * invPower 20 T) := by
      rw [← invPower_product 2 20 T]
      ring
    _ ≤ invPower 2 T *
        (gaussianReal 0 (updateVariance (n : ℝ≥0))
          (Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T))).toReal := hscale
    _ ≤ _ := hl

end

end Junpei.Marginal
