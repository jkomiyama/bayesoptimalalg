import Junpei.ProbabilityExact

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal

namespace Junpei.Probability

open Draft0 BadEvent

noncomputable section

/-- A positive constant for the exact real exponent printed in the paper. -/
def exactRateConstant (Δ CU : ℝ) : ℝ :=
  1 / (32 * (CU + Δ) * Real.pi * Real.sqrt (2 * Real.pi))

theorem exactRateConstant_positive (Δ CU : ℝ) (hΔ : 0 < Δ) (hCU : 0 < CU) :
    0 < exactRateConstant Δ CU := by
  unfold exactRateConstant
  positivity

theorem f1_positive (T : ℕ) (Δ CU : ℝ) (hT : 5 ≤ T)
    (hΔ : 0 < Δ) (hCU : 0 < CU) : 0 < f1 T Δ CU := by
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hlog := Drift.log_horizon_ge_one T hT
  unfold f1
  positivity

theorem f2_square (t : ℝ) (ht : 0 < t) :
    SmallBall.f2 t ^ 2 = 1 / (4 * Real.pi * t ^ 3) := by
  unfold SmallBall.f2
  rw [mul_pow, Real.sq_sqrt (by positivity)]
  ring

theorem f3_le_half_f2 (T : ℕ) (hT : 8 ≤ T) :
    f3 T ≤ SmallBall.f2 (T : ℝ) / 2 := by
  have hTr : (8 : ℝ) ≤ T := by exact_mod_cast hT
  have hTp : 0 < (T : ℝ) := by linarith
  have hlog : 0 ≤ Real.log (T : ℝ) := Real.log_nonneg (by linarith)
  have hsmall : f3 T ≤ 2 / (T : ℝ) ^ 3 := by
    have heq : 2 / (T : ℝ) ^ 3 = 2 * Real.exp (-(3 : ℝ) * Real.log (T : ℝ)) := by
      have he := PredictiveBound.exp_neg_nat_log 3 hTp
      norm_num at he
      have he' : Real.exp (-(3 : ℝ) * Real.log (T : ℝ)) = ((T : ℝ) ^ 3)⁻¹ := by
        calc
          _ = Real.exp (-((3 : ℝ) * Real.log (T : ℝ))) := by congr 1; ring
          _ = ((T : ℝ) ^ 3)⁻¹ := he
      rw [he']
      ring
    rw [heq]
    unfold f3
    apply mul_le_mul_of_nonneg_left _ (by norm_num)
    apply Real.exp_le_exp.mpr
    norm_num
    nlinarith
  have ht3 : (512 : ℝ) ≤ (T : ℝ) ^ 3 := by
    calc
      (512 : ℝ) = (8 : ℝ) ^ 3 := by norm_num
      _ ≤ (T : ℝ) ^ 3 := by gcongr
  have hsq : (2 / (T : ℝ) ^ 3) ^ 2 ≤ (SmallBall.f2 (T : ℝ) / 2) ^ 2 := by
    simp only [div_pow, f2_square _ hTp]
    have heq : (1 / (4 * Real.pi * (T : ℝ) ^ 3)) / (2 : ℝ) ^ 2 =
        1 / (16 * Real.pi * (T : ℝ) ^ 3) := by ring
    rw [heq]
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    have hpi : 64 * Real.pi ≤ (T : ℝ) ^ 3 := by linarith [Real.pi_le_four]
    have hm := mul_le_mul_of_nonneg_right hpi (show 0 ≤ (T : ℝ) ^ 3 by positivity)
    nlinarith only [hm]
  exact hsmall.trans ((sq_le_sq₀ (by positivity) (by unfold SmallBall.f2; positivity)).mp hsq)

theorem f2_minus_f3_square_lower (T : ℕ) (hT : 8 ≤ T) :
    1 / (16 * Real.pi * (T : ℝ) ^ 3) ≤
      (SmallBall.f2 (T : ℝ) - f3 T) ^ 2 := by
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hhalf := f3_le_half_f2 T hT
  have hf2 : 0 ≤ SmallBall.f2 (T : ℝ) := by unfold SmallBall.f2; positivity
  have hgap : SmallBall.f2 (T : ℝ) / 2 ≤ SmallBall.f2 (T : ℝ) - f3 T := by linarith
  have hs := (sq_le_sq₀ (by unfold SmallBall.f2; positivity) (by linarith)).mpr hgap
  rw [div_pow, f2_square _ hTp] at hs
  norm_num at hs
  convert hs using 1
  ring

theorem f1_polynomial_lower (T : ℕ) (Δ CU : ℝ) (hT : 5 ≤ T)
    (hΔ : 0 < Δ) (hCU : 0 < CU) :
    (T : ℝ) ^ (-((CU + Δ) ^ 2) / 2) /
      (2 * (CU + Δ) * Real.sqrt (2 * Real.pi) * (T : ℝ)) ≤ f1 T Δ CU := by
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hTr : (5 : ℝ) ≤ T := by exact_mod_cast hT
  have hlog := Drift.log_horizon_ge_one T hT
  have hroot : Real.sqrt (Real.log (T : ℝ)) ≤ (T : ℝ) := by
    apply Real.sqrt_le_iff.mpr
    refine ⟨hTp.le, ?_⟩
    have hl := Real.log_le_sub_one_of_pos hTp
    nlinarith
  unfold f1
  apply div_le_div_of_nonneg_left (Real.rpow_nonneg hTp.le _)
    (by positivity)
  rw [Real.sqrt_mul (by positivity : 0 ≤ 2 * Real.pi)]
  nlinarith only [mul_le_mul_of_nonneg_left hroot
    (show 0 ≤ 2 * (CU + Δ) * Real.sqrt (2 * Real.pi) by positivity)]

theorem exact_factor_rate_lower (T : ℕ) (Δ CU : ℝ) (hT : 8 ≤ T)
    (hΔ : 0 < Δ) (hCU : 0 < CU) :
    exactRateConstant Δ CU * (T : ℝ) ^ (-((CU + Δ) ^ 2 / 2 + 4)) ≤
      f1 T Δ CU * (SmallBall.f2 (T : ℝ) - f3 T) ^ 2 := by
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hsum : 0 < CU + Δ := by linarith
  have hroot : 0 < Real.sqrt (2 * Real.pi) := by positivity
  have hf1 := f1_polynomial_lower T Δ CU (by omega) hΔ hCU
  have hgap := f2_minus_f3_square_lower T hT
  have heq : exactRateConstant Δ CU * (T : ℝ) ^ (-((CU + Δ) ^ 2 / 2 + 4)) =
      ((T : ℝ) ^ (-((CU + Δ) ^ 2) / 2) /
        (2 * (CU + Δ) * Real.sqrt (2 * Real.pi) * (T : ℝ))) *
        (1 / (16 * Real.pi * (T : ℝ) ^ 3)) := by
    have hpow4 : (T : ℝ) ^ (4 : ℝ) = (T : ℝ) ^ (4 : ℕ) := by
      simp
    rw [show -((CU + Δ) ^ 2 / 2 + 4) = -((CU + Δ) ^ 2) / 2 + -(4 : ℝ) by ring,
      Real.rpow_add hTp, Real.rpow_neg hTp.le, hpow4]
    unfold exactRateConstant
    field_simp [ne_of_gt hTp, ne_of_gt hsum, ne_of_gt hroot, Real.pi_ne_zero]
    ring
  rw [heq]
  exact mul_le_mul hf1 hgap (by positivity) (f1_positive T Δ CU (by omega) hΔ hCU).le

theorem exact_factor_rate_eventually (Δ CU : ℝ) (hΔ : 0 < Δ) (hCU : 0 < CU) :
    ∀ᶠ T : ℕ in atTop,
      exactRateConstant Δ CU * (T : ℝ) ^ (-((CU + Δ) ^ 2 / 2 + 4)) ≤
        f1 T Δ CU * (SmallBall.f2 (T : ℝ) - f3 T) ^ 2 := by
  filter_upwards [eventually_ge_atTop (8 : ℕ)] with T hT
  exact exact_factor_rate_lower T Δ CU hT hΔ hCU

theorem rawBadEvent_exact_rate_eventually (Δ CU : ℝ) (hΔ : 0 < Δ) (hCU : 0 < CU) :
    ∀ᶠ T : ℕ in atTop, ∀ q : ℕ, ∀ (horizon : T = 2 * q + 1),
      exactRateConstant Δ CU * (T : ℝ) ^ (-((CU + Δ) ^ 2 / 2 + 4)) ≤
        (rawArraysLaw T).real
          (rawBadEvent T q Δ CU (by omega) (by omega)) := by
  filter_upwards [exact_factor_rate_eventually Δ CU hΔ hCU,
    rawBadEvent_exact_factor_eventually Δ CU hΔ hCU] with T hfactor hraw
  intro q horizon
  exact hfactor.trans (hraw q horizon)

end

end Junpei.Probability
