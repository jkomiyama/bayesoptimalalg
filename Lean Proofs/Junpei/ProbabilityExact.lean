import Junpei.ProbabilityPolynomial
import Junpei.TailRatio

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal

namespace Junpei.Probability

open Draft0 BadEvent TailRatio

noncomputable section

/-- The exact f1 printed in revised Lemma 3. -/
def f1 (T : ℕ) (Δ CU : ℝ) : ℝ :=
  (T : ℝ) ^ (-((CU + Δ) ^ 2) / 2) /
    (2 * (CU + Δ) * Real.sqrt (2 * Real.pi * Real.log (T : ℝ)))

theorem lemma3 (T : ℕ) (Δ CU : ℝ) (hT : 5 ≤ T) (hΔ : 0 ≤ Δ) (hCU : 0 < CU)
    (hz : Real.sqrt 2 ≤ CU * Real.sqrt (Real.log (T : ℝ)) + Δ) :
    f1 T Δ CU ≤ (gaussianReal Δ 1).real (underestimationSet T CU) := by
  let z : ℝ := CU * Real.sqrt (Real.log (T : ℝ)) + Δ
  let rateBase : ℝ := CU + Δ
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  have hlog := Drift.log_horizon_ge_one T hT
  have hroot : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have hzpos : 0 < z := (Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2)).trans_le hz
  have hrateBase : 0 < rateBase := by dsimp [rateBase]; linarith
  have hzsq : 2 ≤ z ^ 2 := by
    have hs := (sq_le_sq₀ (Real.sqrt_nonneg 2) hzpos.le).mpr hz
    simpa only [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)] using hs
  have hfactor : 1 / (2 * z) ≤ 1 / z - 1 / z ^ 3 := by
    field_simp [ne_of_gt hzpos]
    nlinarith only [hzsq, hzpos]
  have hM := (mul_le_mul_of_nonneg_right hfactor (phi_nonnegative z)).trans (mills_lower z hzpos)
  have hΔscale := mul_le_mul_of_nonneg_left hroot hΔ
  have hzupper : z ≤ rateBase * Real.sqrt (Real.log (T : ℝ)) := by
    dsimp [z, rateBase]
    nlinarith only [hΔscale]
  have hsq : z ^ 2 ≤ rateBase ^ 2 * Real.log (T : ℝ) := by
    have h := (sq_le_sq₀ hzpos.le (by positivity)).mpr hzupper
    simpa only [mul_pow, Real.sq_sqrt (by linarith : 0 ≤ Real.log (T : ℝ))] using h
  have he : Real.exp (-(rateBase ^ 2) * Real.log (T : ℝ) / 2) ≤ Real.exp (-z ^ 2 / 2) := by
    apply Real.exp_le_exp.mpr
    linarith only [hsq]
  have hpower : (T : ℝ) ^ (-(rateBase ^ 2) / 2) =
      Real.exp (-(rateBase ^ 2) * Real.log (T : ℝ) / 2) := by
    rw [Real.rpow_def_of_pos hTp]
    congr 1
    ring
  have hn0 : 0 ≤ normalization := by unfold normalization; positivity
  have hdensity : normalization * (T : ℝ) ^ (-(rateBase ^ 2) / 2) ≤ phi z := by
    rw [phi_eq, hpower]
    exact mul_le_mul_of_nonneg_left he hn0
  have hdenom : 1 / (2 * rateBase * Real.sqrt (Real.log (T : ℝ))) ≤ 1 / (2 * z) := by
    apply one_div_le_one_div_of_le (by positivity)
    nlinarith only [hzupper]
  have hm := mul_le_mul hdenom hdensity
    (mul_nonneg hn0 (Real.rpow_nonneg hTp.le _)) (by positivity)
  have heq : f1 T Δ CU = (1 / (2 * rateBase * Real.sqrt (Real.log (T : ℝ)))) *
      (normalization * (T : ℝ) ^ (-(rateBase ^ 2) / 2)) := by
    unfold f1 normalization
    dsimp [rateBase]
    rw [Real.sqrt_mul (by positivity : 0 ≤ 2 * Real.pi)]
    field_simp
  rw [heq]
  have htail : (gaussianReal Δ 1).real (underestimationSet T CU) = normalTail z := by
    simpa [underestimationSet, z, neg_mul] using
      shifted_left_tail Δ (CU * Real.sqrt (Real.log (T : ℝ)))
  rw [htail]
  exact hm.trans hM

theorem lemma3_eventually (Δ CU : ℝ) (hΔ : 0 < Δ) (hCU : 0 < CU) :
    ∀ᶠ T : ℕ in atTop,
      f1 T Δ CU ≤ (gaussianReal Δ 1).real (underestimationSet T CU) := by
  have hl : Tendsto (fun T : ℕ => Real.log (T : ℝ)) atTop atTop :=
    Real.tendsto_log_atTop.comp tendsto_natCast_atTop_atTop
  have he := hl.eventually_ge_atTop ((Real.sqrt 2 / CU) ^ 2)
  filter_upwards [eventually_ge_atTop (5 : ℕ), he] with T hT hlog
  apply lemma3 T Δ CU hT hΔ.le hCU
  have hs := Real.le_sqrt_of_sq_le hlog
  have hm := (div_le_iff₀ hCU).mp hs
  linarith

theorem rawBadEvent_exact_factor_lower (T q : ℕ) (Δ CU : ℝ)
    (hT : 8 ≤ T) (horizon : T = 2 * q + 1) (hΔ : 0 ≤ Δ) (hCU : 0 < CU)
    (hz : Real.sqrt 2 ≤ CU * Real.sqrt (Real.log (T : ℝ)) + Δ) :
    f1 T Δ CU * (SmallBall.f2 (T : ℝ) - f3 T) ^ 2 ≤
      (rawArraysLaw T).real (rawBadEvent T q Δ CU (by omega) (by omega)) := by
  rw [rawBadEvent_probability_eq]
  have hX := lemma3 T Δ CU (by omega) hΔ hCU hz
  have hY := goodArmSet3_probability_lower T q horizon (by omega)
  have hY0 := f2_minus_f3_nonnegative T hT
  have hYs := (sq_le_sq₀ hY0 measureReal_nonneg).mpr hY
  exact mul_le_mul hX hYs (sq_nonneg _) measureReal_nonneg

theorem rawBadEvent_exact_factor_eventually (Δ CU : ℝ) (hΔ : 0 < Δ) (hCU : 0 < CU) :
    ∀ᶠ T : ℕ in atTop, ∀ q : ℕ, ∀ (horizon : T = 2 * q + 1),
      f1 T Δ CU * (SmallBall.f2 (T : ℝ) - f3 T) ^ 2 ≤
        (rawArraysLaw T).real
          (rawBadEvent T q Δ CU (by omega) (by omega)) := by
  filter_upwards [eventually_ge_atTop (8 : ℕ), lemma3_eventually Δ CU hΔ hCU]
    with T hT hX
  intro q horizon
  rw [rawBadEvent_probability_eq]
  have hY := goodArmSet3_probability_lower T q horizon (by omega)
  have hY0 := f2_minus_f3_nonnegative T hT
  have hYs := (sq_le_sq₀ hY0 measureReal_nonneg).mpr hY
  exact mul_le_mul hX hYs (sq_nonneg _) measureReal_nonneg

end

end Junpei.Probability
