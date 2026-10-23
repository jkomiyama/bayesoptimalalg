import Junpei.Marginal
import Draft0.PredictiveBound

open MeasureTheory ProbabilityTheory Set Filter Asymptotics
open scoped MeasureTheory NNReal ENNReal

namespace Junpei.Marginal

noncomputable section

open Draft0.Gaussian Draft0.Rates Draft0.TerminalBounds
open JunpeiComparison.Gaussian

def intervalConstant : ℝ := Real.exp (-8) * (Real.sqrt (4 * Real.pi))⁻¹
def tailConstant : ℝ := (gaussianReal 0 1 (Icc (1 : ℝ) 2)).toReal
def g1Constant : ℝ := intervalConstant * tailConstant / 2
def g2Constant : ℝ := 5 * intervalConstant ^ 2 * tailConstant / 432

theorem intervalConstant_pos : 0 < intervalConstant := by
  unfold intervalConstant
  positivity

theorem tailConstant_pos : 0 < tailConstant :=
  gaussian_Icc_probability_pos 0 1 2 1 (by norm_num) (by norm_num)

theorem g1Constant_pos : 0 < g1Constant := by
  unfold g1Constant
  exact div_pos (mul_pos intervalConstant_pos tailConstant_pos) (by norm_num)

theorem g2Constant_pos : 0 < g2Constant := by
  unfold g2Constant
  exact div_pos (mul_pos (mul_pos (by norm_num) (pow_pos intervalConstant_pos 2))
    tailConstant_pos) (by norm_num)

/-- A uniform positive lower tail probability whenever the threshold is at
most one standard deviation. The constant is an actual standard Gaussian mass. -/
theorem gaussian_tail_lower (a : ℝ) (v : ℝ≥0) (ha : a ≤ Real.sqrt (v : ℝ)) :
    tailConstant ≤ (gaussianReal 0 v (Ici a)).toReal := by
  let s : ℝ := Real.sqrt (v : ℝ)
  have hs : 0 ≤ s := Real.sqrt_nonneg _
  have hmap : (gaussianReal 0 1).map (fun z : ℝ => s * z) = gaussianReal 0 v := by
    have hv : (⟨s ^ 2, sq_nonneg s⟩ : ℝ≥0) * 1 = v := by
      ext
      simp [s]
    rw [gaussianReal_map_const_mul, mul_zero, hv]
  have hsub : Icc (1 : ℝ) 2 ⊆ (fun z : ℝ => s * z) ⁻¹' Ici a := by
    intro z hz
    change a ≤ s * z
    exact ha.trans (by dsimp [s] at hs ⊢; nlinarith only [hs, hz.1])
  unfold tailConstant
  rw [← hmap, Measure.map_apply (by fun_prop) measurableSet_Ici]
  exact ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono hsub)

/-- Small interval density bound uniform down to variance T^-3. -/
theorem gaussianPDF_small_interval_lower (T : ℕ) (x : ℝ) (v : ℝ≥0)
    (hT : 5 ≤ T) (hv : 0 < v) (hvu : (v : ℝ) ≤ 2)
    (hvl : invPower 3 T ≤ (v : ℝ))
    (hx0 : 0 ≤ x) (hxu : x ≤ 4 * invPower 2 T) :
    intervalConstant ≤ gaussianPDFReal 0 v x := by
  have hTr : 1 ≤ (T : ℝ) := by exact_mod_cast (show 1 ≤ T by omega)
  have hTp : 0 < (T : ℝ) := by linarith
  have hp2 : 0 < (T : ℝ) ^ 2 := by positivity
  have hp3 : 0 < (T : ℝ) ^ 3 := by positivity
  have hi2 : 0 ≤ invPower 2 T := invPower_nonneg _ _
  have hpow : (invPower 2 T) ^ 2 ≤ invPower 3 T := by
    unfold invPower
    rw [inv_pow]
    apply (inv_le_inv₀ (by positivity) (by positivity)).mpr
    calc
      (T : ℝ) ^ 3 ≤ (T : ℝ) ^ 4 := pow_le_pow_right₀ hTr (by norm_num)
      _ = ((T : ℝ) ^ 2) ^ 2 := by ring
  have hsq : x ^ 2 ≤ (4 * invPower 2 T) ^ 2 :=
    (sq_le_sq₀ hx0 (by positivity)).mpr hxu
  have hquad : x ^ 2 ≤ 16 * (v : ℝ) := by nlinarith only [hsq, hpow, hvl]
  have hquot : x ^ 2 / (2 * (v : ℝ)) ≤ 8 := by
    apply (div_le_iff₀ (by positivity)).mpr
    linarith only [hquad]
  have hexp : Real.exp (-8) ≤ Real.exp (-x ^ 2 / (2 * (v : ℝ))) := by
    apply Real.exp_le_exp.mpr
    rw [neg_div]
    linarith only [hquot]
  have hnorm : (Real.sqrt (4 * Real.pi))⁻¹ ≤
      (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ := by
    apply (inv_le_inv₀ (by positivity) (by positivity)).mpr
    apply Real.sqrt_le_sqrt
    nlinarith only [hvu, Real.pi_pos]
  unfold intervalConstant
  rw [mul_comm, gaussianPDFReal]
  simp only [sub_zero]
  exact mul_le_mul hnorm hexp (Real.exp_pos _).le (by positivity)

theorem gaussian_small_interval_lower (T : ℕ) (a b : ℝ) (v : ℝ≥0)
    (hT : 5 ≤ T) (hv : 0 < v) (hvu : (v : ℝ) ≤ 2)
    (hvl : invPower 3 T ≤ (v : ℝ))
    (ha : 0 ≤ a) (hab : a ≤ b) (hb : b ≤ 4 * invPower 2 T) :
    intervalConstant * (b - a) ≤ (gaussianReal 0 v (Icc a b)).toReal := by
  have hmass : (gaussianReal 0 v (Icc a b)).toReal =
      ∫ x in Icc a b, gaussianPDFReal 0 v x := by
    rw [gaussianReal_apply_eq_integral 0 (ne_of_gt hv), ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg _ _)
  have hmono : (∫ _ in Icc a b, intervalConstant) ≤
      ∫ x in Icc a b, gaussianPDFReal 0 v x := by
    apply setIntegral_mono_on (integrableOn_const isCompact_Icc.measure_ne_top)
      (integrable_gaussianPDFReal 0 v).integrableOn measurableSet_Icc
    intro x hx
    exact gaussianPDF_small_interval_lower T x v hT hv hvu hvl (ha.trans hx.1)
      (hx.2.trans hb)
  have hconst : (∫ _ in Icc a b, intervalConstant) = intervalConstant * (b - a) := by
    rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Icc,
      ENNReal.toReal_ofReal (sub_nonneg.mpr hab)]
    ring
  rw [← hconst, hmass]
  exact hmono

theorem tiny_gap_le_sqrt (T : ℕ) (gap : ℝ) (v : ℝ≥0) (hT : 5 ≤ T)
    (hgap : |gap| ≤ 2 * invPower 2 T) (hv : invPower 2 T ≤ (v : ℝ)) :
    |gap| ≤ Real.sqrt (v : ℝ) := by
  have hi : 0 ≤ invPower 2 T := invPower_nonneg _ _
  have hi4 : invPower 2 T ≤ 1 / 4 := by
    have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT
    unfold invPower
    rw [inv_eq_one_div]
    apply (div_le_iff₀ (by positivity)).mpr
    nlinarith only [hTr]
  apply Real.le_sqrt_of_sq_le
  have hsq := (sq_le_sq₀ (abs_nonneg gap) (by positivity)).mpr hgap
  nlinarith only [hsq, hi, hi4, hv]

theorem g1_uniform4_variance (T : ℕ) (gap : ℝ) (vL vR : ℝ≥0)
    (hT : 5 ≤ T) (hgapL : invPower 2 T ≤ |gap|)
    (hgapU : |gap| ≤ 2 * invPower 2 T)
    (hl : 0 < vL) (hlu : (vL : ℝ) ≤ 2) (hll : invPower 3 T ≤ (vL : ℝ))
    (hrl : invPower 2 T ≤ (vR : ℝ)) :
    g1Constant * invPower 4 T ≤ newG1 gap vL vR := by
  have hi := invPower_nonneg 2 T
  have hmass := gaussian_small_interval_lower T (3 * |gap| / 2) (2 * |gap|) vL
    hT hl hlu hll (by positivity) (by linarith [abs_nonneg gap]) (by linarith)
  have htail := gaussian_tail_lower |gap| vR (tiny_gap_le_sqrt T gap vR hT hgapU hrl)
  have hp : 0 ≤ (gaussianReal 0 vL (Icc (3 * |gap| / 2) (2 * |gap|))).toReal := by positivity
  have hq : 0 ≤ (gaussianReal 0 vR (Ici |gap|)).toReal := by positivity
  have hmass' : (intervalConstant / 2) * invPower 2 T ≤
      (gaussianReal 0 vL (Icc (3 * |gap| / 2) (2 * |gap|))).toReal := by
    have hmul := mul_le_mul_of_nonneg_left hgapL intervalConstant_pos.le
    nlinarith only [hmass, hmul]
  have hb := mul_le_mul hmass' htail tailConstant_pos.le hp
  have hb' := mul_le_mul hb hgapL hi (mul_nonneg hp hq)
  unfold newG1 g1Constant
  rw [← invPower_product 2 2 T]
  nlinarith only [hb']

theorem g2_uniform6_variance (T : ℕ) (gap : ℝ) (vL vG vR : ℝ≥0)
    (hT : 5 ≤ T) (hgapL : invPower 2 T ≤ |gap|)
    (hgapU : |gap| ≤ 2 * invPower 2 T)
    (hl : 0 < vL) (hlu : (vL : ℝ) ≤ 2) (hll : invPower 3 T ≤ (vL : ℝ))
    (hg : 0 < vG) (hgu : (vG : ℝ) ≤ 2) (hgl : invPower 3 T ≤ (vG : ℝ))
    (hrl : invPower 2 T ≤ (vR : ℝ)) :
    g2Constant * invPower 6 T ≤ newG2 gap vL vG vR := by
  have hi := invPower_nonneg 2 T
  have hlmass := gaussian_small_interval_lower T (|gap| / 4) (|gap| / 3) vL
    hT hl hlu hll (by positivity) (by linarith [abs_nonneg gap]) (by linarith)
  have hgmass := gaussian_small_interval_lower T (9 * |gap| / 8) (4 * |gap| / 3) vG
    hT hg hgu hgl (by positivity) (by linarith [abs_nonneg gap]) (by linarith)
  have htail := gaussian_tail_lower (3 * |gap| / 4) vR
    (by linarith [tiny_gap_le_sqrt T gap vR hT hgapU hrl, abs_nonneg gap])
  have hlmass' : (intervalConstant / 12) * invPower 2 T ≤
      (gaussianReal 0 vL (Icc (|gap| / 4) (|gap| / 3))).toReal := by
    have hmul := mul_le_mul_of_nonneg_left hgapL intervalConstant_pos.le
    nlinarith only [hlmass, hmul]
  have hgmass' : (5 * intervalConstant / 24) * invPower 2 T ≤
      (gaussianReal 0 vG (Icc (9 * |gap| / 8) (4 * |gap| / 3))).toReal := by
    have hmul := mul_le_mul_of_nonneg_left hgapL intervalConstant_pos.le
    nlinarith only [hgmass, hmul]
  have hp : 0 ≤ (gaussianReal 0 vL (Icc (|gap| / 4) (|gap| / 3))).toReal := by positivity
  have hq : 0 ≤ (gaussianReal 0 vG (Icc (9 * |gap| / 8) (4 * |gap| / 3))).toReal := by positivity
  have hr : 0 ≤ (gaussianReal 0 vR (Ici (3 * |gap| / 4))).toReal := by positivity
  have hcpos := intervalConstant_pos
  have hb := mul_le_mul hlmass' hgmass' (by positivity) hp
  have hb' := mul_le_mul hb htail tailConstant_pos.le (mul_nonneg hp hq)
  have hfactor : (2 / 3) * invPower 2 T ≤ 2 * |gap| / 3 := by linarith only [hgapL]
  have hb'' := mul_le_mul hb' hfactor (by positivity) (mul_nonneg (mul_nonneg hp hq) hr)
  unfold newG2 g2Constant
  rw [← invPower_product 4 2 T, ← invPower_product 2 2 T]
  nlinarith only [hb'']

end

end Junpei.Marginal
