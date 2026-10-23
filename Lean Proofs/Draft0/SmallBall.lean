import Draft0.Gaussian
import Mathlib.Tactic

/-!
Actual Gaussian small-ball bound replacing the flawed variance estimate in
Appendix B. The final probability retains the paper's printed f₂ constant.
-/

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.SmallBall

noncomputable section

def f2 (t : ℝ) : ℝ := (1 / 2) * Real.sqrt (1 / (Real.pi * t ^ 3))

theorem gaussianPDF_smallBall_lower {t x : ℝ} {v : ℝ≥0}
    (ht : 3 ≤ t) (hv : 0 < v) (hvlower : t⁻¹ ≤ (v : ℝ))
    (hvupper : (v : ℝ) ≤ 3 / t)
    (hx : x ∈ Icc (-(t ^ 2)⁻¹) ((t ^ 2)⁻¹)) :
    (3 / 4) * (Real.sqrt (6 * Real.pi / t))⁻¹ ≤ gaussianPDFReal 0 v x := by
  have htpos : 0 < t := by linarith
  have ht2pos : 0 < t ^ 2 := sq_pos_of_pos htpos
  have hvpos : 0 < (v : ℝ) := hv
  have hrnonneg : 0 ≤ (t ^ 2)⁻¹ := inv_nonneg.mpr ht2pos.le
  have hrhalf : (t ^ 2)⁻¹ ≤ 1 / 2 := by
    rw [← one_div (t ^ 2)]
    apply (div_le_iff₀ ht2pos).mpr
    nlinarith
  have hrv : (t ^ 2)⁻¹ ≤ (v : ℝ) := by
    apply le_trans _ hvlower
    apply (inv_le_inv₀ ht2pos htpos).mpr
    nlinarith
  have hxsq : x ^ 2 ≤ ((t ^ 2)⁻¹) ^ 2 := by
    have hxabs : |x| ≤ (t ^ 2)⁻¹ := abs_le.mpr hx
    simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg x) hrnonneg).mpr hxabs
  have hquad : x ^ 2 ≤ (v : ℝ) / 2 := by
    nlinarith
  have hquot : x ^ 2 / (2 * (v : ℝ)) ≤ 1 / 4 := by
    apply (div_le_iff₀ (by positivity : 0 < 2 * (v : ℝ))).mpr
    nlinarith
  have hexp : (3 / 4 : ℝ) ≤ Real.exp (-x ^ 2 / (2 * (v : ℝ))) := by
    have he := Real.add_one_le_exp (-x ^ 2 / (2 * (v : ℝ)))
    rw [neg_div] at he ⊢
    linarith
  have hspos : 0 < Real.sqrt (2 * Real.pi * (v : ℝ)) := by positivity
  have hsboundpos : 0 < Real.sqrt (6 * Real.pi / t) := by positivity
  have hnorm : (Real.sqrt (6 * Real.pi / t))⁻¹ ≤
      (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ := by
    apply (inv_le_inv₀ hsboundpos hspos).mpr
    apply Real.sqrt_le_sqrt
    calc
      2 * Real.pi * (v : ℝ) ≤ 2 * Real.pi * (3 / t) := by gcongr
      _ = 6 * Real.pi / t := by ring
  calc
    _ ≤ (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ *
        Real.exp (-x ^ 2 / (2 * (v : ℝ))) := by
      rw [mul_comm (3 / 4 : ℝ)]
      exact mul_le_mul hnorm hexp (by norm_num) (inv_nonneg.mpr hspos.le)
    _ = gaussianPDFReal 0 v x := by simp only [gaussianPDFReal, sub_zero]

theorem smallBall_constant_comparison {t : ℝ} (ht : 0 < t) :
    f2 t ≤ (3 / 2) * (t ^ 2)⁻¹ * (Real.sqrt (6 * Real.pi / t))⁻¹ := by
  let r := Real.sqrt (1 / (Real.pi * t ^ 3))
  let s := Real.sqrt (6 * Real.pi / t)
  have hr : 0 ≤ r := Real.sqrt_nonneg _
  have hs : 0 < s := by dsimp [s]; positivity
  have ht0 : t ≠ 0 := ne_of_gt ht
  have hp0 : Real.pi ≠ 0 := ne_of_gt Real.pi_pos
  have hr2 : r ^ 2 = 1 / (Real.pi * t ^ 3) := Real.sq_sqrt (by positivity)
  have hs2 : s ^ 2 = 6 * Real.pi / t := Real.sq_sqrt (by positivity)
  have hprod : (r * s * t ^ 2) ^ 2 = 6 := by
    rw [mul_pow, mul_pow, hr2, hs2]
    field_simp [ht0, hp0]
  have hprodle : r * s * t ^ 2 ≤ 3 := by nlinarith
  change (1 / 2) * r ≤ (3 / 2) * (t ^ 2)⁻¹ * s⁻¹
  apply (mul_le_mul_iff_left₀ (by positivity : 0 < t ^ 2 * s)).mp
  calc
    ((1 / 2) * r) * (t ^ 2 * s) = (1 / 2) * (r * s * t ^ 2) := by ring
    _ ≤ (1 / 2) * 3 := by gcongr
    _ = ((3 / 2) * (t ^ 2)⁻¹ * s⁻¹) * (t ^ 2 * s) := by
      field_simp [ht0, ne_of_gt hs]

theorem gaussian_smallBall_probability_lower {t : ℝ} {v : ℝ≥0}
    (ht : 3 ≤ t) (hv : 0 < v) (hvlower : t⁻¹ ≤ (v : ℝ))
    (hvupper : (v : ℝ) ≤ 3 / t) :
    f2 t ≤ (gaussianReal 0 v (Icc (-(t ^ 2)⁻¹) ((t ^ 2)⁻¹))).toReal := by
  have htpos : 0 < t := by linarith
  have hrnonneg : 0 ≤ (t ^ 2)⁻¹ := inv_nonneg.mpr (sq_nonneg _)
  have hmass : (gaussianReal 0 v (Icc (-(t ^ 2)⁻¹) ((t ^ 2)⁻¹))).toReal =
      ∫ x in Icc (-(t ^ 2)⁻¹) ((t ^ 2)⁻¹), gaussianPDFReal 0 v x := by
    rw [gaussianReal_apply_eq_integral 0 (ne_of_gt hv), ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg _ _)
  have hmono :
      (∫ _ in Icc (-(t ^ 2)⁻¹) ((t ^ 2)⁻¹),
        (3 / 4 : ℝ) * (Real.sqrt (6 * Real.pi / t))⁻¹) ≤
      ∫ x in Icc (-(t ^ 2)⁻¹) ((t ^ 2)⁻¹), gaussianPDFReal 0 v x := by
    apply setIntegral_mono_on (integrableOn_const isCompact_Icc.measure_ne_top)
      (integrable_gaussianPDFReal 0 v).integrableOn measurableSet_Icc
    intro x hx
    exact gaussianPDF_smallBall_lower ht hv hvlower hvupper hx
  have hconst :
      (∫ _ in Icc (-(t ^ 2)⁻¹) ((t ^ 2)⁻¹),
        (3 / 4 : ℝ) * (Real.sqrt (6 * Real.pi / t))⁻¹) =
      (3 / 2) * (t ^ 2)⁻¹ * (Real.sqrt (6 * Real.pi / t))⁻¹ := by
    rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith : 0 ≤ (t ^ 2)⁻¹ - -(t ^ 2)⁻¹)]
    ring
  rw [← hmass, hconst] at hmono
  exact (smallBall_constant_comparison htpos).trans hmono

/-- Corrected Appendix B/Lemma 4 for the actual variance 1/q, T=2q+1.
The paper's f₂ lower bound survives the corrected variance estimate. -/
theorem appendixB_probability_lower (q : ℕ) (hq : 0 < q) :
    f2 (2 * (q : ℝ) + 1) ≤
      (gaussianReal 0 (q : ℝ≥0)⁻¹
        (Icc (-((2 * (q : ℝ) + 1) ^ 2)⁻¹) (((2 * (q : ℝ) + 1) ^ 2)⁻¹))).toReal := by
  have hqR : (1 : ℝ) ≤ q := by exact_mod_cast (show 1 ≤ q by omega)
  have hqpos : 0 < (q : ℝ) := by linarith
  have htpos : 0 < 2 * (q : ℝ) + 1 := by linarith
  apply gaussian_smallBall_probability_lower (by linarith) (by positivity)
  · simp only [NNReal.coe_inv, NNReal.coe_natCast]
    apply (inv_le_inv₀ htpos hqpos).mpr
    linarith
  · simp only [NNReal.coe_inv, NNReal.coe_natCast]
    rw [← one_div (q : ℝ)]
    apply (div_le_iff₀ hqpos).mpr
    rw [div_mul_eq_mul_div]
    apply (le_div_iff₀ htpos).mpr
    nlinarith

/-- Law of the sum of q independent unit-variance, zero-mean Gaussian
observations, with convolution expressing independence. -/
def standardNormalSum : ℕ → Measure ℝ
  | 0 => gaussianReal 0 0
  | q + 1 => standardNormalSum q ∗ gaussianReal 0 1

theorem standardNormalSum_law (q : ℕ) :
    standardNormalSum q = gaussianReal 0 (q : ℝ≥0) := by
  induction q with
  | zero => simp only [standardNormalSum, Nat.cast_zero]
  | succ q ih =>
    rw [standardNormalSum, ih, gaussianReal_conv_gaussianReal]
    simp only [add_zero, Nat.cast_add, Nat.cast_one]

def standardNormalAverage (q : ℕ) : Measure ℝ :=
  (standardNormalSum q).map (fun x : ℝ => (q : ℝ)⁻¹ * x)

theorem standardNormalAverage_law (q : ℕ) (hq : 0 < q) :
    standardNormalAverage q = gaussianReal 0 (q : ℝ≥0)⁻¹ := by
  have hqpos : 0 < (q : ℝ) := by exact_mod_cast hq
  unfold standardNormalAverage
  rw [standardNormalSum_law, gaussianReal_map_const_mul, mul_zero]
  congr 1
  ext
  simp only [NNReal.coe_mul, NNReal.coe_mk, NNReal.coe_natCast, NNReal.coe_inv]
  field_simp [ne_of_gt hqpos]

/-- Appendix B's lower bound for the actual iid sample-average law. -/
theorem appendixB_sampleAverage_probability_lower (q : ℕ) (hq : 0 < q) :
    f2 (2 * (q : ℝ) + 1) ≤
      (standardNormalAverage q
        {x : ℝ | |x| ≤ ((2 * (q : ℝ) + 1) ^ 2)⁻¹}).toReal := by
  have hset : {x : ℝ | |x| ≤ ((2 * (q : ℝ) + 1) ^ 2)⁻¹} =
      Icc (-((2 * (q : ℝ) + 1) ^ 2)⁻¹) (((2 * (q : ℝ) + 1) ^ 2)⁻¹) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_Icc, abs_le]
  rw [standardNormalAverage_law q hq, hset]
  exact appendixB_probability_lower q hq

end

end Draft0.SmallBall
