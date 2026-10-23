import Draft0.Rates
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Tactic

/-!
A quantitative replacement for the nonterminal Gaussian probability estimate
in Lemma 17. All probabilities use the actual Gaussian measure. The bound on
the predictive mean is an explicit hypothesis; it is not a custom axiom.
The zero-variance terminal state is deliberately outside this theorem.
-/

open MeasureTheory ProbabilityTheory Set
open Filter Asymptotics
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.PredictiveBound

noncomputable section

def lowerConstant : ℝ := Real.exp (-4) * (Real.sqrt (4 * Real.pi))⁻¹

theorem lowerConstant_pos : 0 < lowerConstant := by
  unfold lowerConstant
  exact mul_pos (Real.exp_pos _) (inv_pos.mpr
    (Real.sqrt_pos.mpr (mul_pos (by norm_num) Real.pi_pos)))

theorem exp_neg_nat_log (D : ℕ) {t : ℝ} (ht : 0 < t) :
    Real.exp (-(D : ℝ) * Real.log t) = (t ^ D)⁻¹ := by
  rw [show -(D : ℝ) * Real.log t = -((D : ℝ) * Real.log t) by ring,
    Real.exp_neg, Real.exp_nat_mul, Real.exp_log ht]

theorem gaussianPDF_interval_lower (D : ℕ) {t δ x : ℝ} {v : ℝ≥0}
    (ht : 1 ≤ t) (hv : 0 < v) (hvupper : (v : ℝ) ≤ 2)
    (hvlower : (t ^ 2)⁻¹ ≤ (v : ℝ))
    (hδ : δ ^ 2 ≤ (D : ℝ) * Real.log t * (v : ℝ))
    (hx : x ∈ Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹)) :
    lowerConstant * (t ^ D)⁻¹ ≤ gaussianPDFReal δ v x := by
  have htpos : 0 < t := lt_of_lt_of_le zero_lt_one ht
  have hvpos : 0 < (v : ℝ) := hv
  have hqnonneg : 0 ≤ (t ^ 2)⁻¹ := inv_nonneg.mpr (sq_nonneg t)
  have hqle : (t ^ 2)⁻¹ ≤ 1 := by
    exact inv_le_one_of_one_le₀ (one_le_pow₀ ht)
  have hxnonneg : 0 ≤ x := hqnonneg.trans hx.1
  have hxx : x ^ 2 ≤ 4 * (v : ℝ) := by
    have hqq : ((t ^ 2)⁻¹) ^ 2 ≤ (t ^ 2)⁻¹ := by nlinarith
    have hsq : x ^ 2 ≤ (2 * (t ^ 2)⁻¹) ^ 2 :=
      (sq_le_sq₀ hxnonneg (by positivity)).mpr hx.2
    nlinarith
  have hquad : (x - δ) ^ 2 ≤
      ((D : ℝ) * Real.log t + 4) * (2 * (v : ℝ)) := by
    nlinarith [sq_nonneg (x + δ)]
  have hquot : (x - δ) ^ 2 / (2 * (v : ℝ)) ≤ (D : ℝ) * Real.log t + 4 :=
    (div_le_iff₀ (mul_pos (by norm_num) hvpos)).mpr hquad
  have hexp : Real.exp (-(D : ℝ) * Real.log t - 4) ≤
      Real.exp (-(x - δ) ^ 2 / (2 * (v : ℝ))) := by
    apply Real.exp_le_exp.mpr
    rw [neg_div]
    linarith
  have hsqrtpos : 0 < Real.sqrt (2 * Real.pi * (v : ℝ)) := by
    apply Real.sqrt_pos.mpr
    positivity
  have hsqrt4pos : 0 < Real.sqrt (4 * Real.pi) := by
    apply Real.sqrt_pos.mpr
    positivity
  have hnorm : (Real.sqrt (4 * Real.pi))⁻¹ ≤
      (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ := by
    apply (inv_le_inv₀ hsqrt4pos hsqrtpos).mpr
    apply Real.sqrt_le_sqrt
    nlinarith [Real.pi_pos]
  have hexpeq : Real.exp (-(D : ℝ) * Real.log t - 4) =
      Real.exp (-4) * (t ^ D)⁻¹ := by
    rw [sub_eq_add_neg, Real.exp_add, exp_neg_nat_log D htpos, mul_comm]
  calc
    lowerConstant * (t ^ D)⁻¹ =
        (Real.sqrt (4 * Real.pi))⁻¹ * Real.exp (-(D : ℝ) * Real.log t - 4) := by
      rw [hexpeq, lowerConstant]
      ring
    _ ≤ (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ *
        Real.exp (-(x - δ) ^ 2 / (2 * (v : ℝ))) :=
      mul_le_mul hnorm hexp (Real.exp_pos _).le (inv_nonneg.mpr hsqrtpos.le)
    _ = gaussianPDFReal δ v x := rfl

theorem gaussian_interval_probability_lower_real (D : ℕ) {t δ : ℝ} {v : ℝ≥0}
    (ht : 1 ≤ t) (hv : 0 < v) (hvupper : (v : ℝ) ≤ 2)
    (hvlower : (t ^ 2)⁻¹ ≤ (v : ℝ))
    (hδ : δ ^ 2 ≤ (D : ℝ) * Real.log t * (v : ℝ)) :
    lowerConstant * (t ^ (D + 2))⁻¹ ≤
      (gaussianReal δ v (Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹))).toReal := by
  have hqnonneg : 0 ≤ (t ^ 2)⁻¹ := inv_nonneg.mpr (sq_nonneg t)
  have hmass :
      (gaussianReal δ v (Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹))).toReal =
        ∫ x in Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹), gaussianPDFReal δ v x := by
    rw [gaussianReal_apply_eq_integral δ (ne_of_gt hv), ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg δ v)
  have hmono :
      (∫ _ in Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹), lowerConstant * (t ^ D)⁻¹) ≤
      ∫ x in Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹), gaussianPDFReal δ v x := by
    apply setIntegral_mono_on
      (integrableOn_const isCompact_Icc.measure_ne_top)
      (integrable_gaussianPDFReal δ v).integrableOn measurableSet_Icc
    intro x hx
    exact gaussianPDF_interval_lower D ht hv hvupper hvlower hδ hx
  have hconst :
      (∫ _ in Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹), lowerConstant * (t ^ D)⁻¹) =
      lowerConstant * (t ^ (D + 2))⁻¹ := by
    rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith : 0 ≤ 2 * (t ^ 2)⁻¹ - (t ^ 2)⁻¹)]
    rw [pow_add, mul_inv]
    ring
  rw [← hconst, hmass]
  exact hmono

/-- Corrected nonterminal small-gap probability on natural-number budgets. -/
theorem gaussian_interval_probability_lower_nat (D T : ℕ) {δ : ℝ} {v : ℝ≥0}
    (hT : 5 ≤ T) (hv : 0 < v) (hvupper : (v : ℝ) ≤ 2)
    (hvlower : Draft0.Rates.invPower 2 T ≤ (v : ℝ))
    (hδ : δ ^ 2 ≤ (D : ℝ) * Real.log (T : ℝ) * (v : ℝ)) :
    lowerConstant * Draft0.Rates.invPower (D + 2) T ≤
      (gaussianReal δ v
        (Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T))).toReal := by
  have hTreal : (1 : ℝ) ≤ T := by exact_mod_cast (show 1 ≤ T by omega)
  exact gaussian_interval_probability_lower_real D hTreal hv hvupper hvlower hδ

/-- The quantitative exponent used after bounding the predictive drift by 204. -/
theorem gaussian_interval_probability_drift204 (T : ℕ) {δ : ℝ} {v : ℝ≥0}
    (hT : 5 ≤ T) (hv : 0 < v) (hvupper : (v : ℝ) ≤ 2)
    (hvlower : Draft0.Rates.invPower 2 T ≤ (v : ℝ))
    (hδ : δ ^ 2 ≤ 204 * Real.log (T : ℝ) * (v : ℝ)) :
    lowerConstant * Draft0.Rates.invPower 206 T ≤
      (gaussianReal δ v
        (Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T))).toReal := by
  exact gaussian_interval_probability_lower_nat 204 T hT hv hvupper hvlower hδ

/-- The same probability bound in the paper's integer-budget Omega notation. -/
theorem gaussian_interval_probability_omega (D : ℕ)
    {δ : ℕ → ℝ} {v : ℕ → ℝ≥0}
    (hv : ∀ᶠ T in atTop, 0 < v T)
    (hvupper : ∀ᶠ T in atTop, (v T : ℝ) ≤ 2)
    (hvlower : ∀ᶠ T in atTop, Draft0.Rates.invPower 2 T ≤ (v T : ℝ))
    (hδ : ∀ᶠ T in atTop,
      (δ T) ^ 2 ≤ (D : ℝ) * Real.log (T : ℝ) * (v T : ℝ)) :
    Draft0.Rates.invPower (D + 2) =O[atTop]
      (fun T => (gaussianReal (δ T) (v T)
        (Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T))).toReal) := by
  apply Draft0.Rates.omega_of_eventual_lower_bound lowerConstant_pos
  filter_upwards [eventually_ge_atTop (5 : ℕ), hv, hvupper, hvlower, hδ]
    with T hT hvT hupperT hlowerT hδT
  exact gaussian_interval_probability_lower_nat D T hT hvT hupperT hlowerT hδT

end

end Draft0.PredictiveBound
