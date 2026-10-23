import Draft0.Gaussian
import Mathlib.MeasureTheory.Integral.IntegralEqImproper

open MeasureTheory ProbabilityTheory Set Filter
open scoped MeasureTheory NNReal ENNReal Topology

namespace Junpei.TailRatio

noncomputable section

def normalTail (x : ℝ) : ℝ := (gaussianReal 0 1 (Ioi x)).toReal
def phi (x : ℝ) : ℝ := gaussianPDFReal 0 1 x
def normalization : ℝ := (Real.sqrt (2 * Real.pi))⁻¹

theorem phi_eq (x : ℝ) : phi x = normalization * Real.exp (-x ^ 2 / 2) := by
  simp [phi, normalization, gaussianPDFReal]

theorem phi_nonnegative (x : ℝ) : 0 ≤ phi x := gaussianPDFReal_nonneg 0 1 x

theorem normalTail_eq_integral (x : ℝ) : normalTail x = ∫ y in Ioi x, phi y := by
  unfold normalTail phi
  rw [gaussianReal_apply_eq_integral 0 (by norm_num : (1 : ℝ≥0) ≠ 0),
    ENNReal.toReal_ofReal]
  exact integral_nonneg (gaussianPDFReal_nonneg _ _)

/-- The underestimation probability in revised Lemma 3 is the corresponding
standard upper tail. Both boundary conventions agree because Gaussians have no atoms. -/
theorem shifted_left_tail (delta a : ℝ) :
    (gaussianReal delta 1).real (Iic (-a)) = normalTail (a + delta) := by
  letI : NoAtoms (gaussianReal 0 1) := noAtoms_gaussianReal (by norm_num : (1 : ℝ≥0) ≠ 0)
  have hmap : (gaussianReal 0 1).map (fun y : ℝ => delta - y) = gaussianReal delta 1 := by
    simpa only [sub_zero] using gaussianReal_map_const_sub (μ := (0 : ℝ)) (v := 1) delta
  have hpre : (fun y : ℝ => delta - y) ⁻¹' Iic (-a) = Ici (a + delta) := by
    ext y
    change delta - y ≤ -a ↔ a + delta ≤ y
    constructor <;> intro h <;> linarith only [h]
  rw [measureReal_def, ← hmap, Measure.map_apply (by fun_prop) measurableSet_Iic, hpre]
  unfold normalTail
  exact congrArg ENNReal.toReal
    (measure_congr (Ioi_ae_eq_Ici (μ := gaussianReal 0 1) (a := a + delta))).symm

theorem phi_integrable : Integrable phi := integrable_gaussianPDFReal 0 1

theorem hasDerivAt_phi (x : ℝ) : HasDerivAt phi (-x * phi x) x := by
  have he : HasDerivAt (fun y : ℝ => Real.exp (-y ^ 2 / 2))
      (Real.exp (-x ^ 2 / 2) * (-x)) x := by
    convert ((((hasDerivAt_id x).pow 2).neg.div_const 2).exp) using 1
    simp only [id_eq, Pi.pow_apply, Pi.neg_apply]
    ring
  have hf : phi = fun y : ℝ => normalization * Real.exp (-y ^ 2 / 2) := by
    funext y
    exact phi_eq y
  rw [hf]
  convert he.const_mul normalization using 1
  ring

theorem phi_tendsto_zero : Tendsto phi atTop (𝓝 0) := by
  have hsq : Tendsto (fun y : ℝ => y * y) atTop atTop :=
    Tendsto.atTop_mul_atTop₀ tendsto_id tendsto_id
  have hscale : Tendsto (fun y : ℝ => y ^ 2 / 2) atTop atTop := by
    convert (tendsto_const_mul_atTop_of_pos (by norm_num : (0 : ℝ) < 1 / 2)).mpr hsq using 1
    funext y
    ring
  have he := Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp hscale)
  have hf : phi = fun y : ℝ => normalization * Real.exp (-y ^ 2 / 2) := by
    funext y
    exact phi_eq y
  rw [hf]
  simpa only [neg_div, mul_zero] using he.const_mul normalization

def primitive (x : ℝ) : ℝ := -phi x / x
def derivative (x : ℝ) : ℝ := phi x + phi x / x ^ 2
def remainder (x : ℝ) : ℝ := phi x / x ^ 2

theorem primitive_tendsto_zero : Tendsto primitive atTop (𝓝 0) := by
  have h := phi_tendsto_zero.neg.mul tendsto_inv_atTop_zero
  change Tendsto (fun x : ℝ => -phi x / x) atTop (𝓝 0)
  simpa [primitive, div_eq_mul_inv] using h

theorem primitive_hasDerivAt (x : ℝ) (hx : 0 < x) :
    HasDerivAt primitive (derivative x) x := by
  have hd := (hasDerivAt_phi x).neg.div (hasDerivAt_id x) hx.ne'
  convert hd using 1
  simp only [id_eq, Pi.neg_apply]
  unfold derivative
  field_simp [hx.ne']
  ring

theorem derivative_nonnegative (x : ℝ) : 0 ≤ derivative x := by
  unfold derivative
  exact add_nonneg (phi_nonnegative _) (div_nonneg (phi_nonnegative _) (sq_nonneg _))

theorem derivative_integrableOn (x : ℝ) (hx : 0 < x) : IntegrableOn derivative (Ioi x) := by
  apply integrableOn_Ioi_deriv_of_nonneg'
  · intro y hy
    exact primitive_hasDerivAt y (hx.trans_le hy)
  · intro y _
    exact derivative_nonnegative y
  · exact primitive_tendsto_zero

theorem remainder_integrableOn (x : ℝ) (hx : 0 < x) : IntegrableOn remainder (Ioi x) := by
  have h := (derivative_integrableOn x hx).sub phi_integrable.integrableOn
  have he : derivative - phi = remainder := by
    funext y
    simp [derivative, remainder]
  rwa [he] at h

theorem tail_remainder_identity (x : ℝ) (hx : 0 < x) :
    normalTail x + (∫ y in Ioi x, remainder y) = phi x / x := by
  have hd : ∀ y ∈ Ici x, HasDerivAt primitive (derivative y) y := by
    intro y hy
    exact primitive_hasDerivAt y (hx.trans_le hy)
  have h := integral_Ioi_of_hasDerivAt_of_nonneg' hd
    (fun y _ => derivative_nonnegative y) primitive_tendsto_zero
  have he : derivative = fun y : ℝ => phi y + remainder y := rfl
  rw [he, integral_add phi_integrable.integrableOn (remainder_integrableOn x hx)] at h
  rw [normalTail_eq_integral]
  simpa [primitive, neg_div] using h

/-- Revised Lemma 10's Mills upper bound for the actual standard Gaussian tail. -/
theorem mills_upper (x : ℝ) (hx : 0 < x) : normalTail x ≤ (1 / x) * phi x := by
  have hi := tail_remainder_identity x hx
  have hr : 0 ≤ ∫ y in Ioi x, remainder y := by
    apply integral_nonneg
    intro y
    exact div_nonneg (phi_nonnegative y) (sq_nonneg y)
  have he : (1 / x) * phi x = phi x / x := by ring
  rw [he]
  linarith only [hi, hr]

theorem remainder_integral_le (x : ℝ) (hx : 0 < x) :
    (∫ y in Ioi x, remainder y) ≤ (1 / x ^ 2) * normalTail x := by
  have hmono : (∫ y in Ioi x, remainder y) ≤ ∫ y in Ioi x, (1 / x ^ 2) * phi y := by
    apply setIntegral_mono_on (remainder_integrableOn x hx)
      (phi_integrable.const_mul (1 / x ^ 2)).integrableOn measurableSet_Ioi
    intro y hy
    have hysq : x ^ 2 ≤ y ^ 2 := (sq_le_sq₀ hx.le (hx.trans hy).le).mpr hy.le
    have hinv : 1 / y ^ 2 ≤ 1 / x ^ 2 := one_div_le_one_div_of_le (sq_pos_of_pos hx) hysq
    have hm := mul_le_mul_of_nonneg_right hinv (phi_nonnegative y)
    simpa only [remainder, div_eq_mul_inv, one_mul, mul_comm, mul_one] using hm
  rw [integral_const_mul, ← normalTail_eq_integral] at hmono
  exact hmono

/-- Revised Lemma 10's Mills lower bound, valid for every x>0, including the
range x<1 where its printed lower endpoint is negative. -/
theorem mills_lower (x : ℝ) (hx : 0 < x) :
    (1 / x - 1 / x ^ 3) * phi x ≤ normalTail x := by
  have hi := tail_remainder_identity x hx
  have hr := remainder_integral_le x hx
  have hu := mills_upper x hx
  have hm := mul_le_mul_of_nonneg_left hu (show 0 ≤ 1 / x ^ 2 by positivity)
  have he : (1 / x ^ 2) * ((1 / x) * phi x) = phi x / x ^ 3 := by
    field_simp
  rw [he] at hm
  have he' : (1 / x - 1 / x ^ 3) * phi x = phi x / x - phi x / x ^ 3 := by ring
  rw [he']
  linarith only [hi, hr, hm]

end

end Junpei.TailRatio
