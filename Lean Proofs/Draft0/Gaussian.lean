import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Group.IntegralConvolution
import Mathlib.Tactic

/-!
Corrections of Gaussian formulas in old_version.pdf.

The statements quantify over every real gap/mean and every admissible positive
variance/count. Probability expressions use actual mathlib Gaussian measures.
No statement here is assumed as a paper-specific axiom.
-/

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.Gaussian

noncomputable section

/-- Bayesian loss written as an expectation with a normalized Gaussian law. -/
def posteriorLoss (gap : ℝ) (variance : ℝ≥0) : ℝ :=
  ∫ x : ℝ, max (x - |gap|) 0 ∂gaussianReal 0 variance

/-- The unnormalized Lebesgue integral printed in Lemma 13. -/
def unnormalizedLoss (gap : ℝ) (variance : ℝ≥0) : ℝ :=
  ∫ x : ℝ, max (x - |gap|) 0 * Real.exp (-x ^ 2 / (2 * variance))

/-- Corrected Lemma 13: retain the Gaussian density's normalization factor. -/
theorem lemma13_normalized (gap : ℝ) (variance : ℝ≥0) (hv : 0 < variance) :
    posteriorLoss gap variance =
      (Real.sqrt (2 * Real.pi * variance))⁻¹ * unnormalizedLoss gap variance := by
  unfold posteriorLoss unnormalizedLoss
  rw [integral_gaussianReal_eq_integral_smul (ne_of_gt hv)]
  simp only [gaussianPDFReal, sub_zero, smul_eq_mul]
  calc
    (∫ x : ℝ, (Real.sqrt (2 * Real.pi * variance))⁻¹ *
        Real.exp (-x ^ 2 / (2 * variance)) * max (x - |gap|) 0) =
        ∫ x : ℝ, (Real.sqrt (2 * Real.pi * variance))⁻¹ *
          (max (x - |gap|) 0 * Real.exp (-x ^ 2 / (2 * variance))) := by
      congr 1
      funext x
      ring
    _ = _ := integral_const_mul _ _

/-- The sample-count-weighted posterior update from Equation (3). -/
def posteriorMeanUpdate (oldMean count sample : ℝ) : ℝ :=
  (count * oldMean + sample) / (count + 1)

theorem posterior_increment_eq (oldMean count sample : ℝ) (hc : count + 1 ≠ 0) :
    posteriorMeanUpdate oldMean count sample - oldMean =
      (sample - oldMean) / (count + 1) := by
  unfold posteriorMeanUpdate
  field_simp [hc]
  ring

/-- Convolution of the posterior and independent unit-variance observation noise. -/
theorem section51_predictive_law (oldMean : ℝ) (count : ℝ≥0) :
    (gaussianReal oldMean count⁻¹) ∗ (gaussianReal 0 1) =
      gaussianReal oldMean (1 + count⁻¹) := by
  rw [gaussianReal_conv_gaussianReal]
  simp only [add_zero, add_comm count⁻¹ 1]

/-- Corrected Section 5.1: the posterior-mean increment has mean zero. -/
theorem section51_increment_law (oldMean : ℝ) (count : ℝ≥0) (hc : 0 < count) :
    (gaussianReal oldMean (1 + count⁻¹)).map
        (fun x => posteriorMeanUpdate oldMean count x - oldMean) =
      gaussianReal 0 (count * (count + 1))⁻¹ := by
  have hcR : 0 < (count : ℝ) := hc
  have hc0 : (count : ℝ) ≠ 0 := ne_of_gt hcR
  have hcp : (count : ℝ) + 1 ≠ 0 := ne_of_gt (by linarith)
  have hf : (fun x : ℝ => posteriorMeanUpdate oldMean count x - oldMean) =
      (fun x : ℝ => ((count : ℝ) + 1)⁻¹ * x) ∘ (fun x : ℝ => x - oldMean) := by
    funext x
    rw [posterior_increment_eq _ _ _ hcp]
    simp only [Function.comp_apply, div_eq_mul_inv, mul_comm]
  rw [hf, ← Measure.map_map (by fun_prop) (by fun_prop),
    gaussianReal_map_sub_const, gaussianReal_map_const_mul]
  congr 1
  · simp
  · ext
    simp only [NNReal.coe_mul, NNReal.coe_mk, NNReal.coe_add, NNReal.coe_one,
      NNReal.coe_inv]
    field_simp [hc0, hcp]

/-- The same corrected law with the paper's positive natural-number sample count. -/
theorem section51_increment_law_nat (oldMean : ℝ) (count : ℕ) (hc : 0 < count) :
    (gaussianReal oldMean (1 + (count : ℝ≥0)⁻¹)).map
        (fun x => posteriorMeanUpdate oldMean count x - oldMean) =
      gaussianReal 0 ((count : ℝ≥0) * (count + 1))⁻¹ := by
  simpa using section51_increment_law oldMean (count : ℝ≥0) (by exact_mod_cast hc)

/-- The probability product displayed as g₁ in Lemma 14. -/
def lemma14_g1 (gap : ℝ) (varianceL varianceR : ℝ≥0) : ℝ :=
  (gaussianReal 0 varianceL (Icc |gap| (2 * |gap|))).toReal *
  (gaussianReal 0 varianceR (Ici |gap|)).toReal * |gap|

theorem lemma14_g1_nonnegative (gap : ℝ) (varianceL varianceR : ℝ≥0) :
    0 ≤ lemma14_g1 gap varianceL varianceR := by
  unfold lemma14_g1
  positivity

theorem lemma14_g1_at_zero (varianceL varianceR : ℝ≥0) :
    lemma14_g1 0 varianceL varianceR = 0 := by
  simp [lemma14_g1]

/-- Every interval of positive length has positive probability under a
positive-variance Gaussian. This uses equivalence of Gaussian and Lebesgue
null sets, rather than numerical quadrature. -/
theorem gaussian_Icc_probability_pos (mean a b : ℝ) (variance : ℝ≥0)
    (hv : 0 < variance) (hab : a < b) :
    0 < (gaussianReal mean variance (Icc a b)).toReal := by
  apply ENNReal.toReal_pos _ (measure_ne_top _ _)
  intro hz
  have hzv := gaussianReal_absolutelyContinuous' mean (ne_of_gt hv) hz
  rw [Real.volume_Icc] at hzv
  exact (ENNReal.ofReal_pos.mpr (sub_pos.mpr hab)).ne' hzv

theorem gaussian_Ici_probability_pos (mean a : ℝ) (variance : ℝ≥0)
    (hv : 0 < variance) :
    0 < (gaussianReal mean variance (Ici a)).toReal := by
  apply ENNReal.toReal_pos _ (measure_ne_top _ _)
  intro hz
  have hsub : Icc a (a + 1) ⊆ Ici a := fun _ hx => hx.1
  have hzcc := measure_mono_null hsub hz
  have hp := gaussian_Icc_probability_pos mean a (a + 1) variance hv (by linarith)
  rw [hzcc, ENNReal.toReal_zero] at hp
  exact (lt_irrefl (0 : ℝ)) hp

/-- Corrected Lemma 14 positivity has the necessary nonzero-gap hypothesis. -/
theorem lemma14_g1_positive (gap : ℝ) (varianceL varianceR : ℝ≥0)
    (hgap : gap ≠ 0) (hl : 0 < varianceL) (hr : 0 < varianceR) :
    0 < lemma14_g1 gap varianceL varianceR := by
  have hg : 0 < |gap| := abs_pos.mpr hgap
  unfold lemma14_g1
  exact mul_pos
    (mul_pos (gaussian_Icc_probability_pos 0 _ _ varianceL hl (by linarith))
      (gaussian_Ici_probability_pos 0 _ varianceR hr)) hg

/-- The exact boundary condition for strict positivity of the printed product. -/
theorem lemma14_g1_positive_iff (gap : ℝ) (varianceL varianceR : ℝ≥0)
    (hl : 0 < varianceL) (hr : 0 < varianceR) :
    0 < lemma14_g1 gap varianceL varianceR ↔ gap ≠ 0 := by
  constructor
  · intro hp hz
    subst gap
    rw [lemma14_g1_at_zero] at hp
    exact (lt_irrefl (0 : ℝ)) hp
  · intro hgap
    exact lemma14_g1_positive gap varianceL varianceR hgap hl hr

/-- Gaussian hinge expectation without taking the absolute value of its center. -/
def hingeLoss (center : ℝ) (variance : ℝ≥0) : ℝ :=
  ∫ x : ℝ, max (x - center) 0 ∂gaussianReal 0 variance

theorem gaussian_id_integrable (mean : ℝ) (variance : ℝ≥0) :
    Integrable (fun x : ℝ => x) (gaussianReal mean variance) := by
  exact memLp_one_iff_integrable.mp (memLp_id_gaussianReal 1)

theorem hinge_integrable (center : ℝ) (variance : ℝ≥0) :
    Integrable (fun x : ℝ => max (x - center) 0) (gaussianReal 0 variance) := by
  exact ((gaussian_id_integrable 0 variance).sub (integrable_const center)).pos_part

theorem hingeLoss_nonnegative (center : ℝ) (variance : ℝ≥0) :
    0 ≤ hingeLoss center variance := by
  exact integral_nonneg (fun _ => le_max_right _ _)

/-- A positive-variance Gaussian has a strictly positive hinge expectation
at every finite threshold, including threshold zero. -/
theorem hingeLoss_positive (center : ℝ) (variance : ℝ≥0) (hv : 0 < variance) :
    0 < hingeLoss center variance := by
  apply (integral_pos_iff_support_of_nonneg (fun _ => le_max_right _ _)
    (hinge_integrable center variance)).mpr
  have hp := gaussian_Icc_probability_pos 0 (center + 1) (center + 2) variance hv
    (by linarith)
  have hm := (ENNReal.toReal_pos_iff.mp hp).1
  apply lt_of_lt_of_le hm (measure_mono ?_)
  intro x hx
  change max (x - center) 0 ≠ 0
  have hxpos : 0 < x - center := by linarith [hx.1]
  rw [max_eq_left hxpos.le]
  exact ne_of_gt hxpos

/-- Gaussian symmetry and the identity u₊-(-u)₊=u. -/
theorem hingeLoss_reflection (center : ℝ) (variance : ℝ≥0) :
    hingeLoss center variance - hingeLoss (-center) variance = -center := by
  have hsym : (gaussianReal 0 variance).map (fun x : ℝ => -x) =
      gaussianReal 0 variance := by simpa using (gaussianReal_map_neg (μ := (0 : ℝ)))
  have href : hingeLoss (-center) variance =
      ∫ x : ℝ, max (-(x - center)) 0 ∂gaussianReal 0 variance := by
    unfold hingeLoss
    calc
      (∫ x : ℝ, max (x - -center) 0 ∂gaussianReal 0 variance) =
          ∫ x : ℝ, max (x - -center) 0
            ∂((gaussianReal 0 variance).map (fun x : ℝ => -x)) := by rw [hsym]
      _ = ∫ x : ℝ, max (-x - -center) 0 ∂gaussianReal 0 variance := by
        rw [integral_map (by fun_prop) (by fun_prop)]
      _ = _ := by
        congr 1
        funext x
        congr 1
        ring
  have hi := ((gaussian_id_integrable 0 variance).sub (integrable_const center))
  have hiNeg : Integrable (fun x : ℝ => max (-(x - center)) 0)
      (gaussianReal 0 variance) := by
    simpa only [Pi.neg_apply, Pi.sub_apply] using hi.neg.pos_part
  rw [href, hingeLoss, ← integral_sub (hinge_integrable center variance) hiNeg]
  calc
    (∫ x : ℝ, (max (x - center) 0 - max (-(x - center)) 0)
        ∂gaussianReal 0 variance) = ∫ x : ℝ, (x - center) ∂gaussianReal 0 variance := by
      congr 1
      funext x
      by_cases hx : 0 ≤ x - center
      · rw [max_eq_left hx, max_eq_right (by linarith)]
        ring
      · rw [max_eq_right (le_of_not_ge hx), max_eq_left (by linarith)]
        ring
    _ = -center := by rw [integral_sub (gaussian_id_integrable 0 variance)
      (integrable_const center), integral_id_gaussianReal]; simp

theorem hingeLoss_sub_abs (center : ℝ) (variance : ℝ≥0) :
    hingeLoss center variance - hingeLoss |center| variance = max (-center) 0 := by
  by_cases hc : 0 ≤ center
  · rw [abs_of_nonneg hc, sub_self, max_eq_right (by linarith)]
  · have hcn : center < 0 := lt_of_not_ge hc
    rw [abs_of_neg hcn, hingeLoss_reflection, max_eq_left (by linarith)]

/-- Expected loss after one draw, with posterior variance varianceR and
independent mean-update variance varianceL. -/
def expectedLossAfterDraw (gap : ℝ) (varianceL varianceR : ℝ≥0) : ℝ :=
  ∫ l : ℝ, posteriorLoss (|gap| - l) varianceR ∂gaussianReal 0 varianceL

/-- Correct one-step improvement, as a Gaussian hinge expectation. -/
def oneStepImprovement (gap : ℝ) (varianceL varianceR : ℝ≥0) : ℝ :=
  posteriorLoss gap (varianceL + varianceR) -
    expectedLossAfterDraw gap varianceL varianceR

/-- The exact loss reduction is independent of the remaining posterior
variance: it equals the expected positive part of the posterior mean update
minus the current gap. -/
theorem oneStepImprovement_eq_hinge (gap : ℝ) (varianceL varianceR : ℝ≥0) :
    oneStepImprovement gap varianceL varianceR = hingeLoss |gap| varianceL := by
  have hconv : (gaussianReal 0 varianceL) ∗ (gaussianReal 0 varianceR) =
      gaussianReal 0 (varianceL + varianceR) := by
    simpa using (gaussianReal_conv_gaussianReal (m₁ := (0 : ℝ)) (m₂ := (0 : ℝ))
      (v₁ := varianceL) (v₂ := varianceR))
  have hi : Integrable (fun x : ℝ => max (x - |gap|) 0)
      ((gaussianReal 0 varianceL) ∗ (gaussianReal 0 varianceR)) := by
    rw [hconv]
    exact hinge_integrable |gap| (varianceL + varianceR)
  have hbefore : posteriorLoss gap (varianceL + varianceR) =
      ∫ l : ℝ, hingeLoss (|gap| - l) varianceR ∂gaussianReal 0 varianceL := by
    unfold posteriorLoss hingeLoss
    rw [← hconv, integral_conv hi]
    congr 1
    funext l
    congr 1
    funext r
    congr 1
    ring
  have hiOuter : Integrable (fun l : ℝ => hingeLoss (|gap| - l) varianceR)
      (gaussianReal 0 varianceL) := by
    have h := (integrable_conv_iff hi.aestronglyMeasurable).mp hi
    have hf : (fun l : ℝ => ∫ r : ℝ, ‖max (l + r - |gap|) 0‖
        ∂gaussianReal 0 varianceR) = (fun l : ℝ => hingeLoss (|gap| - l) varianceR) := by
      funext l
      unfold hingeLoss
      congr 1
      funext r
      rw [Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
      congr 1
      ring
    rw [hf] at h
    exact h.2
  have hpoint : (fun l : ℝ => posteriorLoss (|gap| - l) varianceR) =
      (fun l : ℝ => hingeLoss (|gap| - l) varianceR - max (l - |gap|) 0) := by
    funext l
    have h := hingeLoss_sub_abs (|gap| - l) varianceR
    change hingeLoss (|(|gap| - l)|) varianceR = _
    have hneg : -(|gap| - l) = l - |gap| := by ring
    rw [hneg] at h
    linarith
  unfold oneStepImprovement expectedLossAfterDraw
  rw [hbefore, hpoint, integral_sub hiOuter (hinge_integrable |gap| varianceL)]
  unfold hingeLoss
  ring

/-- Corrected strict improvement holds at every gap, including a tie.
Only the sampled arm's mean-update variance must be positive. -/
theorem oneStepImprovement_positive (gap : ℝ) (varianceL varianceR : ℝ≥0)
    (hl : 0 < varianceL) :
    0 < oneStepImprovement gap varianceL varianceR := by
  rw [oneStepImprovement_eq_hinge]
  exact hingeLoss_positive |gap| varianceL hl

/-- A usable lower bound comes from an interval on which the hinge is at
least epsilon. Unlike the printed g₁, it can remain positive at a tie. -/
theorem hingeLoss_ge_interval (center epsilon : ℝ) (variance : ℝ≥0) :
    epsilon * (gaussianReal 0 variance
      (Icc (center + epsilon) (center + 2 * epsilon))).toReal ≤
    hingeLoss center variance := by
  calc
    _ ≤ ∫ x : ℝ in Icc (center + epsilon) (center + 2 * epsilon),
        max (x - center) 0 ∂gaussianReal 0 variance := by
      exact setIntegral_ge_of_const_le measurableSet_Icc (measure_ne_top _ _)
        (fun x hx => le_trans (by linarith [hx.1]) (le_max_left _ _))
        (hinge_integrable center variance).restrict
    _ ≤ _ := setIntegral_le_integral (hinge_integrable center variance)
      (ae_of_all _ (fun _ => le_max_right _ _))

def correctedImprovementLowerBound (gap epsilon : ℝ) (varianceL : ℝ≥0) : ℝ :=
  epsilon * (gaussianReal 0 varianceL
    (Icc (|gap| + epsilon) (|gap| + 2 * epsilon))).toReal

theorem correctedImprovementLowerBound_positive (gap epsilon : ℝ) (varianceL : ℝ≥0)
    (he : 0 < epsilon) (hl : 0 < varianceL) :
    0 < correctedImprovementLowerBound gap epsilon varianceL := by
  exact mul_pos he (gaussian_Icc_probability_pos 0 _ _ varianceL hl (by linarith))

theorem oneStepImprovement_ge_corrected_bound (gap epsilon : ℝ)
    (varianceL varianceR : ℝ≥0) :
    correctedImprovementLowerBound gap epsilon varianceL ≤
      oneStepImprovement gap varianceL varianceR := by
  rw [oneStepImprovement_eq_hinge]
  exact hingeLoss_ge_interval |gap| epsilon varianceL

def updateVariance (count : ℝ≥0) : ℝ≥0 := (count * (count + 1))⁻¹

def posteriorVariance (countL countR : ℝ≥0) : ℝ≥0 := countL⁻¹ + countR⁻¹

/-- The posterior variance splits into update and remaining variances. -/
theorem posteriorVariance_after_draw (countL countR : ℝ≥0) (hc : 0 < countL) :
    posteriorVariance countL countR =
      updateVariance countL + posteriorVariance (countL + 1) countR := by
  have hcR : 0 < (countL : ℝ) := hc
  have hc0 : (countL : ℝ) ≠ 0 := ne_of_gt hcR
  have hcp : (countL : ℝ) + 1 ≠ 0 := ne_of_gt (by linarith)
  ext
  simp only [posteriorVariance, updateVariance, NNReal.coe_inv, NNReal.coe_add,
    NNReal.coe_mul, NNReal.coe_one]
  field_simp [hc0, hcp]
  ring

/-- Strict terminal loss reduction for every real gap and every positive
sample count of the drawn arm, using the paper's actual variance update. -/
theorem terminal_loss_strictly_reduced (gap : ℝ) (countL countR : ℝ≥0)
    (hc : 0 < countL) :
    expectedLossAfterDraw gap (updateVariance countL)
        (posteriorVariance (countL + 1) countR) <
      posteriorLoss gap (posteriorVariance countL countR) := by
  rw [posteriorVariance_after_draw countL countR hc]
  have hp := oneStepImprovement_positive gap (updateVariance countL)
    (posteriorVariance (countL + 1) countR) (by unfold updateVariance; positivity)
  unfold oneStepImprovement at hp
  exact sub_pos.mp hp

/-- Difference of independent Gaussian posteriors has the sum of variances. -/
theorem posterior_difference_law (meanL meanR : ℝ) (countL countR : ℝ≥0) :
    (gaussianReal meanL countL⁻¹) ∗
      ((gaussianReal meanR countR⁻¹).map (fun x : ℝ => -x)) =
    gaussianReal (meanL - meanR) (posteriorVariance countL countR) := by
  rw [gaussianReal_map_neg, gaussianReal_conv_gaussianReal]
  rfl

/-- Coupling all zero-mean Gaussians by scaling a unit-variance Gaussian. -/
theorem hingeLoss_eq_standard_integral (center : ℝ) (variance : ℝ≥0) :
    hingeLoss center variance =
      ∫ x : ℝ, max (Real.sqrt (variance : ℝ) * x - center) 0
        ∂gaussianReal 0 1 := by
  have hv : (⟨(Real.sqrt (variance : ℝ)) ^ 2, sq_nonneg _⟩ : ℝ≥0) * 1 = variance := by
    ext
    simp only [mul_one]
    exact Real.sq_sqrt variance.property
  have hmap : (gaussianReal 0 1).map (fun x : ℝ => Real.sqrt (variance : ℝ) * x) =
      gaussianReal 0 variance := by
    rw [gaussianReal_map_const_mul, mul_zero, hv]
  unfold hingeLoss
  rw [← hmap, integral_map (by fun_prop) (by fun_prop)]

/-- For a nonnegative gap, increasing the variance strictly increases the
expected gain. The proof couples Gaussians and integrates a pointwise strict
inequality on an interval of positive Gaussian probability. -/
theorem hingeLoss_strictMono_variance (center : ℝ) (hcenter : 0 ≤ center)
    (varianceL varianceR : ℝ≥0) (hv : varianceL < varianceR) :
    hingeLoss center varianceL < hingeLoss center varianceR := by
  let sL := Real.sqrt (varianceL : ℝ)
  let sR := Real.sqrt (varianceR : ℝ)
  have hsL : 0 ≤ sL := Real.sqrt_nonneg _
  have hs : sL < sR := Real.sqrt_lt_sqrt varianceL.property hv
  have hsR : 0 < sR := lt_of_le_of_lt hsL hs
  have hiL : Integrable (fun x : ℝ => max (sL * x - center) 0) (gaussianReal 0 1) :=
    (((gaussian_id_integrable 0 1).const_mul sL).sub (integrable_const center)).pos_part
  have hiR : Integrable (fun x : ℝ => max (sR * x - center) 0) (gaussianReal 0 1) :=
    (((gaussian_id_integrable 0 1).const_mul sR).sub (integrable_const center)).pos_part
  have hmono : ∀ x : ℝ, max (sL * x - center) 0 ≤ max (sR * x - center) 0 := by
    intro x
    by_cases hx : 0 ≤ x
    · exact max_le_max (sub_le_sub_right (mul_le_mul_of_nonneg_right hs.le hx) center) le_rfl
    · have hxn : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
      rw [max_eq_right (by nlinarith : sL * x - center ≤ 0)]
      exact le_max_right _ _
  have hp : 0 < ∫ x : ℝ, (max (sR * x - center) 0 - max (sL * x - center) 0)
      ∂gaussianReal 0 1 := by
    apply (integral_pos_iff_support_of_nonneg
      (fun x => sub_nonneg.mpr (hmono x)) (hiR.sub hiL)).mpr
    let a := (center + 1) / sR
    have ha : 0 < a := div_pos (by linarith) hsR
    have hprob := gaussian_Icc_probability_pos 0 a (a + 1) 1 (by norm_num) (by linarith)
    apply lt_of_lt_of_le (ENNReal.toReal_pos_iff.mp hprob).1 (measure_mono ?_)
    intro x hx
    change max (sR * x - center) 0 - max (sL * x - center) 0 ≠ 0
    have hxpos : 0 < x := lt_of_lt_of_le ha hx.1
    have hxR : center + 1 ≤ sR * x := by
      have hh := (div_le_iff₀ hsR).mp hx.1
      simpa only [mul_comm x sR] using hh
    have hmul : sL * x < sR * x := mul_lt_mul_of_pos_right hs hxpos
    have hmax : max (sL * x - center) 0 < max (sR * x - center) 0 := by
      rw [max_eq_left (by linarith : 0 ≤ sR * x - center)]
      exact max_lt_iff.mpr ⟨by linarith, by linarith⟩
    exact ne_of_gt (sub_pos.mpr hmax)
  rw [integral_sub hiR hiL] at hp
  rw [hingeLoss_eq_standard_integral, hingeLoss_eq_standard_integral]
  exact sub_pos.mp hp

/-- One-step gain is strictly increasing in the update variance. -/
theorem oneStepImprovement_strictMono_updateVariance (gap : ℝ)
    (varianceL varianceL' varianceR varianceR' : ℝ≥0) (hv : varianceL < varianceL') :
    oneStepImprovement gap varianceL varianceR <
      oneStepImprovement gap varianceL' varianceR' := by
  rw [oneStepImprovement_eq_hinge, oneStepImprovement_eq_hinge]
  exact hingeLoss_strictMono_variance |gap| (abs_nonneg _) varianceL varianceL' hv

theorem updateVariance_strictAntitone (countL countR : ℝ≥0)
    (hl : 0 < countL) (hcounts : countL < countR) :
    updateVariance countR < updateVariance countL := by
  have hr : 0 < countR := lt_trans hl hcounts
  unfold updateVariance
  apply inv_lt_inv₀ (by positivity) (by positivity) |>.mpr
  have hlR : 0 < (countL : ℝ) := hl
  have hcountsR : (countL : ℝ) < countR := hcounts
  change (countL : ℝ) * ((countL : ℝ) + 1) <
    (countR : ℝ) * ((countR : ℝ) + 1)
  nlinarith

theorem posteriorVariance_comm (countL countR : ℝ≥0) :
    posteriorVariance countL countR = posteriorVariance countR countL := by
  unfold posteriorVariance
  exact add_comm _ _

/-- Corrected terminal version of Lemma 15: sampling the arm with the
smaller count gives the strictly smaller expected terminal loss. -/
theorem terminal_sample_smaller_count_optimal (gap : ℝ) (countL countR : ℝ≥0)
    (hr : 0 < countR) (hcounts : countR < countL) :
    expectedLossAfterDraw gap (updateVariance countR)
        (posteriorVariance (countR + 1) countL) <
      expectedLossAfterDraw gap (updateVariance countL)
        (posteriorVariance (countL + 1) countR) := by
  have hl : 0 < countL := lt_trans hr hcounts
  have hvariance := updateVariance_strictAntitone countR countL hr hcounts
  have hgain := hingeLoss_strictMono_variance |gap| (abs_nonneg _)
    (updateVariance countL) (updateVariance countR) hvariance
  have hleft := oneStepImprovement_eq_hinge gap (updateVariance countL)
    (posteriorVariance (countL + 1) countR)
  have hright := oneStepImprovement_eq_hinge gap (updateVariance countR)
    (posteriorVariance (countR + 1) countL)
  unfold oneStepImprovement at hleft hright
  rw [← posteriorVariance_after_draw countL countR hl] at hleft
  rw [← posteriorVariance_after_draw countR countL hr,
    posteriorVariance_comm countR countL] at hright
  linarith

theorem gaussian_positivePart_integral (mean : ℝ) (variance : ℝ≥0) :
    (∫ x : ℝ, max x 0 ∂gaussianReal mean variance) = hingeLoss (-mean) variance := by
  have hmap := gaussianReal_map_add_const (μ := (0 : ℝ)) (v := variance) mean
  simp only [zero_add] at hmap
  rw [← hmap, integral_map (by fun_prop) (by fun_prop)]
  unfold hingeLoss
  congr 1
  funext x
  congr 1
  ring

theorem gaussian_negativePart_integral (mean : ℝ) (variance : ℝ≥0) :
    (∫ x : ℝ, max (-x) 0 ∂gaussianReal mean variance) = hingeLoss mean variance := by
  have hmap := gaussianReal_map_neg (μ := mean) (v := variance)
  have hm : (∫ x : ℝ, max (-x) 0 ∂gaussianReal mean variance) =
      ∫ x : ℝ, max x 0 ∂gaussianReal (-mean) variance := by
    rw [← hmap, integral_map (by fun_prop) (by fun_prop)]
  rw [hm, gaussian_positivePart_integral, neg_neg]

/-- Posterior expected simple regret when recommending the arm with the
larger posterior mean (choose the right arm at a tie). The posterior
difference law is Gaussian with the sum of posterior variances. -/
def twoArmPosteriorLoss (meanL meanR : ℝ) (varianceL varianceR : ℝ≥0) : ℝ :=
  if meanL ≤ meanR then
    ∫ d : ℝ, max d 0 ∂gaussianReal (meanL - meanR) (varianceL + varianceR)
  else
    ∫ d : ℝ, max (-d) 0 ∂gaussianReal (meanL - meanR) (varianceL + varianceR)

/-- Corrected Lemma 13 also holds for the explicit two-arm recommendation
loss, for either ordering of the posterior means. -/
theorem twoArmPosteriorLoss_eq (meanL meanR : ℝ) (varianceL varianceR : ℝ≥0) :
    twoArmPosteriorLoss meanL meanR varianceL varianceR =
      posteriorLoss (meanL - meanR) (varianceL + varianceR) := by
  unfold twoArmPosteriorLoss
  by_cases hm : meanL ≤ meanR
  · rw [if_pos hm, gaussian_positivePart_integral]
    unfold posteriorLoss hingeLoss
    rw [abs_of_nonpos (sub_nonpos.mpr hm)]
  · rw [if_neg hm, gaussian_negativePart_integral]
    unfold posteriorLoss hingeLoss
    rw [abs_of_nonneg (sub_nonneg.mpr (le_of_lt (lt_of_not_ge hm)))]

theorem twoArmPosteriorLoss_normalized (meanL meanR : ℝ) (varianceL varianceR : ℝ≥0)
    (hv : 0 < varianceL + varianceR) :
    twoArmPosteriorLoss meanL meanR varianceL varianceR =
      (Real.sqrt (2 * Real.pi * (varianceL + varianceR)))⁻¹ *
        unnormalizedLoss (meanL - meanR) (varianceL + varianceR) := by
  rw [twoArmPosteriorLoss_eq]
  exact lemma13_normalized _ _ hv

end

end Draft0.Gaussian
