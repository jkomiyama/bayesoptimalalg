import Draft0.Gaussian
import Draft0.Rates
import Draft0.Counts

/-!
# Quantitative corrected terminal loss bounds

The bounds use the actual Gaussian measures and loss definitions proved in
Draft0.Gaussian. In particular the lower bound for Lemma 14 is valid at gap
zero, where the product printed in the paper vanishes.
-/

open MeasureTheory ProbabilityTheory Set
open Filter Asymptotics
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.TerminalBounds

noncomputable section

open Draft0.Gaussian Draft0.Rates

def gainConstant : ℝ := Real.exp (-5) * (Real.sqrt (4 * Real.pi))⁻¹

theorem gainConstant_positive : 0 < gainConstant := by
  unfold gainConstant
  exact mul_pos (Real.exp_pos _) (inv_pos.mpr (Real.sqrt_pos.mpr (by positivity)))

theorem terminal_gaussianPDF_lower (T : ℕ) (gap x : ℝ) (v : ℝ≥0)
    (hT : 5 ≤ T) (hgap : |gap| ≤ 2 * invPower 2 T)
    (hv : 0 < v) (hvupper : (v : ℝ) ≤ 2) (hvlower : invPower 2 T ≤ (v : ℝ))
    (hx : x ∈ Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T)) :
    gainConstant ≤ gaussianPDFReal 0 v x := by
  have hTp : 0 < T := by omega
  have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT
  have hvp : 0 < (v : ℝ) := hv
  have hep := invPower_pos 2 T hTp
  have hehalf : invPower 2 T ≤ 1 / 2 := by
    unfold invPower
    rw [inv_eq_one_div]
    apply (div_le_iff₀ (by positivity)).mpr
    nlinarith only [hTr]
  have hx0 : 0 ≤ x := by linarith only [hx.1, abs_nonneg gap, hep]
  have hxupper : x ≤ 4 * invPower 2 T := by linarith only [hx.2, hgap]
  have hxsq : x ^ 2 ≤ (4 * invPower 2 T) ^ 2 :=
    (sq_le_sq₀ hx0 (by positivity)).mpr hxupper
  have heq : (invPower 2 T) ^ 2 ≤ (invPower 2 T) / 2 := by
    nlinarith only [hehalf, hep]
  have hquad : x ^ 2 ≤ 10 * (v : ℝ) := by
    nlinarith only [hxsq, heq, hvlower, hep]
  have hquot : x ^ 2 / (2 * (v : ℝ)) ≤ 5 := by
    apply (div_le_iff₀ (by positivity)).mpr
    linarith only [hquad]
  have hexp : Real.exp (-5) ≤ Real.exp (-x ^ 2 / (2 * (v : ℝ))) := by
    apply Real.exp_le_exp.mpr
    rw [neg_div]
    linarith only [hquot]
  have hsqrtp : 0 < Real.sqrt (2 * Real.pi * (v : ℝ)) := by positivity
  have hsqrt4p : 0 < Real.sqrt (4 * Real.pi) := by positivity
  have hnorm : (Real.sqrt (4 * Real.pi))⁻¹ ≤
      (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ := by
    apply (inv_le_inv₀ hsqrt4p hsqrtp).mpr
    apply Real.sqrt_le_sqrt
    nlinarith only [hvupper, Real.pi_pos]
  calc
    gainConstant = (Real.sqrt (4 * Real.pi))⁻¹ * Real.exp (-5) := by
      unfold gainConstant
      ring
    _ ≤ (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ *
        Real.exp (-x ^ 2 / (2 * (v : ℝ))) :=
      mul_le_mul hnorm hexp (Real.exp_pos _).le (inv_nonneg.mpr hsqrtp.le)
    _ = gaussianPDFReal 0 v x := by simp only [gaussianPDFReal, sub_zero]

theorem terminal_gaussian_interval_lower (T : ℕ) (gap : ℝ) (v : ℝ≥0)
    (hT : 5 ≤ T) (hgap : |gap| ≤ 2 * invPower 2 T)
    (hv : 0 < v) (hvupper : (v : ℝ) ≤ 2) (hvlower : invPower 2 T ≤ (v : ℝ)) :
    gainConstant * invPower 2 T ≤
      (gaussianReal 0 v (Icc (|gap| + invPower 2 T)
        (|gap| + 2 * invPower 2 T))).toReal := by
  have he0 := invPower_nonneg 2 T
  have hmass : (gaussianReal 0 v (Icc (|gap| + invPower 2 T)
        (|gap| + 2 * invPower 2 T))).toReal =
      ∫ x in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T), gaussianPDFReal 0 v x := by
    rw [gaussianReal_apply_eq_integral 0 (ne_of_gt hv), ENNReal.toReal_ofReal]
    exact integral_nonneg (gaussianPDFReal_nonneg 0 v)
  have hmono : (∫ _ in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T), gainConstant) ≤
      ∫ x in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T), gaussianPDFReal 0 v x := by
    apply setIntegral_mono_on
      (integrableOn_const isCompact_Icc.measure_ne_top)
      (integrable_gaussianPDFReal 0 v).integrableOn measurableSet_Icc
    intro x hx
    exact terminal_gaussianPDF_lower T gap x v hT hgap hv hvupper hvlower hx
  have hconst : (∫ _ in Icc (|gap| + invPower 2 T) (|gap| + 2 * invPower 2 T), gainConstant) =
      gainConstant * invPower 2 T := by
    rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Icc,
      ENNReal.toReal_ofReal (by linarith)]
    ring
  rw [← hconst, hmass]
  exact hmono

/-- Corrected Lemma 14: a uniform lower bound on the actual improvement,
including when the two posterior means are tied. -/
theorem lemma14_uniform_gain_lower (T : ℕ) (gap : ℝ) (varianceL varianceR : ℝ≥0)
    (hT : 5 ≤ T) (hgap : |gap| ≤ 2 * invPower 2 T)
    (hv : 0 < varianceL) (hvupper : (varianceL : ℝ) ≤ 2)
    (hvlower : invPower 2 T ≤ (varianceL : ℝ)) :
    gainConstant * invPower 4 T ≤ oneStepImprovement gap varianceL varianceR := by
  have hmass := terminal_gaussian_interval_lower T gap varianceL hT hgap hv hvupper hvlower
  have hscale := mul_le_mul_of_nonneg_left hmass (invPower_nonneg 2 T)
  have hbound := oneStepImprovement_ge_corrected_bound gap (invPower 2 T) varianceL varianceR
  unfold correctedImprovementLowerBound at hbound
  calc
    gainConstant * invPower 4 T =
        invPower 2 T * (gainConstant * invPower 2 T) := by
      rw [← invPower_product 2 2 T]
      ring
    _ ≤ invPower 2 T *
        (gaussianReal 0 varianceL (Icc (|gap| + invPower 2 T)
          (|gap| + 2 * invPower 2 T))).toReal := hscale
    _ ≤ oneStepImprovement gap varianceL varianceR := hbound

theorem lemma14_uniform_gain_conservative (T : ℕ) (gap : ℝ) (varianceL varianceR : ℝ≥0)
    (hT : 5 ≤ T) (hgap : |gap| ≤ 2 * invPower 2 T)
    (hv : 0 < varianceL) (hvupper : (varianceL : ℝ) ≤ 2)
    (hvlower : invPower 2 T ≤ (varianceL : ℝ)) :
    gainConstant * invPower 6 T ≤ oneStepImprovement gap varianceL varianceR := by
  exact (mul_le_mul_of_nonneg_left
    (invPower_antitone (by norm_num : 4 ≤ 6) (by omega : 1 ≤ T))
    gainConstant_positive.le).trans
      (lemma14_uniform_gain_lower T gap varianceL varianceR hT hgap hv hvupper hvlower)

theorem updateVariance_count_bounds (n T : ℕ) (hn : 1 ≤ n) (hnT : n < T) :
    invPower 2 T ≤ (updateVariance (n : ℝ≥0) : ℝ) ∧
      (updateVariance (n : ℝ≥0) : ℝ) ≤ 1 / 2 := by
  have hnr : 1 ≤ (n : ℝ) := by exact_mod_cast hn
  have hTr : 1 ≤ (T : ℝ) := by exact_mod_cast (show 1 ≤ T by omega)
  have hnTr : (n : ℝ) + 1 ≤ (T : ℝ) := by exact_mod_cast (show n + 1 ≤ T by omega)
  simp only [updateVariance, NNReal.coe_inv, NNReal.coe_mul, NNReal.coe_add,
    NNReal.coe_natCast, NNReal.coe_one, invPower]
  constructor
  · apply (inv_le_inv₀ (by positivity) (by positivity)).mpr
    nlinarith only [hnTr, hnr, hTr]
  · rw [inv_eq_one_div]
    apply (div_le_iff₀ (by positivity)).mpr
    nlinarith only [hnr]

/-- Lemma 14's quantitative conclusion for the paper's actual sampled arm
variance, with no variance lower bound supplied as a hypothesis. -/
theorem lemma14_sample_count_lower (T n : ℕ) (gap : ℝ) (varianceR : ℝ≥0)
    (hT : 5 ≤ T) (hn : 1 ≤ n) (hnT : n < T)
    (hgap : |gap| ≤ 2 * invPower 2 T) :
    gainConstant * invPower 6 T ≤
      oneStepImprovement gap (updateVariance (n : ℝ≥0)) varianceR := by
  have hb := updateVariance_count_bounds n T hn hnT
  apply lemma14_uniform_gain_conservative T gap _ varianceR hT hgap
  · unfold updateVariance
    have hnp : 0 < (n : ℝ≥0) := by exact_mod_cast (show 0 < n by omega)
    positivity
  · linarith only [hb.2]
  · exact hb.1

def comparisonConstant : ℝ := (gaussianReal 0 1 (Icc (1 : ℝ) 2)).toReal / 2

theorem comparisonConstant_positive : 0 < comparisonConstant := by
  exact div_pos (gaussian_Icc_probability_pos 0 1 2 1 (by norm_num) (by norm_num))
    (by norm_num)

/-- A quantitative form of the standard-Gaussian coupling used in Lemma 15.
On x∈[1,2] both hinges are linear if the smaller standard deviation is at
least the center, and their difference is at least the standard deviation gap. -/
theorem hingeLoss_difference_ge_standard_interval (center : ℝ) (vL vR : ℝ≥0)
    (hcenter : 0 ≤ center) (hsmall : center ≤ Real.sqrt (vL : ℝ)) (hv : vL ≤ vR) :
    (Real.sqrt (vR : ℝ) - Real.sqrt (vL : ℝ)) *
        (gaussianReal 0 1 (Icc (1 : ℝ) 2)).toReal ≤
      hingeLoss center vR - hingeLoss center vL := by
  let sL := Real.sqrt (vL : ℝ)
  let sR := Real.sqrt (vR : ℝ)
  have hsL : 0 ≤ sL := Real.sqrt_nonneg _
  have hsR : 0 ≤ sR := Real.sqrt_nonneg _
  have hs : sL ≤ sR := Real.sqrt_le_sqrt hv
  have hiL : Integrable (fun x : ℝ => max (sL * x - center) 0) (gaussianReal 0 1) :=
    (((gaussian_id_integrable 0 1).const_mul sL).sub (integrable_const center)).pos_part
  have hiR : Integrable (fun x : ℝ => max (sR * x - center) 0) (gaussianReal 0 1) :=
    (((gaussian_id_integrable 0 1).const_mul sR).sub (integrable_const center)).pos_part
  have hmono : ∀ x : ℝ, 0 ≤ max (sR * x - center) 0 - max (sL * x - center) 0 := by
    intro x
    apply sub_nonneg.mpr
    by_cases hx : 0 ≤ x
    · exact max_le_max (sub_le_sub_right (mul_le_mul_of_nonneg_right hs hx) center) le_rfl
    · rw [max_eq_right (by nlinarith only [hsL, hcenter, hx] : sL * x - center ≤ 0)]
      exact le_max_right _ _
  have hinterval : ∀ x ∈ Icc (1 : ℝ) 2,
      sR - sL ≤ max (sR * x - center) 0 - max (sL * x - center) 0 := by
    intro x hx
    have hSLcenter : center ≤ sL := hsmall
    have hx0 : 0 ≤ x := by linarith only [hx.1]
    have hleft : 0 ≤ sL * x - center := by nlinarith only [hx.1, hsL, hSLcenter]
    have hright : 0 ≤ sR * x - center := by nlinarith only [hx.1, hsR, hs, hSLcenter]
    rw [max_eq_left hleft, max_eq_left hright]
    nlinarith only [hx.1, hs]
  have hint : (sR - sL) * (gaussianReal 0 1 (Icc (1 : ℝ) 2)).toReal ≤
      ∫ x : ℝ, (max (sR * x - center) 0 - max (sL * x - center) 0) ∂gaussianReal 0 1 := by
    calc
      _ ≤ ∫ x : ℝ in Icc (1 : ℝ) 2,
          (max (sR * x - center) 0 - max (sL * x - center) 0) ∂gaussianReal 0 1 := by
        exact setIntegral_ge_of_const_le measurableSet_Icc (measure_ne_top _ _)
          hinterval (hiR.sub hiL).restrict
      _ ≤ _ := setIntegral_le_integral (hiR.sub hiL) (ae_of_all _ hmono)
  rw [integral_sub hiR hiL] at hint
  rw [hingeLoss_eq_standard_integral, hingeLoss_eq_standard_integral]
  exact hint

theorem updateVariance_difference_lower (ni nj T : ℕ)
    (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T) :
    invPower 3 T ≤ (updateVariance (nj : ℝ≥0) : ℝ) -
      (updateVariance (ni : ℝ≥0) : ℝ) := by
  have hjr : 1 ≤ (nj : ℝ) := by exact_mod_cast hj
  have hir : 1 ≤ (ni : ℝ) := by exact_mod_cast (show 1 ≤ ni by omega)
  have hTr : 1 ≤ (T : ℝ) := by exact_mod_cast (show 1 ≤ T by omega)
  have hji : (nj : ℝ) + 1 ≤ (ni : ℝ) := by exact_mod_cast (show nj + 1 ≤ ni by omega)
  have hiTr : (ni : ℝ) + 1 ≤ (T : ℝ) := by exact_mod_cast (show ni + 1 ≤ T by omega)
  have hjTr : (nj : ℝ) ≤ (T : ℝ) := by linarith only [hji, hiTr]
  have hj1Tr : (nj : ℝ) + 1 ≤ (T : ℝ) := by linarith only [hji, hiTr]
  have hnum : (ni : ℝ) ≤ (ni : ℝ) * ((ni : ℝ) + 1) - (nj : ℝ) * ((nj : ℝ) + 1) := by
    have hprod1 := mul_le_mul_of_nonneg_left hji (by linarith : 0 ≤ (nj : ℝ))
    have hprod2 := mul_le_mul_of_nonneg_left (show (nj : ℝ) ≤ (ni : ℝ) by linarith)
      (by linarith : 0 ≤ (ni : ℝ))
    nlinarith only [hprod1, hprod2]
  have hnjprod : (nj : ℝ) * ((nj : ℝ) + 1) ≤ (T : ℝ) ^ 2 := by
    have hp := mul_le_mul hjTr hj1Tr (by linarith : 0 ≤ (nj : ℝ) + 1) (by linarith : 0 ≤ (T : ℝ))
    nlinarith only [hp]
  have hden : ((nj : ℝ) * ((nj : ℝ) + 1)) * ((ni : ℝ) * ((ni : ℝ) + 1)) ≤
      (ni : ℝ) * (T : ℝ) ^ 3 := by
    have hiTproduct : (ni : ℝ) * ((ni : ℝ) + 1) ≤ (ni : ℝ) * (T : ℝ) :=
      mul_le_mul_of_nonneg_left hiTr (by linarith)
    have hp := mul_le_mul hnjprod hiTproduct (by positivity : 0 ≤ (ni : ℝ) * ((ni : ℝ) + 1))
      (by positivity : 0 ≤ (T : ℝ) ^ 2)
    nlinarith only [hp]
  have hscale := mul_le_mul_of_nonneg_right hnum (by positivity : 0 ≤ (T : ℝ) ^ 3)
  simp only [updateVariance, NNReal.coe_inv, NNReal.coe_mul, NNReal.coe_add,
    NNReal.coe_natCast, NNReal.coe_one, invPower]
  have hid : ((nj : ℝ) * ((nj : ℝ) + 1))⁻¹ - ((ni : ℝ) * ((ni : ℝ) + 1))⁻¹ =
      ((ni : ℝ) * ((ni : ℝ) + 1) - (nj : ℝ) * ((nj : ℝ) + 1)) /
        (((nj : ℝ) * ((nj : ℝ) + 1)) * ((ni : ℝ) * ((ni : ℝ) + 1))) := by
    have hjp : (nj : ℝ) ≠ 0 := by linarith
    have hip : (ni : ℝ) ≠ 0 := by linarith
    have hj1p : (nj : ℝ) + 1 ≠ 0 := by linarith
    have hi1p : (ni : ℝ) + 1 ≠ 0 := by linarith
    field_simp
  rw [hid, inv_eq_one_div]
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  linarith only [hden, hscale]

theorem sqrt_variance_difference_ge_half (vL vR : ℝ≥0)
    (hv : vL ≤ vR) (hupper : (vR : ℝ) ≤ 1) :
    ((vR : ℝ) - (vL : ℝ)) / 2 ≤ Real.sqrt (vR : ℝ) - Real.sqrt (vL : ℝ) := by
  have hsL := Real.sqrt_nonneg (vL : ℝ)
  have hsR := Real.sqrt_nonneg (vR : ℝ)
  have hsqL : (Real.sqrt (vL : ℝ)) ^ 2 = (vL : ℝ) := Real.sq_sqrt vL.property
  have hsqR : (Real.sqrt (vR : ℝ)) ^ 2 = (vR : ℝ) := Real.sq_sqrt vR.property
  have hs := Real.sqrt_le_sqrt (show (vL : ℝ) ≤ (vR : ℝ) from hv)
  have hsR1 : Real.sqrt (vR : ℝ) ≤ 1 := Real.sqrt_le_one.mpr hupper
  have hsum : Real.sqrt (vR : ℝ) + Real.sqrt (vL : ℝ) ≤ 2 := by linarith only [hsR1, hs]
  have hprod := mul_le_mul_of_nonneg_right hsum (sub_nonneg.mpr hs)
  nlinarith only [hprod, hsqL, hsqR]

theorem terminal_actionLoss_difference_eq (gap : ℝ) (ni nj : ℝ≥0)
    (hj : 0 < nj) (hi : 0 < ni) :
    expectedLossAfterDraw gap (updateVariance ni) (posteriorVariance (ni + 1) nj) -
        expectedLossAfterDraw gap (updateVariance nj) (posteriorVariance (nj + 1) ni) =
      hingeLoss |gap| (updateVariance nj) - hingeLoss |gap| (updateVariance ni) := by
  have hleft := oneStepImprovement_eq_hinge gap (updateVariance ni) (posteriorVariance (ni + 1) nj)
  have hright := oneStepImprovement_eq_hinge gap (updateVariance nj) (posteriorVariance (nj + 1) ni)
  unfold oneStepImprovement at hleft hright
  rw [← posteriorVariance_after_draw ni nj hi] at hleft
  rw [← posteriorVariance_after_draw nj ni hj, posteriorVariance_comm nj ni] at hright
  linarith only [hleft, hright]

/-- Corrected Lemma 15 has a uniform polynomial lower bound on the actual
action-loss gap, with the larger-count arm i and smaller-count arm j. -/
theorem lemma15_terminal_action_gap_lower (T ni nj : ℕ) (gap : ℝ)
    (hT : 5 ≤ T) (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T)
    (hgap : |gap| ≤ 2 * invPower 2 T) :
    comparisonConstant * invPower 3 T ≤
      expectedLossAfterDraw gap (updateVariance (ni : ℝ≥0))
          (posteriorVariance ((ni : ℝ≥0) + 1) (nj : ℝ≥0)) -
        expectedLossAfterDraw gap (updateVariance (nj : ℝ≥0))
          (posteriorVariance ((nj : ℝ≥0) + 1) (ni : ℝ≥0)) := by
  have hi : 1 ≤ ni := by omega
  have hjT : nj < T := by omega
  have hip : 0 < (ni : ℝ≥0) := by exact_mod_cast (show 0 < ni by omega)
  have hjp : 0 < (nj : ℝ≥0) := by exact_mod_cast (show 0 < nj by omega)
  have hcounts : (nj : ℝ≥0) < (ni : ℝ≥0) := by exact_mod_cast hstrict
  have hv := updateVariance_strictAntitone (nj : ℝ≥0) (ni : ℝ≥0) hjp hcounts
  have hbL := updateVariance_count_bounds ni T hi hiT
  have hbR := updateVariance_count_bounds nj T hj hjT
  have he0 := invPower_nonneg 2 T
  have hequarter : invPower 2 T ≤ 1 / 4 := by
    have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT
    unfold invPower
    rw [inv_eq_one_div]
    apply (div_le_iff₀ (by positivity)).mpr
    nlinarith only [hTr]
  have hcenter : |gap| ≤ Real.sqrt (updateVariance (ni : ℝ≥0) : ℝ) := by
    apply le_trans hgap
    apply Real.le_sqrt_of_sq_le
    nlinarith only [hequarter, he0, hbL.1]
  have hhinge := hingeLoss_difference_ge_standard_interval |gap|
    (updateVariance (ni : ℝ≥0)) (updateVariance (nj : ℝ≥0)) (abs_nonneg _) hcenter hv.le
  have hsig := sqrt_variance_difference_ge_half (updateVariance (ni : ℝ≥0))
    (updateVariance (nj : ℝ≥0)) hv.le (by linarith only [hbR.2])
  have hvar := updateVariance_difference_lower ni nj T hj hstrict hiT
  have hmass : 0 ≤ (gaussianReal 0 1 (Icc (1 : ℝ) 2)).toReal := by positivity
  have hscale := mul_le_mul_of_nonneg_right hsig hmass
  have hscale2 := mul_le_mul_of_nonneg_right hvar (div_nonneg hmass (by norm_num : (0 : ℝ) ≤ 2))
  rw [terminal_actionLoss_difference_eq gap (ni : ℝ≥0) (nj : ℝ≥0) hjp hip]
  unfold comparisonConstant
  nlinarith only [hhinge, hscale, hscale2]

theorem lemma15_terminal_action_gap_conservative (T ni nj : ℕ) (gap : ℝ)
    (hT : 5 ≤ T) (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T)
    (hgap : |gap| ≤ 2 * invPower 2 T) :
    comparisonConstant * invPower 8 T ≤
      expectedLossAfterDraw gap (updateVariance (ni : ℝ≥0))
          (posteriorVariance ((ni : ℝ≥0) + 1) (nj : ℝ≥0)) -
        expectedLossAfterDraw gap (updateVariance (nj : ℝ≥0))
          (posteriorVariance ((nj : ℝ≥0) + 1) (ni : ℝ≥0)) := by
  exact (mul_le_mul_of_nonneg_left
    (invPower_antitone (by norm_num : 3 ≤ 8) (by omega : 1 ≤ T))
    comparisonConstant_positive.le).trans
      (lemma15_terminal_action_gap_lower T ni nj gap hT hj hstrict hiT hgap)

end

end Draft0.TerminalBounds
