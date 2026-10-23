import Draft0.ThreeArmComparison
import Draft0.TerminalBounds
import Draft0.PredictiveBound
import Draft0.CorrectedLemma17
import Draft0.TwoArmGain

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Draft0.DynamicGain

open Draft0.Gaussian Draft0.Rates Draft0.TerminalBounds Draft0.PredictiveBound

noncomputable section

theorem gaussian_integral_band_lower {mean : ℝ} {v : ℝ≥0} {T r : ℕ} {c : ℝ}
    {f : ℝ → ℝ} (hf : Integrable f (gaussianReal mean v))
    (hf0 : ∀ x, 0 ≤ f x)
    (hband : ∀ x ∈ Icc (invPower 2 T) (2 * invPower 2 T), c * invPower r T ≤ f x) :
    (c * invPower r T) *
      (gaussianReal mean v (Icc (invPower 2 T) (2 * invPower 2 T))).toReal ≤
        ∫ x : ℝ, f x ∂gaussianReal mean v := by
  have hi : Integrable
      ((Icc (invPower 2 T) (2 * invPower 2 T)).indicator (fun _ : ℝ => c * invPower r T))
      (gaussianReal mean v) := (integrable_const _).indicator measurableSet_Icc
  have hm := integral_mono hi hf (fun x => by
    by_cases hx : x ∈ Icc (invPower 2 T) (2 * invPower 2 T)
    · rw [indicator_of_mem hx]
      exact hband x hx
    · rw [indicator_of_notMem hx]
      exact hf0 x)
  rw [integral_indicator_const _ measurableSet_Icc, smul_eq_mul] at hm
  simpa only [measureReal_def, mul_comm] using hm

theorem nonterminal_hinge_gain_lower (T n : ℕ) (δ : ℝ) (a : ℝ≥0)
    (hT : 5 ≤ T) (hn : 1 ≤ n) (hnT : n < T)
    (ha : 0 < a) (haupper : (a : ℝ) ≤ 2) (halower : invPower 2 T ≤ (a : ℝ))
    (hδ : δ ^ 2 ≤ 204 * Real.log (T : ℝ) * (a : ℝ)) :
    (gainConstant * lowerConstant) * invPower 212 T ≤
      ∫ x : ℝ, hingeLoss |x| (updateVariance (n : ℝ≥0)) ∂gaussianReal δ a := by
  have hmass := gaussian_interval_probability_drift204 T hT ha haupper halower hδ
  have hband := gaussian_integral_band_lower
    (Draft0.TwoArmGain.absolute_hinge_gaussian_integrable δ a (updateVariance (n : ℝ≥0)))
    (fun x => hingeLoss_nonnegative |x| _) (r := 6)
    (fun x hx => by
      have hx0 : 0 ≤ x := (invPower_nonneg 2 T).trans hx.1
      have hsmall : |x| ≤ 2 * invPower 2 T := by rw [abs_of_nonneg hx0]; exact hx.2
      have hg := lemma14_sample_count_lower T n x 0 hT hn hnT hsmall
      rw [oneStepImprovement_eq_hinge] at hg
      exact hg)
  have hp := mul_le_mul_of_nonneg_left hmass
    (mul_nonneg gainConstant_positive.le (invPower_nonneg 6 T))
  have he : (gainConstant * invPower 6 T) * (lowerConstant * invPower 206 T) =
      (gainConstant * lowerConstant) * invPower 212 T := by
    rw [← invPower_product 6 206 T]
    ring
  rw [he] at hp
  exact hp.trans hband

theorem nonterminal_hinge_difference_lower (T ni nj : ℕ) (δ : ℝ) (a : ℝ≥0)
    (hT : 5 ≤ T) (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T)
    (ha : 0 < a) (haupper : (a : ℝ) ≤ 2) (halower : invPower 2 T ≤ (a : ℝ))
    (hδ : δ ^ 2 ≤ 204 * Real.log (T : ℝ) * (a : ℝ)) :
    (comparisonConstant * lowerConstant) * invPower 214 T ≤
      ∫ x : ℝ,
        hingeLoss |x| (updateVariance (nj : ℝ≥0)) -
        hingeLoss |x| (updateVariance (ni : ℝ≥0)) ∂gaussianReal δ a := by
  have hip : 0 < (ni : ℝ≥0) := by exact_mod_cast (show 0 < ni by omega)
  have hjp : 0 < (nj : ℝ≥0) := by exact_mod_cast (show 0 < nj by omega)
  have hv := updateVariance_strictAntitone (nj : ℝ≥0) (ni : ℝ≥0) hjp
    (by exact_mod_cast hstrict)
  have hi := (Draft0.TwoArmGain.absolute_hinge_gaussian_integrable δ a
    (updateVariance (nj : ℝ≥0))).sub
    (Draft0.TwoArmGain.absolute_hinge_gaussian_integrable δ a (updateVariance (ni : ℝ≥0)))
  have hmass := gaussian_interval_probability_drift204 T hT ha haupper halower hδ
  have hband := gaussian_integral_band_lower hi
    (fun x => sub_nonneg.mpr (hingeLoss_strictMono_variance |x| (abs_nonneg _) _ _ hv).le)
    (r := 8) (fun x hx => by
      have hx0 : 0 ≤ x := (invPower_nonneg 2 T).trans hx.1
      have hsmall : |x| ≤ 2 * invPower 2 T := by rw [abs_of_nonneg hx0]; exact hx.2
      have hg := lemma15_terminal_action_gap_conservative T ni nj x hT hj hstrict hiT hsmall
      rw [terminal_actionLoss_difference_eq x (ni : ℝ≥0) (nj : ℝ≥0) hjp hip] at hg
      exact hg)
  have hp := mul_le_mul_of_nonneg_left hmass
    (mul_nonneg comparisonConstant_positive.le (invPower_nonneg 8 T))
  have he : (comparisonConstant * invPower 8 T) * (lowerConstant * invPower 206 T) =
      (comparisonConstant * lowerConstant) * invPower 214 T := by
    rw [← invPower_product 8 206 T]
    ring
  rw [he] at hp
  exact hp.trans hband

/-- A terminal gap of order sqrt(log T) standard deviations still gives a polynomial gain. -/
theorem terminal_hinge_gain_lower (T : ℕ) (gap : ℝ) (v : ℝ≥0)
    (hT : 5 ≤ T) (hv : 0 < v) (hvupper : (v : ℝ) ≤ 2)
    (hvlower : invPower 2 T ≤ (v : ℝ))
    (hgap : gap ^ 2 ≤ 204 * Real.log (T : ℝ) * (v : ℝ)) :
    lowerConstant * invPower 208 T ≤ hingeLoss |gap| v := by
  have hmean : (-|gap|) ^ 2 ≤ 204 * Real.log (T : ℝ) * (v : ℝ) := by
    simpa only [neg_sq, sq_abs] using hgap
  have hmass := gaussian_interval_probability_drift204 T hT hv hvupper hvlower hmean
  have hband := gaussian_integral_band_lower
    ((gaussian_id_integrable (-|gap|) v).pos_part)
    (fun x => le_max_right x 0) (c := 1) (T := T) (r := 2)
    (fun x hx => by simpa only [one_mul] using hx.1.trans (le_max_left x 0))
  have hp := mul_le_mul_of_nonneg_left hmass (invPower_nonneg 2 T)
  have he : invPower 2 T * (lowerConstant * invPower 206 T) =
      lowerConstant * invPower 208 T := by
    rw [← invPower_product 2 206 T]
    ring
  rw [he] at hp
  simp only [one_mul] at hband
  have h := hp.trans hband
  rw [gaussian_positivePart_integral] at h
  simpa only [neg_neg] using h

theorem terminal_hinge_difference_lower (T ni nj : ℕ) (gap : ℝ)
    (hT : 5 ≤ T) (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T)
    (hgap : gap ^ 2 ≤ 204 * Real.log (T : ℝ) * (updateVariance (ni : ℝ≥0) : ℝ)) :
    (lowerConstant / 2) * invPower 211 T ≤
      hingeLoss |gap| (updateVariance (nj : ℝ≥0)) -
        hingeLoss |gap| (updateVariance (ni : ℝ≥0)) := by
  let vL := updateVariance (ni : ℝ≥0)
  let vR := updateVariance (nj : ℝ≥0)
  let sL := Real.sqrt (vL : ℝ)
  let sR := Real.sqrt (vR : ℝ)
  let a := |gap| / sL
  have hjp : 0 < (nj : ℝ≥0) := by exact_mod_cast (show 0 < nj by omega)
  have hip : 0 < (ni : ℝ≥0) := by exact_mod_cast (show 0 < ni by omega)
  have hvL : 0 < vL := by dsimp [vL, updateVariance]; positivity
  have hsL : 0 < sL := Real.sqrt_pos.mpr hvL
  have hsR : 0 ≤ sR := Real.sqrt_nonneg _
  have ha0 : 0 ≤ a := div_nonneg (abs_nonneg _) hsL.le
  have haeq : sL * a = |gap| := by dsimp [a]; field_simp
  have hvar : vL < vR := updateVariance_strictAntitone (nj : ℝ≥0) (ni : ℝ≥0)
    hjp (by exact_mod_cast hstrict)
  have hs : sL ≤ sR := Real.sqrt_le_sqrt hvar.le
  have hbound := updateVariance_count_bounds nj T hj (by omega)
  have hsig := sqrt_variance_difference_ge_half vL vR hvar.le (by
    dsimp [vR]; linarith only [hbound.2])
  have hdiff := updateVariance_difference_lower ni nj T hj hstrict hiT
  have hsiglower : (1 / 2 : ℝ) * invPower 3 T ≤ sR - sL := by
    dsimp [sL, sR, vL, vR] at *
    linarith only [hsig, hdiff]
  have haLog : (-a) ^ 2 ≤ 204 * Real.log (T : ℝ) := by
    dsimp [a, sL]
    rw [neg_sq, div_pow, sq_abs, Real.sq_sqrt (show 0 ≤ (vL : ℝ) from hvL.le)]
    exact (div_le_iff₀ (show 0 < (vL : ℝ) from hvL)).mpr hgap
  have hmass := gaussian_interval_probability_drift204 T (δ := -a) (v := 1) hT
    (by norm_num) (by norm_num) (by
      simpa [invPower] using invPower_antitone (show 0 ≤ 2 by omega) (by omega : 1 ≤ T))
    (by simpa only [NNReal.coe_one, mul_one] using haLog)
  let f := fun y : ℝ => max (sR * (y + a) - |gap|) 0 - max (sL * (y + a) - |gap|) 0
  have hiR : Integrable (fun y : ℝ => max (sR * (y + a) - |gap|) 0)
      (gaussianReal (-a) 1) :=
    ((((gaussian_id_integrable (-a) 1).add (integrable_const a)).const_mul sR).sub
      (integrable_const |gap|)).pos_part
  have hiL : Integrable (fun y : ℝ => max (sL * (y + a) - |gap|) 0)
      (gaussianReal (-a) 1) :=
    ((((gaussian_id_integrable (-a) 1).add (integrable_const a)).const_mul sL).sub
      (integrable_const |gap|)).pos_part
  have hf0 : ∀ y, 0 ≤ f y := by
    intro y
    dsimp [f]
    apply sub_nonneg.mpr
    by_cases hy : 0 ≤ y + a
    · exact max_le_max (sub_le_sub_right (mul_le_mul_of_nonneg_right hs hy) _) le_rfl
    · have hy' : y + a ≤ 0 := le_of_not_ge hy
      rw [max_eq_right (by nlinarith [mul_nonpos_of_nonneg_of_nonpos hsR hy', abs_nonneg gap]),
        max_eq_right (by nlinarith [mul_nonpos_of_nonneg_of_nonpos hsL.le hy', abs_nonneg gap])]
  have hband := gaussian_integral_band_lower (hiR.sub hiL) hf0 (c := 1 / 2) (T := T) (r := 5)
    (fun y hy => by
      have hy0 : 0 ≤ y := (invPower_nonneg 2 T).trans hy.1
      have hx : a ≤ y + a := by linarith
      have hlinL : 0 ≤ sL * (y + a) - |gap| := by nlinarith [mul_nonneg hsL.le hy0]
      have hlinR : 0 ≤ sR * (y + a) - |gap| := by
        have hmul := mul_le_mul_of_nonneg_right hs (show 0 ≤ y + a by positivity)
        linarith
      dsimp [f]
      rw [max_eq_left hlinL, max_eq_left hlinR]
      have hp1 := mul_le_mul_of_nonneg_right hsiglower (show 0 ≤ y + a by positivity)
      have hp2 := mul_le_mul_of_nonneg_left (hy.1.trans (by linarith : y ≤ y + a))
        (mul_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2) (invPower_nonneg 3 T))
      have he : (1 / 2 : ℝ) * invPower 5 T =
          ((1 / 2 : ℝ) * invPower 3 T) * invPower 2 T := by
        rw [← invPower_product 3 2 T]
        ring
      rw [he]
      nlinarith only [hp1, hp2])
  have hp := mul_le_mul_of_nonneg_left hmass
    (mul_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2) (invPower_nonneg 5 T))
  have he : ((1 / 2 : ℝ) * invPower 5 T) * (lowerConstant * invPower 206 T) =
      (lowerConstant / 2) * invPower 211 T := by
    rw [← invPower_product 5 206 T]
    ring
  rw [he] at hp
  have hlow := hp.trans hband
  change (lowerConstant / 2) * invPower 211 T ≤ ∫ y : ℝ, f y ∂gaussianReal (-a) 1 at hlow
  have hmap : (gaussianReal 0 1).map (fun x : ℝ => x - a) = gaussianReal (-a) 1 := by
    simpa using gaussianReal_map_sub_const (μ := (0 : ℝ)) (v := 1) a
  have heq : (∫ y : ℝ, f y ∂gaussianReal (-a) 1) =
      hingeLoss |gap| vR - hingeLoss |gap| vL := by
    rw [← hmap, integral_map (by fun_prop) (by fun_prop)]
    simp only [f, sub_add_cancel]
    have hiR0 : Integrable (fun x : ℝ => max (sR * x - |gap|) 0) (gaussianReal 0 1) := by
      simpa only [Pi.sub_apply] using
        ((((gaussian_id_integrable 0 1).const_mul sR).sub (integrable_const |gap|)).pos_part)
    have hiL0 : Integrable (fun x : ℝ => max (sL * x - |gap|) 0) (gaussianReal 0 1) := by
      simpa only [Pi.sub_apply] using
        ((((gaussian_id_integrable 0 1).const_mul sL).sub (integrable_const |gap|)).pos_part)
    rw [integral_sub hiR0 hiL0,
      hingeLoss_eq_standard_integral, hingeLoss_eq_standard_integral]
  rw [heq] at hlow
  exact hlow

end

end Draft0.DynamicGain
