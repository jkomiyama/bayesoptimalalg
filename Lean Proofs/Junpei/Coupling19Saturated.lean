import Junpei.Coupling19Rates

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Coupling19

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain
open Draft0.Rates Draft0.TerminalBounds Draft0.PredictiveBound
open Junpei.CommonSample Junpei.Forecast JunpeiComparison

noncomputable section

theorem saturated_left_forced_permutation (q k : ℕ) (c : Counts)
    (hsat : c.n1 = q) (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (true :: schedule leftTie k (leftStep c)).Perm (schedule leftTie k c ++ [true]) := by
  apply permutation_of_finalCounts _ _ c
  simp only [finalCounts, step, ↓reduceIte, schedule_final, leftTie_run, finalCounts_append]
  exact balancedRun_left_commute k c (by omega)

theorem saturated_right_skip_rules_agree (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hsat : c.n2 = q) (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    ruleRun rightTie k c = balancedRun k c := by
  have hl := oriented_skip_left_counts q k c hq hcap1 hcap2 hs
  have hr := equation22_terminal_counts q k c hq hcap1 hcap2 hs
  have hsmall : c.n1 < q := by omega
  have hn3 : (ruleRun rightTie k c).n3 = c.n3 := by
    rw [rightTie_run_swap]
    exact balancedRun_n3 k (swapCounts c)
  apply counts_ext
  · rw [hl.1, hr.1, if_neg (by omega : ¬ c.n1 = q), if_pos hsat]
  · rw [hl.2, hr.2, if_neg (by omega : ¬ c.n1 = q), if_pos hsat]
  · rw [hn3, balancedRun_n3]

theorem saturated_right_forced_permutation (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hsat : c.n2 = q) (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (false :: schedule leftTie k (rightStep c)).Perm (schedule rightTie k c ++ [false]) := by
  apply permutation_of_finalCounts _ _ c
  simp only [finalCounts, step, Bool.false_eq_true, ↓reduceIte,
    schedule_final, leftTie_run, finalCounts_append]
  rw [saturated_right_skip_rules_agree q k c hq hcap1 hcap2 hsat hs]
  exact repaired_right_allocation_plus_one q k c hq hcap1 hcap2 hs

/-- Fubini and allocation permutation identify the actual forced-first risks
with two experiments that have a shared prefix and different final samples. -/
theorem paired_forced_action_gap_identity (r : ℕ → ℕ → Bool) (k : ℕ)
    (c : Counts) (m1 m2 : ℝ) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hleft : (true :: schedule leftTie k (leftStep c)).Perm (schedule r k c ++ [true]))
    (hright : (false :: schedule leftTie k (rightStep c)).Perm (schedule r k c ++ [false])) :
    rightAction k m1 m2 c - leftAction k m1 m2 c =
      ∫ x : ℝ, (hingeLoss |x| (updateVariance (ruleRun r k c).n2) -
        hingeLoss |x| (updateVariance (ruleRun r k c).n1))
        ∂gaussianReal (m1 - m2) (ruleAccumulatedVariance r k c) := by
  have hl := risk_permutation _ _ hleft c m1 m2 h1 h2
  have hr := risk_permutation _ _ hright c m1 m2 h1 h2
  rw [forced_left_risk_eq k c m1 m2 h1 h2] at hl
  rw [forced_right_risk_eq k c m1 m2 h1 h2] at hr
  have hg := common_prefix_last_action_gap (schedule r k c) c m1 m2 h1 h2
  rw [← hl, ← hr, schedule_final, schedule_variance] at hg
  linarith only [hg]

theorem arbitrary_rule_hinge_difference_lower (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (T q t ni nj : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c)
    (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T) :
    (comparisonConstant * lowerConstant) * invPower 178 T ≤
      ∫ x : ℝ, (hingeLoss |x| (updateVariance (nj : ℝ≥0)) -
        hingeLoss |x| (updateVariance (ni : ℝ≥0)))
        ∂gaussianReal (terminalGap p c) (ruleAccumulatedVariance r (T - t) c) := by
  have h5 : 5 ≤ T := by omega
  have hip : 0 < (ni : ℝ≥0) := by exact_mod_cast (show 0 < ni by omega)
  have hjp : 0 < (nj : ℝ≥0) := by exact_mod_cast (show 0 < nj by omega)
  have hv := updateVariance_strictAntitone (nj : ℝ≥0) (ni : ℝ≥0) hjp
    (by exact_mod_cast hstrict)
  have hi : Integrable (fun x : ℝ => hingeLoss |x| (updateVariance (nj : ℝ≥0)) -
      hingeLoss |x| (updateVariance (ni : ℝ≥0)))
      (gaussianReal (terminalGap p c) (ruleAccumulatedVariance r (T - t) c)) :=
    (absolute_hinge_gaussian_integrable _ _ _).sub (absolute_hinge_gaussian_integrable _ _ _)
  have hmass := Junpei.Forecast.lemma18 r hr T q t CU p c hT ht hpre hc hW
  rw [ruleForecast_gaussian] at hmass
  have hband := Draft0.DynamicGain.gaussian_integral_band_lower hi
    (fun x => sub_nonneg.mpr (hingeLoss_strictMono_variance |x| (abs_nonneg _) _ _ hv).le)
    (r := 8) (fun x hx => by
      have hx0 : 0 ≤ x := (invPower_nonneg 2 T).trans hx.1
      have hsmall : |x| ≤ 2 * invPower 2 T := by rw [abs_of_nonneg hx0]; exact hx.2
      have hg := lemma15_terminal_action_gap_conservative T ni nj x h5 hj hstrict hiT hsmall
      rw [terminal_actionLoss_difference_eq x (ni : ℝ≥0) (nj : ℝ≥0) hjp hip] at hg
      exact hg)
  have hp := mul_le_mul_of_nonneg_left hmass
    (mul_nonneg comparisonConstant_positive.le (invPower_nonneg 8 T))
  have he : (comparisonConstant * invPower 8 T) * (lowerConstant * invPower 170 T) =
      (comparisonConstant * lowerConstant) * invPower 178 T := by
    rw [← invPower_product 8 170 T]
    ring
  rw [he] at hp
  exact hp.trans hband

/-- The full saturated-arm Eq. (25), using the literal shared-prefix experiments. -/
theorem lemma19_saturated_left_gap_via_coupling (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts) (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) (hsat : c.n1 = q) :
    (comparisonConstant * lowerConstant) * invPower 178 T ≤
      rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c -
        leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hq : 2 ≤ q := by omega
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    have := hc.2.2.2
    have := hc.2.2.1
    omega
  have hcap1 := hW.2.2.2.2.2.1
  have hcap2 := hW.2.2.2.2.2.2.1
  have hsmall : c.n2 < q := by omega
  have hf := equation22_terminal_counts q (T - t) c (by omega) hcap1 hcap2 hs
  simp only [if_neg (by omega : ¬ c.n2 = q)] at hf
  rw [paired_forced_action_gap_identity leftTie (T - t) c _ _ h1 h2
    (saturated_left_forced_permutation q (T - t) c hsat hs)
    (forced_right_is_common_then_last q (T - t) c (by omega) hcap1 hcap2 hs)]
  simp only [leftTie_run, hf.1, hf.2]
  exact arbitrary_rule_hinge_difference_lower leftTie leftTie_smaller T q t q (q - 1)
    CU p c hT ht hpre hc hW (by omega) (by omega) (by omega)

theorem lemma19_saturated_right_gap_via_coupling (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts) (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) (hsat : c.n2 = q) :
    (comparisonConstant * lowerConstant) * invPower 178 T ≤
      leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c -
        rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hq : 2 ≤ q := by omega
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    have := hc.2.2.2
    have := hc.2.2.1
    omega
  have hcap1 := hW.2.2.2.2.2.1
  have hcap2 := hW.2.2.2.2.2.2.1
  have hsmall : c.n1 < q := by omega
  have hf := oriented_skip_left_counts q (T - t) c (by omega) hcap1 hcap2 hs
  simp only [if_neg (by omega : ¬ c.n1 = q)] at hf
  have hgap := paired_forced_action_gap_identity rightTie (T - t) c (p 1 c.n1) (p 2 c.n2) h1 h2
    (forced_left_is_common_then_last q (T - t) c (by omega) hcap1 hcap2 hs)
    (saturated_right_forced_permutation q (T - t) c (by omega) hcap1 hcap2 hsat hs)
  rw [hf.1, hf.2] at hgap
  rw [integral_sub (absolute_hinge_gaussian_integrable _ _ _)
    (absolute_hinge_gaussian_integrable _ _ _)] at hgap
  have hbound := arbitrary_rule_hinge_difference_lower rightTie rightTie_smaller T q t q (q - 1)
    CU p c hT ht hpre hc hW (by omega) (by omega) (by omega)
  rw [integral_sub (absolute_hinge_gaussian_integrable _ _ _)
    (absolute_hinge_gaussian_integrable _ _ _)] at hbound
  unfold terminalGap at hbound
  linarith only [hgap, hbound]

end
end Junpei.Coupling19
