import Junpei.CountComparison
import Draft0.DynamicGain

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Dynamic

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain
open Draft0.Rates Draft0.TerminalBounds Draft0.PredictiveBound Draft0.PredictiveLaw
open JunpeiComparison

noncomputable section

theorem nonterminal_hinge_gain_lower (T q t n : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c)
    (hn : 1 ≤ n) (hnT : n < T) :
    (gainConstant * lowerConstant) * invPower 176 T ≤
      ∫ x : ℝ, hingeLoss |x| (updateVariance (n : ℝ≥0))
        ∂balancedPredictiveLaw (T - t) c (terminalGap p c) := by
  have h5 : 5 ≤ T := by omega
  have hmass := new_lemma18_fixed_tie170 T q t CU p c hT ht hpre hc hW
  rw [balancedPredictiveLaw_gaussian] at hmass ⊢
  have hband := Draft0.DynamicGain.gaussian_integral_band_lower
    (absolute_hinge_gaussian_integrable (terminalGap p c) (accumulatedVariance (T - t) c)
      (updateVariance (n : NNReal)))
    (fun x => hingeLoss_nonnegative |x| _) (r := 6)
    (fun x hx => by
      have hx0 : 0 ≤ x := (invPower_nonneg 2 T).trans hx.1
      have hsmall : |x| ≤ 2 * invPower 2 T := by rw [abs_of_nonneg hx0]; exact hx.2
      have hg := lemma14_sample_count_lower T n x 0 h5 hn hnT hsmall
      rw [oneStepImprovement_eq_hinge] at hg
      exact hg)
  have hp := mul_le_mul_of_nonneg_left hmass
    (mul_nonneg gainConstant_positive.le (invPower_nonneg 6 T))
  have he : (gainConstant * invPower 6 T) * (lowerConstant * invPower 170 T) =
      (gainConstant * lowerConstant) * invPower 176 T := by
    rw [← invPower_product 6 170 T]; ring
  rw [he] at hp
  exact hp.trans hband

theorem nonterminal_hinge_difference_lower (T q t ni nj : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c)
    (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T) :
    (comparisonConstant * lowerConstant) * invPower 178 T ≤
      ∫ x : ℝ, (hingeLoss |x| (updateVariance (nj : ℝ≥0)) -
        hingeLoss |x| (updateVariance (ni : ℝ≥0)))
        ∂balancedPredictiveLaw (T - t) c (terminalGap p c) := by
  have h5 : 5 ≤ T := by omega
  have hip : 0 < (ni : ℝ≥0) := by exact_mod_cast (show 0 < ni by omega)
  have hjp : 0 < (nj : ℝ≥0) := by exact_mod_cast (show 0 < nj by omega)
  have hv := updateVariance_strictAntitone (nj : ℝ≥0) (ni : ℝ≥0) hjp
    (by exact_mod_cast hstrict)
  have hi : Integrable (fun x : ℝ => hingeLoss |x| (updateVariance (nj : ℝ≥0)) -
      hingeLoss |x| (updateVariance (ni : ℝ≥0)))
      (balancedPredictiveLaw (T - t) c (terminalGap p c)) := by
    rw [balancedPredictiveLaw_gaussian]
    exact (absolute_hinge_gaussian_integrable _ _ _).sub
      (absolute_hinge_gaussian_integrable _ _ _)
  have hmass := new_lemma18_fixed_tie170 T q t CU p c hT ht hpre hc hW
  rw [balancedPredictiveLaw_gaussian] at hmass ⊢
  rw [balancedPredictiveLaw_gaussian] at hi
  have hband := Draft0.DynamicGain.gaussian_integral_band_lower hi
    (fun x => sub_nonneg.mpr (hingeLoss_strictMono_variance |x| (abs_nonneg _) _ _ hv).le)
    (r := 8) (fun x hx => by
      have hx0 : 0 ≤ x := (invPower_nonneg 2 T).trans hx.1
      have hsmall : |x| ≤ 2 * invPower 2 T := by rw [abs_of_nonneg hx0]; exact hx.2
      have hg := lemma15_terminal_action_gap_conservative T ni nj x h5 hj hstrict hiT hsmall
      rw [terminal_actionLoss_difference_eq x (ni : NNReal) (nj : NNReal) hjp hip] at hg
      exact hg)
  have hp := mul_le_mul_of_nonneg_left hmass
    (mul_nonneg comparisonConstant_positive.le (invPower_nonneg 8 T))
  have he : (comparisonConstant * invPower 8 T) * (lowerConstant * invPower 170 T) =
      (comparisonConstant * lowerConstant) * invPower 178 T := by
    rw [← invPower_product 8 170 T]; ring
  rw [he] at hp
  exact hp.trans hband

theorem preterminal_variance (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    Draft0.TwoArmBellman.variance (balancedRun k c) = posteriorVariance (q : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0) := by
  have hf := equation22_terminal_counts q k c hq hcap1 hcap2 hs
  unfold Draft0.TwoArmBellman.variance
  by_cases h : c.n2 = q
  · simp only [if_pos h] at hf
    rw [hf.1, hf.2, posteriorVariance_comm]
  · simp only [if_neg h] at hf
    rw [hf.1, hf.2]

theorem forced_left_final_counts (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    ((balancedRun k (leftStep c)).n1 = if c.n1 = q then q + 1 else q) ∧
    ((balancedRun k (leftStep c)).n2 = if c.n1 = q then q - 1 else q) := by
  by_cases h : c.n1 = q
  · have hr : (leftStep c).n2 + k ≤ (leftStep c).n1 := by simp only [leftStep]; omega
    rw [balancedRun_right_only k (leftStep c) hr]
    simp only [leftStep, if_pos h]
    constructor <;> omega
  · have h1 : (leftStep c).n1 ≤ q := by simp only [leftStep]; omega
    have h2 : (leftStep c).n2 ≤ q := hcap2
    have hsum : (leftStep c).n1 + (leftStep c).n2 + k = 2 * q := by simp only [leftStep]; omega
    simpa only [if_neg h] using balancedRun_final_counts q k (leftStep c) h1 h2 hsum

theorem forced_right_final_counts (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    ((balancedRun k (rightStep c)).n1 = if c.n2 = q then q - 1 else q) ∧
    ((balancedRun k (rightStep c)).n2 = if c.n2 = q then q + 1 else q) := by
  by_cases h : c.n2 = q
  · have hl : (rightStep c).n1 + k ≤ (rightStep c).n2 := by simp only [rightStep]; omega
    rw [balancedRun_left_only k (rightStep c) hl]
    simp only [rightStep, if_pos h]
    constructor <;> omega
  · have h1 : (rightStep c).n1 ≤ q := hcap1
    have h2 : (rightStep c).n2 ≤ q := by simp only [rightStep]; omega
    have hsum : (rightStep c).n1 + (rightStep c).n2 + k = 2 * q := by simp only [rightStep]; omega
    simpa only [if_neg h] using balancedRun_final_counts q k (rightStep c) h1 h2 hsum

theorem forced_left_variance_lower (q k : ℕ) (c : Counts) (hq : 2 ≤ q)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    cumulativeVariance k c + updateVariance (q : ℝ≥0) ≤
      updateVariance c.n1 + cumulativeVariance k (leftStep c) := by
  have hp := preterminal_variance q k c (by omega) hcap1 hcap2 hs
  have hf := forced_left_final_counts q k c (by omega) hcap1 hcap2 hs
  have ha := leftAction_variance_decomposition k c h1 h2
  have hb := cumulativeVariance_decomposition k c h1 h2
  by_cases h : c.n1 = q
  · simp only [if_pos h] at hf
    rw [Draft0.TwoArmBellman.variance, hf.1, hf.2] at ha
    rw [hp] at hb
    have hv := posteriorVariance_after_draw (q : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0)
      (by exact_mod_cast (show 0 < q by omega))
    have hv' : posteriorVariance (q : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0) =
        updateVariance (q : ℝ≥0) +
          posteriorVariance ((q + 1 : ℕ) : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0) := by
      simpa only [Nat.cast_add, Nat.cast_one] using hv
    rw [hv', ← add_assoc] at hb
    exact (add_right_cancel (ha.trans hb.symm)).symm.le
  · simp only [if_neg h] at hf
    rw [Draft0.TwoArmBellman.variance, hf.1, hf.2] at ha
    rw [hp] at hb
    have hv := posteriorVariance_after_draw ((q - 1 : ℕ) : ℝ≥0) (q : ℝ≥0)
      (by exact_mod_cast (show 0 < q - 1 by omega))
    have hvs := updateVariance_strictAntitone ((q - 1 : ℕ) : ℝ≥0) (q : ℝ≥0)
      (by exact_mod_cast (show 0 < q - 1 by omega))
      (by exact_mod_cast (show q - 1 < q by omega))
    have he : updateVariance c.n1 + cumulativeVariance k (leftStep c) =
        cumulativeVariance k c + updateVariance ((q - 1 : ℕ) : ℝ≥0) := by
      have hqa : (((q - 1 : ℕ) : ℝ≥0) + 1) = (q : ℝ≥0) := by
        exact_mod_cast (show q - 1 + 1 = q by omega)
      rw [hqa] at hv
      rw [posteriorVariance_comm (q : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0), hv, ← add_assoc] at hb
      exact add_right_cancel (ha.trans hb.symm)
    rw [he]
    exact add_le_add_left hvs.le _

theorem forced_right_variance_lower (q k : ℕ) (c : Counts) (hq : 2 ≤ q)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    cumulativeVariance k c + updateVariance (q : ℝ≥0) ≤
      updateVariance c.n2 + cumulativeVariance k (rightStep c) := by
  have hp := preterminal_variance q k c (by omega) hcap1 hcap2 hs
  have hf := forced_right_final_counts q k c (by omega) hcap1 hcap2 hs
  have ha := rightAction_variance_decomposition k c h1 h2
  have hb := cumulativeVariance_decomposition k c h1 h2
  by_cases h : c.n2 = q
  · simp only [if_pos h] at hf
    rw [Draft0.TwoArmBellman.variance, hf.1, hf.2, posteriorVariance_comm] at ha
    rw [hp] at hb
    have hv := posteriorVariance_after_draw (q : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0)
      (by exact_mod_cast (show 0 < q by omega))
    have hv' : posteriorVariance (q : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0) =
        updateVariance (q : ℝ≥0) +
          posteriorVariance ((q + 1 : ℕ) : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0) := by
      simpa only [Nat.cast_add, Nat.cast_one] using hv
    rw [hv', ← add_assoc] at hb
    exact (add_right_cancel (ha.trans hb.symm)).symm.le
  · simp only [if_neg h] at hf
    rw [Draft0.TwoArmBellman.variance, hf.1, hf.2] at ha
    rw [hp] at hb
    have hv := posteriorVariance_after_draw ((q - 1 : ℕ) : ℝ≥0) (q : ℝ≥0)
      (by exact_mod_cast (show 0 < q - 1 by omega))
    have hvs := updateVariance_strictAntitone ((q - 1 : ℕ) : ℝ≥0) (q : ℝ≥0)
      (by exact_mod_cast (show 0 < q - 1 by omega))
      (by exact_mod_cast (show q - 1 < q by omega))
    have he : updateVariance c.n2 + cumulativeVariance k (rightStep c) =
        cumulativeVariance k c + updateVariance ((q - 1 : ℕ) : ℝ≥0) := by
      have hqa : (((q - 1 : ℕ) : ℝ≥0) + 1) = (q : ℝ≥0) := by
        exact_mod_cast (show q - 1 + 1 = q by omega)
      rw [hqa] at hv
      rw [posteriorVariance_comm (q : ℝ≥0) ((q - 1 : ℕ) : ℝ≥0), hv, ← add_assoc] at hb
      exact add_right_cancel (ha.trans hb.symm)
    rw [he]
    exact add_le_add_left hvs.le _

theorem left_gain_integral_lower (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 2 ≤ q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (∫ x : ℝ, hingeLoss |x| (updateVariance (q : ℝ≥0))
      ∂balancedPredictiveLaw k c (m1 - m2)) ≤
      leftAction k m1 m2 c - payoff k m1 m2 c := by
  have hv := forced_left_variance_lower q k c hq h1 h2 hcap1 hcap2 hs
  have hm := hingeLoss_mono_variance (m2 - m1) _ _ hv
  rw [leftAction_eq, payoff_eq]
  have he := hinge_variance_increment_integral (m2 - m1)
    (cumulativeVariance k c) (updateVariance (q : ℝ≥0))
  rw [cumulativeVariance_eq_accumulatedVariance k c h1 h2] at he
  rw [balancedPredictiveLaw_gaussian]
  have hneg : -(m2 - m1) = m1 - m2 := by ring
  rw [hneg] at he
  rw [cumulativeVariance_eq_accumulatedVariance k c h1 h2] at hm ⊢
  linarith only [he, hm]

theorem right_gain_integral_lower (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 2 ≤ q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (∫ x : ℝ, hingeLoss |x| (updateVariance (q : ℝ≥0))
      ∂balancedPredictiveLaw k c (m1 - m2)) ≤
      rightAction k m1 m2 c - payoff k m1 m2 c := by
  have hv := forced_right_variance_lower q k c hq h1 h2 hcap1 hcap2 hs
  have hm := hingeLoss_mono_variance (m2 - m1) _ _ hv
  rw [rightAction_eq, payoff_eq]
  have he := hinge_variance_increment_integral (m2 - m1)
    (cumulativeVariance k c) (updateVariance (q : ℝ≥0))
  rw [cumulativeVariance_eq_accumulatedVariance k c h1 h2] at he
  rw [balancedPredictiveLaw_gaussian]
  have hneg : -(m2 - m1) = m1 - m2 := by ring
  rw [hneg] at he
  rw [cumulativeVariance_eq_accumulatedVariance k c h1 h2] at hm ⊢
  linarith only [he, hm]

theorem lemma19_individual_gains (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) :
    (gainConstant * lowerConstant) * invPower 176 T ≤
      leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c -
        payoff (T - t) (p 1 c.n1) (p 2 c.n2) c ∧
    (gainConstant * lowerConstant) * invPower 176 T ≤
      rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c -
        payoff (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hq : 2 ≤ q := by omega
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    have := hc.2.2.2; have := hc.2.2.1; omega
  have hcap1 := hW.2.2.2.2.2.1
  have hcap2 := hW.2.2.2.2.2.2.1
  have hg := nonterminal_hinge_gain_lower T q t q CU p c hT ht hpre hc hW
    (by omega) (by omega)
  exact ⟨hg.trans (left_gain_integral_lower q (T - t) _ _ c hq h1 h2 hcap1 hcap2 hs),
    hg.trans (right_gain_integral_lower q (T - t) _ _ c hq h1 h2 hcap1 hcap2 hs)⟩

theorem lemma19_saturated_left_gap (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) (hsat : c.n1 = q) :
    (comparisonConstant * lowerConstant) * invPower 178 T ≤
      rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c -
        leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hq : 2 ≤ q := by omega
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    have := hc.2.2.2; have := hc.2.2.1; omega
  have hsmall : c.n2 < c.n1 := by omega
  rw [← smaller_count_right_optimal (T - t) _ _ c h1 h2 hsmall,
    saturated_left_gap_preterminal q (T - t) _ _ c (by omega) h1 h2 hsat hs]
  exact nonterminal_hinge_difference_lower T q t q (q - 1) CU p c hT ht hpre hc hW
    (by omega) (by omega) (by omega)

theorem lemma19_saturated_right_gap (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) (hsat : c.n2 = q) :
    (comparisonConstant * lowerConstant) * invPower 178 T ≤
      leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c -
        rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hq : 2 ≤ q := by omega
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    have := hc.2.2.2; have := hc.2.2.1; omega
  have hsmall : c.n1 ≤ c.n2 := by omega
  rw [← smaller_count_left_optimal (T - t) _ _ c h1 h2 hsmall,
    saturated_right_gap_preterminal q (T - t) _ _ c (by omega) h1 h2 hsat hs]
  exact nonterminal_hinge_difference_lower T q t q (q - 1) CU p c hT ht hpre hc hW
    (by omega) (by omega) (by omega)

end
end Junpei.Dynamic
