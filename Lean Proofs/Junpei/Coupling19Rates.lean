import Junpei.Coupling19

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Coupling19

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain
open Draft0.Rates Draft0.TerminalBounds Draft0.PredictiveBound
open Junpei.CommonSample Junpei.Forecast JunpeiComparison

noncomputable section

theorem schedule_variance (r : ℕ → ℕ → Bool) (k : ℕ) (c : Counts) :
    allocationVariance (schedule r k c) c = ruleAccumulatedVariance r k c := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
    simp only [schedule, allocationVariance, ruleAccumulatedVariance]
    have hs : step (r c.n1 c.n2) c = ruleStep r c := by
      unfold step ruleStep
      split <;> rfl
    rw [hs, ih]
    rfl

theorem skip_risk_eq (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (k : ℕ) (c : Counts) (m1 m2 : ℝ) (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    risk (schedule r k c) c m1 m2 = potential c m1 m2 - payoff k m1 m2 c := by
  rw [risk_eq _ _ _ _ h1 h2, schedule_optimal r hr k c m1 m2 h1 h2]

theorem forced_left_risk_eq (k : ℕ) (c : Counts) (m1 m2 : ℝ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    risk (true :: schedule leftTie k (leftStep c)) c m1 m2 =
      potential c m1 m2 - leftAction k m1 m2 c := by
  rw [risk_eq _ _ _ _ h1 h2]
  simp only [utility, ↓reduceIte]
  simp_rw [schedule_optimal leftTie leftTie_smaller k (leftStep c) _ _
    (by simp [leftStep]) (by simpa [leftStep] using h2)]
  rfl

theorem forced_right_risk_eq (k : ℕ) (c : Counts) (m1 m2 : ℝ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    risk (false :: schedule leftTie k (rightStep c)) c m1 m2 =
      potential c m1 m2 - rightAction k m1 m2 c := by
  rw [risk_eq _ _ _ _ h1 h2]
  simp only [utility, Bool.false_eq_true, ↓reduceIte]
  simp_rw [schedule_optimal leftTie leftTie_smaller k (rightStep c) _ _
    (by simpa [rightStep] using h1) (by simp [rightStep])]
  rfl

/-- Repaired Eq. (24) is an exact common-sample coupling identity for arm 1. -/
theorem left_gain_common_sample_identity (q k : ℕ) (c : Counts) (m1 m2 : ℝ)
    (hq : 0 < q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    leftAction k m1 m2 c - payoff k m1 m2 c =
      ∫ x : ℝ, hingeLoss |x| (updateVariance (ruleRun rightTie k c).n1)
        ∂gaussianReal (m1 - m2) (ruleAccumulatedVariance rightTie k c) := by
  have hcoup := risk_permutation _ _
    (forced_left_is_common_then_last q k c hq hcap1 hcap2 hs) c m1 m2 h1 h2
  rw [forced_left_risk_eq k c m1 m2 h1 h2] at hcoup
  have hg := common_prefix_last_gain (schedule rightTie k c) true c m1 m2 h1 h2
  rw [skip_risk_eq rightTie rightTie_smaller k c m1 m2 h1 h2, ← hcoup,
    schedule_final, schedule_variance] at hg
  simp only [sampled, ↓reduceIte] at hg
  linarith only [hg]

theorem right_gain_common_sample_identity (q k : ℕ) (c : Counts) (m1 m2 : ℝ)
    (hq : 0 < q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    rightAction k m1 m2 c - payoff k m1 m2 c =
      ∫ x : ℝ, hingeLoss |x| (updateVariance (ruleRun leftTie k c).n2)
        ∂gaussianReal (m1 - m2) (ruleAccumulatedVariance leftTie k c) := by
  have hcoup := risk_permutation _ _
    (forced_right_is_common_then_last q k c hq hcap1 hcap2 hs) c m1 m2 h1 h2
  rw [forced_right_risk_eq k c m1 m2 h1 h2] at hcoup
  have hg := common_prefix_last_gain (schedule leftTie k c) false c m1 m2 h1 h2
  rw [skip_risk_eq leftTie leftTie_smaller k c m1 m2 h1 h2, ← hcoup,
    schedule_final, schedule_variance] at hg
  simp only [sampled, Bool.false_eq_true, ↓reduceIte] at hg
  linarith only [hg]

theorem arbitrary_rule_hinge_gain_lower (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (T q t n : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) (hn : 1 ≤ n) (hnT : n < T) :
    (gainConstant * lowerConstant) * invPower 176 T ≤
      ∫ x : ℝ, hingeLoss |x| (updateVariance (n : ℝ≥0))
        ∂gaussianReal (terminalGap p c) (ruleAccumulatedVariance r (T - t) c) := by
  have h5 : 5 ≤ T := by omega
  have hmass := Junpei.Forecast.lemma18 r hr T q t CU p c hT ht hpre hc hW
  rw [ruleForecast_gaussian] at hmass
  have hband := Draft0.DynamicGain.gaussian_integral_band_lower
    (absolute_hinge_gaussian_integrable (terminalGap p c) (ruleAccumulatedVariance r (T - t) c)
      (updateVariance (n : ℝ≥0))) (fun x => hingeLoss_nonnegative |x| _)
    (r := 6) (fun x hx => by
      have hx0 : 0 ≤ x := (invPower_nonneg 2 T).trans hx.1
      have hsmall : |x| ≤ 2 * invPower 2 T := by rw [abs_of_nonneg hx0]; exact hx.2
      have hg := lemma14_sample_count_lower T n x 0 h5 hn hnT hsmall
      rw [oneStepImprovement_eq_hinge] at hg
      exact hg)
  have hp := mul_le_mul_of_nonneg_left hmass
    (mul_nonneg gainConstant_positive.le (invPower_nonneg 6 T))
  have he : (gainConstant * invPower 6 T) * (lowerConstant * invPower 170 T) =
      (gainConstant * lowerConstant) * invPower 176 T := by
    rw [← invPower_product 6 170 T]
    ring
  rw [he] at hp
  exact hp.trans hband

/-- Both Eq. (24) inequalities proved THROUGH the explicit common-sample coupling. -/
theorem lemma19_individual_gains_via_coupling (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts) (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
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
    have := hc.2.2.2
    have := hc.2.2.1
    omega
  have hcap1 := hW.2.2.2.2.2.1
  have hcap2 := hW.2.2.2.2.2.2.1
  have hl := oriented_skip_left_counts q (T - t) c (by omega) hcap1 hcap2 hs
  have hr := equation22_terminal_counts q (T - t) c (by omega) hcap1 hcap2 hs
  rw [left_gain_common_sample_identity q (T - t) c _ _ (by omega) h1 h2 hcap1 hcap2 hs,
    right_gain_common_sample_identity q (T - t) c _ _ (by omega) h1 h2 hcap1 hcap2 hs]
  constructor
  · apply arbitrary_rule_hinge_gain_lower rightTie rightTie_smaller T q t
      (ruleRun rightTie (T - t) c).n1 CU p c hT ht hpre hc hW
    · rw [hl.1]; split_ifs <;> omega
    · rw [hl.1]; split_ifs <;> omega
  · apply arbitrary_rule_hinge_gain_lower leftTie leftTie_smaller T q t
      (ruleRun leftTie (T - t) c).n2 CU p c hT ht hpre hc hW
    · rw [leftTie_run, hr.2]; split_ifs <;> omega
    · rw [leftTie_run, hr.2]; split_ifs <;> omega

end
end Junpei.Coupling19
