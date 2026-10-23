import Draft0.TwoArmGain
import Draft0.Drift

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.EvenCounts

open Draft0.Counts Draft0.Drift Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain

noncomputable section

/-- Even budgets have two fewer arm-1/2 samples before the terminal draw.
Already saturated arms keep q observations; otherwise both reach q-1. -/
theorem even_prefix_counts (q k : ℕ) (c : Counts)
    (hq : 2 ≤ q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    (balancedRun k c).n1 =
      (if c.n1 = q then q else if c.n2 = q then q - 2 else q - 1) ∧
    (balancedRun k c).n2 =
      (if c.n1 = q then q - 2 else if c.n2 = q then q else q - 1) := by
  have hmono := balancedRun_monotone k c
  have hsum := balancedRun_sum k c
  have hmax := balancedRun_max k c
  by_cases hn1 : c.n1 = q
  · simp only [if_pos hn1]
    constructor <;> omega
  · by_cases hn2 : c.n2 = q
    · simp only [if_neg hn1, if_pos hn2]
      constructor <;> omega
    · simp only [if_neg hn1, if_neg hn2]
      constructor <;> omega

theorem even_prefix_caps (q k : ℕ) (c : Counts)
    (hq : 2 ≤ q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    (balancedRun k c).n1 ≤ q ∧ (balancedRun k c).n2 ≤ q := by
  have hc := even_prefix_counts q k c hq h1 h2 hs
  constructor
  · rw [hc.1]; split_ifs <;> omega
  · rw [hc.2]; split_ifs <;> omega

theorem even_transition_residual_lower (q k : ℕ) (c : Counts)
    (hq : 3 ≤ q) (hk : 0 < k) (hc1 : 1 ≤ c.n1) (hc2 : 1 ≤ c.n2)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    1 / ((q : ℝ) * ((q : ℝ) - 2)) ≤ transitionVariance k c := by
  have hc := even_prefix_counts q k c (by omega) h1 h2 hs
  have hmono := balancedRun_monotone k c
  have hsum := balancedRun_sum k c
  have hprogress : c.n1 < (balancedRun k c).n1 ∨ c.n2 < (balancedRun k c).n2 := by omega
  have hqr : 3 ≤ (q : ℝ) := by exact_mod_cast hq
  have hqp : 0 < (q : ℝ) := by linarith
  have hqsp : 0 < (q : ℝ) - 2 := by linarith
  have hq1p : 0 < (q : ℝ) - 1 := by linarith
  have hq11p : 0 < ((q : ℝ) - 1) - 1 := by linarith
  have hq1 : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - 1 := by
    simp only [Nat.cast_sub (show 1 ≤ q by omega), Nat.cast_one]
  have hcomp : 1 / ((q : ℝ) * ((q : ℝ) - 2)) ≤
      1 / (((q - 1 : ℕ) : ℝ) * (((q - 1 : ℕ) : ℝ) - 1)) := by
    rw [hq1]
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith
  unfold transitionVariance
  rcases hprogress with hp1 | hp2
  · have hb : (balancedRun k c).n1 ≤ q - 1 := by
      rw [hc.1] at *
      split_ifs at * <;> omega
    have hi := inverse_difference_preterminal_lower_bound c.n1 (balancedRun k c).n1
      (q - 1) hc1 hp1 hb
    have hn := inverse_difference_nonnegative c.n2 _ (by omega) hmono.2
    linarith
  · have hb : (balancedRun k c).n2 ≤ q - 1 := by
      rw [hc.2] at *
      split_ifs at * <;> omega
    have hi := inverse_difference_preterminal_lower_bound c.n2 (balancedRun k c).n2
      (q - 1) hc2 hp2 hb
    have hn := inverse_difference_nonnegative c.n1 _ (by omega) hmono.1
    linarith

theorem even_terminal_anchor_residual_upper (q k : ℕ) (c : Counts)
    (hq : 3 ≤ q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    (1 / ((balancedRun k c).n1 : ℝ) - 1 / (q : ℝ)) +
      (1 / ((balancedRun k c).n2 : ℝ) - 1 / (q : ℝ)) ≤
      2 / ((q : ℝ) * ((q : ℝ) - 2)) := by
  have hc := even_prefix_counts q k c (by omega) h1 h2 hs
  have hqr : 3 ≤ (q : ℝ) := by exact_mod_cast hq
  have hqp : 0 < (q : ℝ) := by linarith
  have hq1p : 0 < (q : ℝ) - 1 := by linarith
  have hq2p : 0 < (q : ℝ) - 2 := by linarith
  have hq1 : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - 1 := by
    simp only [Nat.cast_sub (show 1 ≤ q by omega), Nat.cast_one]
  have hq2 : ((q - 2 : ℕ) : ℝ) = (q : ℝ) - 2 := by
    simp only [Nat.cast_sub (show 2 ≤ q by omega), Nat.cast_ofNat]
  have hd1 : 1 / ((q : ℝ) - 1) - 1 / (q : ℝ) =
      1 / ((q : ℝ) * ((q : ℝ) - 1)) := by field_simp; ring
  have hd2 : 1 / ((q : ℝ) - 2) - 1 / (q : ℝ) =
      2 / ((q : ℝ) * ((q : ℝ) - 2)) := by field_simp; ring
  rw [hc.1, hc.2]
  by_cases hn1 : c.n1 = q
  · simp only [if_pos hn1, hq2, sub_self, zero_add, hd2]
    exact le_rfl
  · by_cases hn2 : c.n2 = q
    · simp only [if_neg hn1, if_pos hn2, hq2, sub_self, add_zero, hd2]
      exact le_rfl
    · simp only [if_neg hn1, if_neg hn2, hq1, hd1]
      have hd : 2 / ((q : ℝ) * ((q : ℝ) - 1)) ≤
          2 / ((q : ℝ) * ((q : ℝ) - 2)) := by
        apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
        nlinarith
      calc
        _ = 2 / ((q : ℝ) * ((q : ℝ) - 1)) := by ring
        _ ≤ _ := hd

theorem even_to_anchor_variance_bound (q k : ℕ) (c : Counts)
    (hq : 3 ≤ q) (hk : 0 < k) (hc1 : 1 ≤ c.n1) (hc2 : 1 ≤ c.n2)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    (1 / (c.n1 : ℝ) - 1 / (q : ℝ)) +
      (1 / (c.n2 : ℝ) - 1 / (q : ℝ)) ≤ 3 * transitionVariance k c := by
  have hl := even_transition_residual_lower q k c hq hk hc1 hc2 h1 h2 hs
  have hu := even_terminal_anchor_residual_upper q k c hq h1 h2 hs
  have hscale : 2 / ((q : ℝ) * ((q : ℝ) - 2)) ≤ 2 * transitionVariance k c := by
    calc
      _ = 2 * (1 / ((q : ℝ) * ((q : ℝ) - 2))) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hl (by norm_num)
  unfold transitionVariance at *
  linarith

theorem even_balancedInnovation_preterminal (q k : ℕ) (c : Counts)
    (hq : 3 ≤ q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    balancedInnovation (balancedRun k c) =
      if c.n1 = q ∨ c.n2 = q then updateVariance ((q - 2 : ℕ) : ℝ≥0)
      else updateVariance ((q - 1 : ℕ) : ℝ≥0) := by
  have hc := even_prefix_counts q k c (by omega) h1 h2 hs
  unfold balancedInnovation
  rw [hc.1, hc.2]
  by_cases hn1 : c.n1 = q
  · have hsat : c.n1 = q ∨ c.n2 = q := Or.inl hn1
    simp only [if_pos hn1, if_pos hsat]
    rw [if_neg (by omega : ¬ q ≤ q - 2)]
  · by_cases hn2 : c.n2 = q
    · simp only [if_neg hn1, if_pos hn2]
      have hsat : c.n1 = q ∨ c.n2 = q := Or.inr hn2
      rw [if_pos hsat]
      rw [if_pos (by omega : q - 2 ≤ q)]
    · have hsat : ¬ (c.n1 = q ∨ c.n2 = q) := not_or.mpr ⟨hn1, hn2⟩
      simp only [if_neg hn1, if_neg hn2, if_neg hsat, le_refl, if_true]

theorem even_min_count_bounds (q k : ℕ) (c : Counts)
    (hq : 3 ≤ q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    q - 2 ≤ min (balancedRun k c).n1 (balancedRun k c).n2 ∧
      min (balancedRun k c).n1 (balancedRun k c).n2 ≤ q - 1 := by
  have hc := even_prefix_counts q k c (by omega) h1 h2 hs
  rw [hc.1, hc.2]
  split_ifs <;> constructor <;> omega

theorem even_horizon_gain_preterminal (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 3 ≤ q) (hc1 : 0 < c.n1) (hc2 : 0 < c.n2)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    payoff (k + 1) m1 m2 c - payoff k m1 m2 c =
      ∫ x : ℝ, hingeLoss |x|
        (if c.n1 = q ∨ c.n2 = q then updateVariance ((q - 2 : ℕ) : ℝ≥0)
         else updateVariance ((q - 1 : ℕ) : ℝ≥0))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  rw [payoff_horizon_gain_integral _ _ _ _ hc1 hc2,
    even_balancedInnovation_preterminal _ _ _ hq h1 h2 hs]

theorem even_saturated_left_gap_preterminal (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 3 ≤ q) (hc1 : 0 < c.n1) (hc2 : 0 < c.n2)
    (hsaturated : c.n1 = q) (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    payoff (k + 1) m1 m2 c - leftAction k m1 m2 c =
      ∫ x : ℝ, (hingeLoss |x| (updateVariance ((q - 2 : ℕ) : ℝ≥0)) -
        hingeLoss |x| (updateVariance (q : ℝ≥0)))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  have htot : c.n2 + k = q - 2 := by omega
  have hright : c.n2 + k ≤ c.n1 := by omega
  have hcap2 : c.n2 ≤ q := by omega
  have hc := even_prefix_counts q k c (by omega) (by omega) hcap2 hs
  simp only [if_pos hsaturated] at hc
  rw [saturated_left_action_gap_integral _ _ _ _ hc1 hc2 hright]
  have hb : balancedInnovation (balancedRun k c) =
      updateVariance ((q - 2 : ℕ) : ℝ≥0) := by
    unfold balancedInnovation
    rw [hc.1, hc.2, if_neg (by omega : ¬ q ≤ q - 2)]
  rw [hb, hc.1]

theorem even_saturated_right_gap_preterminal (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 3 ≤ q) (hc1 : 0 < c.n1) (hc2 : 0 < c.n2)
    (hsaturated : c.n2 = q) (hs : c.n1 + c.n2 + k = 2 * q - 2) :
    payoff (k + 1) m1 m2 c - rightAction k m1 m2 c =
      ∫ x : ℝ, (hingeLoss |x| (updateVariance ((q - 2 : ℕ) : ℝ≥0)) -
        hingeLoss |x| (updateVariance (q : ℝ≥0)))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  have htot : c.n1 + k = q - 2 := by omega
  have hleft : c.n1 + k ≤ c.n2 := by omega
  have hn1 : ¬ c.n1 = q := by omega
  have hcap1 : c.n1 ≤ q := by omega
  have hc := even_prefix_counts q k c (by omega) hcap1 (by omega) hs
  simp only [if_neg hn1, if_pos hsaturated] at hc
  rw [saturated_right_action_gap_integral _ _ _ _ hc1 hc2 hleft]
  have hb : balancedInnovation (balancedRun k c) =
      updateVariance ((q - 2 : ℕ) : ℝ≥0) := by
    unfold balancedInnovation
    rw [hc.1, hc.2, if_pos (by omega : q - 2 ≤ q)]
  rw [hb, hc.2]

end

end Draft0.EvenCounts
