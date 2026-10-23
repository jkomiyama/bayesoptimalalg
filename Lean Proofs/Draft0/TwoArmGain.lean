import Draft0.TwoArmBellman
import Draft0.PredictiveLaw

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.TwoArmGain

open Draft0.Gaussian Draft0.Counts Draft0.TwoArmBellman

noncomputable section

theorem absolute_hinge_smoothing_integrable (center : ℝ) (a b : ℝ≥0) :
    Integrable (fun z : ℝ => hingeLoss |center - z| b) (gaussianReal 0 a) := by
  have hf : (fun z : ℝ => hingeLoss |center - z| b) =
      (fun z : ℝ => hingeLoss (center - z) b - max (z - center) 0) := by
    funext z
    have h := hingeLoss_sub_abs (center - z) b
    have he : -(center - z) = z - center := by ring
    rw [he] at h
    linarith
  rw [hf]
  exact (hinge_smoothing_integrable center a b).sub (hinge_integrable center a)

theorem absolute_hinge_gaussian_integrable (mean : ℝ) (a b : ℝ≥0) :
    Integrable (fun x : ℝ => hingeLoss |x| b) (gaussianReal mean a) := by
  have hmap := gaussianReal_map_add_const (μ := (0 : ℝ)) (v := a) mean
  simp only [zero_add] at hmap
  rw [← hmap]
  apply (Homeomorph.addRight mean).measurableEmbedding.integrable_map_iff.mpr
  have h := absolute_hinge_smoothing_integrable (-mean) a b
  convert h using 1
  funext x
  change hingeLoss |x + mean| b = hingeLoss |-mean - x| b
  have he : -mean - x = -(x + mean) := by ring
  rw [he, abs_neg]

/-- Gaussian smoothing converts a terminal one-step gain into the exact
increase of an earlier value function. -/
theorem hinge_variance_increment_integral (center : ℝ) (a b : ℝ≥0) :
    hingeLoss center (a + b) - hingeLoss center a =
      ∫ x : ℝ, hingeLoss |x| b ∂gaussianReal (-center) a := by
  have hf : (fun z : ℝ => hingeLoss |center - z| b) =
      (fun z : ℝ => hingeLoss (center - z) b - max (z - center) 0) := by
    funext z
    have h := hingeLoss_sub_abs (center - z) b
    have he : -(center - z) = z - center := by ring
    rw [he] at h
    linarith
  have hsmall : (∫ z : ℝ, hingeLoss |center - z| b ∂gaussianReal 0 a) =
      hingeLoss center (a + b) - hingeLoss center a := by
    rw [hf, integral_sub (hinge_smoothing_integrable center a b) (hinge_integrable center a),
      hinge_smoothing]
    rfl
  have hmap := gaussianReal_map_add_const (μ := (0 : ℝ)) (v := a) (-center)
  simp only [zero_add] at hmap
  rw [← hmap, integral_map (by fun_prop) (by
    rw [hmap]
    exact (absolute_hinge_gaussian_integrable (-center) a b).aestronglyMeasurable)]
  rw [← hsmall]
  congr 1
  funext x
  have he : x + -center = -(center - x) := by ring
  rw [he, abs_neg]

theorem balancedRun_succ_tail (k : ℕ) (c : Counts) :
    balancedRun (k + 1) c = balancedStep (balancedRun k c) := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
    rw [balancedRun, ih, balancedRun]

theorem balancedCumulative_succ_tail (k : ℕ) (c : Counts) :
    balancedCumulative (k + 1) c =
      balancedCumulative k c + balancedInnovation (balancedRun k c) := by
  induction k generalizing c with
  | zero => simp only [balancedCumulative, balancedRun, add_zero, zero_add]
  | succ k ih =>
    change balancedInnovation c + balancedCumulative (k + 1) (balancedStep c) =
      (balancedInnovation c + balancedCumulative k (balancedStep c)) +
        balancedInnovation (balancedRun (k + 1) c)
    rw [ih (balancedStep c), balancedRun]
    ac_rfl

theorem cumulativeVariance_succ_tail (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    cumulativeVariance (k + 1) c =
      cumulativeVariance k c + balancedInnovation (balancedRun k c) := by
  rw [cumulativeVariance_eq_balancedCumulative _ _ h1 h2,
    cumulativeVariance_eq_balancedCumulative _ _ h1 h2,
    balancedCumulative_succ_tail]

theorem cumulativeVariance_eq_accumulatedVariance (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    cumulativeVariance k c = Draft0.PredictiveLaw.accumulatedVariance k c := by
  apply NNReal.coe_injective
  rw [cumulativeVariance_eq_transitionVariance _ _ h1 h2,
    Draft0.PredictiveLaw.accumulatedVariance_coe _ _ h1 h2]

/-- Actual two-arm horizon gain is the terminal hinge gain averaged under
the exact balanced Gaussian forecast already used in corrected Lemma 17. -/
theorem payoff_horizon_gain_integral (k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    payoff (k + 1) m1 m2 c - payoff k m1 m2 c =
      ∫ x : ℝ, hingeLoss |x| (balancedInnovation (balancedRun k c))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  rw [payoff_eq, payoff_eq, cumulativeVariance_succ_tail _ _ h1 h2]
  have he : m2 + hingeLoss (m2 - m1)
      (cumulativeVariance k c + balancedInnovation (balancedRun k c)) -
      (m2 + hingeLoss (m2 - m1) (cumulativeVariance k c)) =
      hingeLoss (m2 - m1) (cumulativeVariance k c + balancedInnovation (balancedRun k c)) -
        hingeLoss (m2 - m1) (cumulativeVariance k c) := by ring
  rw [he, hinge_variance_increment_integral,
    cumulativeVariance_eq_accumulatedVariance _ _ h1 h2,
    Draft0.PredictiveLaw.balancedPredictiveLaw_gaussian]
  congr 2
  ring

theorem leftAction_variance_decomposition (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    (updateVariance c.n1 + cumulativeVariance k (leftStep c)) +
      variance (balancedRun k (leftStep c)) = variance c := by
  rw [cumulativeVariance_eq_balancedCumulative k (leftStep c)
    (by simp only [leftStep]; omega) (by simpa only [leftStep] using h2),
    add_assoc, balancedCumulative_decomposition k (leftStep c)
    (by simp only [leftStep]; omega) (by simpa only [leftStep] using h2)]
  exact (leftStep_variance c h1).symm

theorem rightAction_variance_decomposition (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    (updateVariance c.n2 + cumulativeVariance k (rightStep c)) +
      variance (balancedRun k (rightStep c)) = variance c := by
  rw [cumulativeVariance_eq_balancedCumulative k (rightStep c)
    (by simpa only [rightStep] using h1) (by simp only [rightStep]; omega),
    add_assoc, balancedCumulative_decomposition k (rightStep c)
    (by simpa only [rightStep] using h1) (by simp only [rightStep]; omega)]
  exact (rightStep_variance c h2).symm

theorem cumulativeVariance_decomposition (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    cumulativeVariance k c + variance (balancedRun k c) = variance c := by
  rw [cumulativeVariance_eq_balancedCumulative _ _ h1 h2]
  exact balancedCumulative_decomposition _ _ h1 h2

theorem hinge_variance_comparison_integral (center : ℝ) (a small large : ℝ≥0) :
    hingeLoss center (a + large) - hingeLoss center (a + small) =
      ∫ x : ℝ, (hingeLoss |x| large - hingeLoss |x| small)
        ∂gaussianReal (-center) a := by
  rw [integral_sub (absolute_hinge_gaussian_integrable _ _ _)
    (absolute_hinge_gaussian_integrable _ _ _),
    ← hinge_variance_increment_integral, ← hinge_variance_increment_integral]
  ring

theorem balancedRun_right_only (k : ℕ) (c : Counts) (h : c.n2 + k ≤ c.n1) :
    balancedRun k c = ⟨c.n1, c.n2 + k, c.n3⟩ := by
  induction k generalizing c with
  | zero => simp only [balancedRun, Nat.add_zero]
  | succ k ih =>
    have hn : ¬ c.n1 ≤ c.n2 := by omega
    have hstep : balancedStep c = rightStep c := by
      simp only [balancedStep, rightStep, if_neg hn]
    have hnext : (rightStep c).n2 + k ≤ (rightStep c).n1 := by
      change (c.n2 + 1) + k ≤ c.n1
      omega
    rw [balancedRun, hstep, ih (rightStep c) hnext]
    change (⟨c.n1, (c.n2 + 1) + k, c.n3⟩ : Counts) = ⟨c.n1, c.n2 + (k + 1), c.n3⟩
    congr 1
    omega

theorem balancedRun_left_only (k : ℕ) (c : Counts) (h : c.n1 + k ≤ c.n2) :
    balancedRun k c = ⟨c.n1 + k, c.n2, c.n3⟩ := by
  induction k generalizing c with
  | zero => simp only [balancedRun, Nat.add_zero]
  | succ k ih =>
    have hn : c.n1 ≤ c.n2 := by omega
    have hstep : balancedStep c = leftStep c := by
      simp only [balancedStep, leftStep, if_pos hn]
    have hnext : (leftStep c).n1 + k ≤ (leftStep c).n2 := by
      change (c.n1 + 1) + k ≤ c.n2
      omega
    rw [balancedRun, hstep, ih (leftStep c) hnext]
    change (⟨(c.n1 + 1) + k, c.n2, c.n3⟩ : Counts) = ⟨c.n1 + (k + 1), c.n2, c.n3⟩
    congr 1
    omega

theorem balancedRun_left_commute (k : ℕ) (c : Counts) (h : c.n2 + k ≤ c.n1) :
    balancedRun k (leftStep c) = leftStep (balancedRun k c) := by
  have hnext : (leftStep c).n2 + k ≤ (leftStep c).n1 := by
    change c.n2 + k ≤ c.n1 + 1
    omega
  rw [balancedRun_right_only k c h, balancedRun_right_only k (leftStep c) hnext]
  rfl

theorem balancedRun_right_commute (k : ℕ) (c : Counts) (h : c.n1 + k ≤ c.n2) :
    balancedRun k (rightStep c) = rightStep (balancedRun k c) := by
  have hnext : (rightStep c).n1 + k ≤ (rightStep c).n2 := by
    change c.n1 + k ≤ c.n2 + 1
    omega
  rw [balancedRun_left_only k c h, balancedRun_left_only k (rightStep c) hnext]
  rfl

theorem leftAction_variance_eq_append (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcomm : balancedRun k (leftStep c) = leftStep (balancedRun k c)) :
    updateVariance c.n1 + cumulativeVariance k (leftStep c) =
      cumulativeVariance k c + updateVariance (balancedRun k c).n1 := by
  have hmono := balancedRun_monotone k c
  have hstep := leftStep_variance (balancedRun k c) (by omega)
  have hcum := cumulativeVariance_decomposition k c h1 h2
  have hact := leftAction_variance_decomposition k c h1 h2
  rw [hcomm] at hact
  apply add_right_cancel (b := variance (leftStep (balancedRun k c)))
  calc
    _ = variance c := hact
    _ = cumulativeVariance k c + variance (balancedRun k c) := hcum.symm
    _ = _ := by rw [hstep]; ac_rfl

theorem rightAction_variance_eq_append (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcomm : balancedRun k (rightStep c) = rightStep (balancedRun k c)) :
    updateVariance c.n2 + cumulativeVariance k (rightStep c) =
      cumulativeVariance k c + updateVariance (balancedRun k c).n2 := by
  have hmono := balancedRun_monotone k c
  have hstep := rightStep_variance (balancedRun k c) (by omega)
  have hcum := cumulativeVariance_decomposition k c h1 h2
  have hact := rightAction_variance_decomposition k c h1 h2
  rw [hcomm] at hact
  apply add_right_cancel (b := variance (rightStep (balancedRun k c)))
  calc
    _ = variance c := hact
    _ = cumulativeVariance k c + variance (balancedRun k c) := hcum.symm
    _ = _ := by rw [hstep]; ac_rfl

/-- Sampling an arm already saturated at the larger count has a loss gap
equal to the averaged terminal comparison under the exact Gaussian forecast. -/
theorem saturated_left_action_gap_integral (k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) (h : c.n2 + k ≤ c.n1) :
    payoff (k + 1) m1 m2 c - leftAction k m1 m2 c =
      ∫ x : ℝ, (hingeLoss |x| (balancedInnovation (balancedRun k c)) -
        hingeLoss |x| (updateVariance (balancedRun k c).n1))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  rw [payoff_eq, leftAction_eq, cumulativeVariance_succ_tail _ _ h1 h2,
    leftAction_variance_eq_append _ _ h1 h2 (balancedRun_left_commute k c h)]
  have he : m2 + hingeLoss (m2 - m1)
      (cumulativeVariance k c + balancedInnovation (balancedRun k c)) -
      (m2 + hingeLoss (m2 - m1)
        (cumulativeVariance k c + updateVariance (balancedRun k c).n1)) =
      hingeLoss (m2 - m1) (cumulativeVariance k c + balancedInnovation (balancedRun k c)) -
        hingeLoss (m2 - m1) (cumulativeVariance k c + updateVariance (balancedRun k c).n1) := by ring
  rw [he, hinge_variance_comparison_integral,
    cumulativeVariance_eq_accumulatedVariance _ _ h1 h2,
    Draft0.PredictiveLaw.balancedPredictiveLaw_gaussian]
  congr 2
  ring

theorem saturated_right_action_gap_integral (k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) (h : c.n1 + k ≤ c.n2) :
    payoff (k + 1) m1 m2 c - rightAction k m1 m2 c =
      ∫ x : ℝ, (hingeLoss |x| (balancedInnovation (balancedRun k c)) -
        hingeLoss |x| (updateVariance (balancedRun k c).n2))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  rw [payoff_eq, rightAction_eq, cumulativeVariance_succ_tail _ _ h1 h2,
    rightAction_variance_eq_append _ _ h1 h2 (balancedRun_right_commute k c h)]
  have he : m2 + hingeLoss (m2 - m1)
      (cumulativeVariance k c + balancedInnovation (balancedRun k c)) -
      (m2 + hingeLoss (m2 - m1)
        (cumulativeVariance k c + updateVariance (balancedRun k c).n2)) =
      hingeLoss (m2 - m1) (cumulativeVariance k c + balancedInnovation (balancedRun k c)) -
        hingeLoss (m2 - m1) (cumulativeVariance k c + updateVariance (balancedRun k c).n2) := by ring
  rw [he, hinge_variance_comparison_integral,
    cumulativeVariance_eq_accumulatedVariance _ _ h1 h2,
    Draft0.PredictiveLaw.balancedPredictiveLaw_gaussian]
  congr 2
  ring

theorem balancedInnovation_preterminal (q k : ℕ) (c : Counts)
    (hq : 0 < q) (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    balancedInnovation (balancedRun k c) = updateVariance ((q - 1 : ℕ) : ℝ≥0) := by
  have hc := equation22_terminal_counts q k c hq hcap1 hcap2 hs
  unfold balancedInnovation
  by_cases hn : c.n2 = q
  · simp only [if_pos hn] at hc
    rw [hc.1, hc.2, if_pos (by omega : q - 1 ≤ q)]
  · simp only [if_neg hn] at hc
    rw [hc.1, hc.2, if_neg (by omega : ¬ q ≤ q - 1)]

theorem payoff_horizon_gain_preterminal (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 0 < q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    payoff (k + 1) m1 m2 c - payoff k m1 m2 c =
      ∫ x : ℝ, hingeLoss |x| (updateVariance ((q - 1 : ℕ) : ℝ≥0))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  rw [payoff_horizon_gain_integral _ _ _ _ h1 h2,
    balancedInnovation_preterminal _ _ _ hq hcap1 hcap2 hs]

theorem saturated_left_prefix_counts (q k : ℕ) (c : Counts)
    (hq : 0 < q) (hsaturated : c.n1 = q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (balancedRun k c).n1 = q ∧ (balancedRun k c).n2 = q - 1 := by
  have htot : c.n2 + k = q - 1 := by omega
  have hright : c.n2 + k ≤ c.n1 := by omega
  rw [balancedRun_right_only _ _ hright]
  exact ⟨hsaturated, htot⟩

theorem saturated_right_prefix_counts (q k : ℕ) (c : Counts)
    (hq : 0 < q) (hsaturated : c.n2 = q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (balancedRun k c).n1 = q - 1 ∧ (balancedRun k c).n2 = q := by
  have htot : c.n1 + k = q - 1 := by omega
  have hleft : c.n1 + k ≤ c.n2 := by omega
  rw [balancedRun_left_only _ _ hleft]
  exact ⟨htot, hsaturated⟩

theorem saturated_left_gap_preterminal (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 0 < q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hsaturated : c.n1 = q) (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    payoff (k + 1) m1 m2 c - leftAction k m1 m2 c =
      ∫ x : ℝ, (hingeLoss |x| (updateVariance ((q - 1 : ℕ) : ℝ≥0)) -
        hingeLoss |x| (updateVariance (q : ℝ≥0)))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  have hc := saturated_left_prefix_counts q k c hq hsaturated hs
  have hright : c.n2 + k ≤ c.n1 := by omega
  rw [saturated_left_action_gap_integral _ _ _ _ h1 h2 hright]
  have hb : balancedInnovation (balancedRun k c) =
      updateVariance ((q - 1 : ℕ) : ℝ≥0) := by
    unfold balancedInnovation
    rw [hc.1, hc.2, if_neg (by omega : ¬ q ≤ q - 1)]
  rw [hb, hc.1]

theorem saturated_right_gap_preterminal (q k : ℕ) (m1 m2 : ℝ) (c : Counts)
    (hq : 0 < q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hsaturated : c.n2 = q) (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    payoff (k + 1) m1 m2 c - rightAction k m1 m2 c =
      ∫ x : ℝ, (hingeLoss |x| (updateVariance ((q - 1 : ℕ) : ℝ≥0)) -
        hingeLoss |x| (updateVariance (q : ℝ≥0)))
        ∂Draft0.PredictiveLaw.balancedPredictiveLaw k c (m1 - m2) := by
  have hc := saturated_right_prefix_counts q k c hq hsaturated hs
  have hleft : c.n1 + k ≤ c.n2 := by omega
  rw [saturated_right_action_gap_integral _ _ _ _ h1 h2 hleft]
  have hb : balancedInnovation (balancedRun k c) =
      updateVariance ((q - 1 : ℕ) : ℝ≥0) := by
    unfold balancedInnovation
    rw [hc.1, hc.2, if_pos (by omega : q - 1 ≤ q)]
  rw [hb, hc.2]

end

end Draft0.TwoArmGain
