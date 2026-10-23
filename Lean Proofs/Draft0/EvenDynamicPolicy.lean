import Draft0.DynamicPolicy
import Draft0.EvenDrift
import Draft0.EvenDynamicGain

open MeasureTheory ProbabilityTheory
open Filter Asymptotics
open scoped NNReal ENNReal

namespace Draft0.EvenDynamicPolicy

open Draft0.Counts Draft0.Drift Draft0.Gaussian Draft0.Rates
open Draft0.TerminalBounds Draft0.PredictiveBound Draft0.PredictiveLaw
open Draft0.DynamicPolicy Draft0.EvenCounts

noncomputable section

theorem slack_bad_state_upper (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : 5 ≤ T) (hc : horizonConsistent t c) (hW : paperW T q 50 p c) :
    Draft0.ThreeArmComparison.stateSlack (modelCounts c) (modelMeans p c) ≤
      2 * invPower 400 T := by
  have hv : (posteriorVariance (c.n1 : ℝ≥0) (c.n3 : ℝ≥0) : ℝ) ≤ 2 := by
    rw [hc.2.2.1]
    simp only [posteriorVariance, NNReal.coe_add, NNReal.coe_inv, NNReal.coe_natCast,
      ]
    have hn : (1 : ℝ) ≤ c.n1 := by exact_mod_cast hc.1
    have hsmall : (c.n1 : ℝ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hn
    linarith
  have he : Draft0.ThreeArmComparison.stateSlack (modelCounts c) (modelMeans p c) =
      hingeLoss (p 1 c.n1 - p 3 1) (posteriorVariance (c.n1 : ℝ≥0) (c.n3 : ℝ≥0)) := by
    simp [Draft0.ThreeArmComparison.stateSlack, Draft0.ThreeArmComparison.slack,
      modelCounts, modelMeans, hc.2.2.1]
  rw [he]
  exact Draft0.GaussianTail.bad_state_tail50 T q c.n1 p hT hc.1 hW.2.2.2.2.2.1
    hW.1 hW.2.1 hW.2.2.2.1 _ hv

theorem forecast_parameters (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 50 p c) :
    0 < accumulatedVariance (T - t) c ∧
      (accumulatedVariance (T - t) c : ℝ) ≤ 2 ∧
      invPower 2 T ≤ (accumulatedVariance (T - t) c : ℝ) ∧
      (terminalGap p c) ^ 2 ≤ 300 * Real.log (T : ℝ) *
        (accumulatedVariance (T - t) c : ℝ) := by
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hV := accumulatedVariance_coe (T - t) c h1 h2
  have hv : 0 < accumulatedVariance (T - t) c := by
    change (0 : ℝ) < (accumulatedVariance (T - t) c : ℝ)
    rw [hV]
    exact transitionVariance_positive (T - t) c (by omega) h1 h2
  have hu : (accumulatedVariance (T - t) c : ℝ) ≤ 2 := by
    rw [hV]; exact transitionVariance_le_two _ _ hc.1 hc.2.1
  have hl : invPower 2 T ≤ (accumulatedVariance (T - t) c : ℝ) := by
    have hs : c.n1 + c.n2 + (T - t) = 2 * q - 2 := by
      rcases hc with ⟨_, _, h3, htotal⟩; omega
    have hcaps := even_prefix_caps q (T - t) c (by omega)
      hW.2.2.2.2.2.1 hW.2.2.2.2.2.2.1 hs
    rw [invPower, inv_eq_one_div, hV]
    exact transitionVariance_ge_inverse_square_of_caps (T - t) T c (by omega)
      h1 h2 (by omega) (by omega)
  have hd : (terminalGap p c) ^ 2 ≤ 300 * Real.log (T : ℝ) *
      (accumulatedVariance (T - t) c : ℝ) := by
    rw [hV]; exact Draft0.EvenDrift.even_fullW_drift_bound T q t 50 p c hT ht hpre hc hW
  exact ⟨hv, hu, hl, hd⟩

theorem nonterminal_horizon_gain (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 50 p c) :
    gainCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.payoff (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hpar := forecast_parameters T q t p c hT ht hpre hc hW
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 2 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  rw [even_horizon_gain_preterminal q (T - t) _ _ c (by omega)
    h1 h2 hW.2.2.2.2.2.1 hW.2.2.2.2.2.2.1 hs,
    balancedPredictiveLaw_gaussian]
  apply (mul_le_mul_of_nonneg_right (min_le_left _ _) (invPower_nonneg 370 T)).trans
  split_ifs
  · exact Draft0.EvenDynamicGain.nonterminal_hinge_gain_lower_conservative T (q - 2)
      (terminalGap p c) _ (by omega) (by omega) (by omega)
      hpar.1 hpar.2.1 hpar.2.2.1 hpar.2.2.2
  · exact Draft0.EvenDynamicGain.nonterminal_hinge_gain_lower_conservative T (q - 1)
      (terminalGap p c) _ (by omega) (by omega) (by omega)
      hpar.1 hpar.2.1 hpar.2.2.1 hpar.2.2.2

theorem nonterminal_left_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 50 p c) (hsat : c.n1 = q) :
    comparisonCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hpar := forecast_parameters T q t p c hT ht hpre hc hW
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 2 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  rw [even_saturated_left_gap_preterminal q (T - t) _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_gaussian]
  exact (mul_le_mul_of_nonneg_right (min_le_left _ _) (invPower_nonneg 370 T)).trans
    (Draft0.EvenDynamicGain.nonterminal_hinge_difference_lower_conservative
      T q (q - 2) (terminalGap p c) _ (by omega)
      (by omega) (by omega) (by omega) hpar.1 hpar.2.1 hpar.2.2.1 hpar.2.2.2)

theorem nonterminal_right_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 50 p c) (hsat : c.n2 = q) :
    comparisonCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hpar := forecast_parameters T q t p c hT ht hpre hc hW
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 2 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  rw [even_saturated_right_gap_preterminal q (T - t) _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_gaussian]
  exact (mul_le_mul_of_nonneg_right (min_le_left _ _) (invPower_nonneg 370 T)).trans
    (Draft0.EvenDynamicGain.nonterminal_hinge_difference_lower_conservative
      T q (q - 2) (terminalGap p c) _ (by omega)
      (by omega) (by omega) (by omega) hpar.1 hpar.2.1 hpar.2.2.1 hpar.2.2.2)

theorem terminal_horizon_gain (T q : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (hT12 : 12 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q 50 p c) :
    gainCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff 1 (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.payoff 0 (p 1 c.n1) (p 2 c.n2) c := by
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hq : 6 ≤ q := by omega
  have hs : c.n1 + c.n2 + 0 = 2 * q - 2 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  have hd := Draft0.EvenDrift.even_terminal_fullW_drift_bound T q 50 p c hT hT12 hc hW
  have hb := updateVariance_count_bounds q T (by omega) (by omega)
  have hv : 0 < updateVariance (q : ℝ≥0) := by unfold updateVariance; positivity
  have hlower := Draft0.EvenDynamicGain.terminal_hinge_gain_lower_conservative
    T (terminalGap p c) _ (by omega) hv
    (by linarith only [hb.2]) hb.1 hd
  rw [even_horizon_gain_preterminal q 0 _ _ c (by omega)
    h1 h2 hW.2.2.2.2.2.1 hW.2.2.2.2.2.2.1 hs, balancedPredictiveLaw_terminal,
    integral_dirac]
  apply (mul_le_mul_of_nonneg_right (min_le_right _ _) (invPower_nonneg 370 T)).trans
  apply hlower.trans
  split_ifs
  · exact (hingeLoss_strictMono_variance |terminalGap p c| (abs_nonneg _)
      (updateVariance (q : ℝ≥0)) (updateVariance ((q - 2 : ℕ) : ℝ≥0))
      (updateVariance_strictAntitone _ _
        (by exact_mod_cast (show 0 < q - 2 by omega))
        (by exact_mod_cast (show q - 2 < q by omega)))).le
  · exact (hingeLoss_strictMono_variance |terminalGap p c| (abs_nonneg _)
      (updateVariance (q : ℝ≥0)) (updateVariance ((q - 1 : ℕ) : ℝ≥0))
      (updateVariance_strictAntitone _ _
        (by exact_mod_cast (show 0 < q - 1 by omega))
        (by exact_mod_cast (show q - 1 < q by omega)))).le

theorem terminal_left_gap (T q : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (hT12 : 12 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q 50 p c) (hsat : c.n1 = q) :
    comparisonCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff 1 (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.leftAction 0 (p 1 c.n1) (p 2 c.n2) c := by
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + 0 = 2 * q - 2 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  have hd := Draft0.EvenDrift.even_terminal_fullW_drift_bound T q 50 p c hT hT12 hc hW
  have hlower := Draft0.EvenDynamicGain.terminal_hinge_difference_lower_conservative
    T q (q - 2) (terminalGap p c) (by omega) (by omega) (by omega) (by omega) hd
  rw [even_saturated_left_gap_preterminal q 0 _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_terminal, integral_dirac]
  exact (mul_le_mul_of_nonneg_right (min_le_right _ _) (invPower_nonneg 370 T)).trans hlower

theorem terminal_right_gap (T q : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (hT12 : 12 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q 50 p c) (hsat : c.n2 = q) :
    comparisonCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff 1 (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.rightAction 0 (p 1 c.n1) (p 2 c.n2) c := by
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + 0 = 2 * q - 2 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  have hd := Draft0.EvenDrift.even_terminal_fullW_drift_bound T q 50 p c hT hT12 hc hW
  have hlower := Draft0.EvenDynamicGain.terminal_hinge_difference_lower_conservative
    T q (q - 2) (terminalGap p c) (by omega) (by omega) (by omega) (by omega) hd
  rw [even_saturated_right_gap_preterminal q 0 _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_terminal, integral_dirac]
  exact (mul_le_mul_of_nonneg_right (min_le_right _ _) (invPower_nonneg 370 T)).trans hlower

theorem all_round_horizon_gain (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (hT12 : 12 ≤ T) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (hW : paperW T q 50 p c) :
    gainCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.payoff (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  by_cases hpre : t < T
  · exact nonterminal_horizon_gain T q t p c hT ht hpre hc hW
  · have heq : t = T := by omega
    subst t
    simpa only [Nat.sub_self, zero_add] using terminal_horizon_gain T q p c hT hT12 hc hW

theorem all_round_left_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (hT12 : 12 ≤ T) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (hW : paperW T q 50 p c) (hsat : c.n1 = q) :
    comparisonCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  by_cases hpre : t < T
  · exact nonterminal_left_gap T q t p c hT ht hpre hc hW hsat
  · have heq : t = T := by omega
    subst t
    simpa only [Nat.sub_self, zero_add] using terminal_left_gap T q p c hT hT12 hc hW hsat

theorem all_round_right_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (hT12 : 12 ≤ T) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (hW : paperW T q 50 p c) (hsat : c.n2 = q) :
    comparisonCoefficient * invPower 370 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  by_cases hpre : t < T
  · exact nonterminal_right_gap T q t p c hT ht hpre hc hW hsat
  · have heq : t = T := by omega
    subst t
    simpa only [Nat.sub_self, zero_add] using terminal_right_gap T q p c hT hT12 hc hW hsat

/-- A single budget threshold controls all W states and all decision rounds, including T. -/
theorem eventually_all_W_admissible :
    ∃ T0 : ℕ, ∀ T : ℕ, T0 ≤ T → ∀ (q t : ℕ) (p : MeanPath) (c : Counts),
      T = 2 * q → 4 ≤ t → t ≤ T → horizonConsistent t c →
      paperW T q 50 p c → admissible q c (decision T t p c) := by
  have herr : (fun T => 2 * invPower 400 T) =O[atTop] invPower 400 :=
    (isBigO_refl (invPower 400) atTop).const_mul_left 2
  have hg := eventual_subtraction_positive gainCoefficient_pos (by norm_num : (0 : ℝ) < 1)
    (by omega : 370 < 400) herr (Filter.Eventually.of_forall (fun T => le_rfl))
  have hb := eventual_subtraction_positive comparisonCoefficient_pos (by norm_num : (0 : ℝ) < 1)
    (by omega : 370 < 400) herr (Filter.Eventually.of_forall (fun T => le_rfl))
  obtain ⟨T0, hT0⟩ := Filter.eventually_atTop.mp
    (hg.and (hb.and (eventually_ge_atTop (12 : ℕ))))
  refine ⟨T0, ?_⟩
  intro T hlarge q t p c hT ht horizon hc hW
  have hbudget := hT0 T hlarge
  have hT12 : 12 ≤ T := hbudget.2.2
  have hslack := slack_bad_state_upper T q t p c (by omega) hc hW
  have hpositive := modelCounts_positive t c hc
  have hgain := all_round_horizon_gain T q t p c hT hT12 ht horizon hc hW
  have hgainGap : Draft0.ThreeArmComparison.stateSlack (modelCounts c) (modelMeans p c) <
      Draft0.ThreeArmComparison.twoValue (T - t + 1) (modelCounts c) (modelMeans p c) -
      Draft0.ThreeArmComparison.twoValue (T - t) (modelCounts c) (modelMeans p c) := by
    rw [twoValue_model, twoValue_model]
    linarith only [hslack, hgain, hbudget.1]
  have hthird := Draft0.ThreeArmComparison.third_never_chosen (T - t)
    (modelCounts c) (modelMeans p c) hpositive hgainGap
  refine ⟨hthird, ?_, ?_⟩
  · intro hchoose
    by_contra hnot
    have hsat : c.n1 = q := by have := hW.2.2.2.2.2.1; omega
    have hbnd := all_round_left_gap T q t p c hT hT12 ht horizon hc hW hsat
    have hgap : Draft0.ThreeArmComparison.stateSlack (modelCounts c) (modelMeans p c) <
        Draft0.ThreeArmComparison.twoValue (T - t + 1) (modelCounts c) (modelMeans p c) -
        Draft0.ThreeArmComparison.twoAction (T - t) (modelCounts c) (modelMeans p c) 0 := by
      rw [twoValue_model, twoAction_model_zero]
      linarith only [hslack, hbnd, hbudget.2.1]
    exact Draft0.ThreeArmComparison.deficient_action_never_chosen (T - t)
      (modelCounts c) (modelMeans p c) 0 hpositive hgap hchoose
  · intro hchoose
    by_contra hnot
    have hsat : c.n2 = q := by have := hW.2.2.2.2.2.2.1; omega
    have hbnd := all_round_right_gap T q t p c hT hT12 ht horizon hc hW hsat
    have hgap : Draft0.ThreeArmComparison.stateSlack (modelCounts c) (modelMeans p c) <
        Draft0.ThreeArmComparison.twoValue (T - t + 1) (modelCounts c) (modelMeans p c) -
        Draft0.ThreeArmComparison.twoAction (T - t) (modelCounts c) (modelMeans p c) 1 := by
      rw [twoValue_model, twoAction_model_one]
      linarith only [hslack, hbnd, hbudget.2.1]
    exact Draft0.ThreeArmComparison.deficient_action_never_chosen (T - t)
      (modelCounts c) (modelMeans p c) 1 hpositive hgap hchoose

end

end Draft0.EvenDynamicPolicy
