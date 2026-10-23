import Draft0.DynamicGain
import Draft0.GaussianTail
import Draft0.TerminalDrift

open MeasureTheory ProbabilityTheory
open Filter Asymptotics
open scoped NNReal ENNReal

namespace Draft0.DynamicPolicy

open Draft0.Counts Draft0.Drift Draft0.Gaussian Draft0.Rates
open Draft0.TerminalBounds Draft0.PredictiveBound Draft0.PredictiveLaw
open Draft0.DynamicGain

noncomputable section

def modelCounts (c : Counts) : Draft0.BayesianModel.Counts :=
  fun i => if i = 0 then c.n1 else if i = 1 then c.n2 else c.n3

def modelMeans (p : MeanPath) (c : Counts) : Draft0.BayesianModel.Means :=
  fun i => if i = 0 then p 1 c.n1 else if i = 1 then p 2 c.n2 else p 3 c.n3

def decision (T t : ℕ) (p : MeanPath) (c : Counts) : Draft0.BayesianModel.Arm :=
  Draft0.BayesianModel.choose (T - t) (modelCounts c) (modelMeans p c)

def admissible (q : ℕ) (c : Counts) (a : Draft0.BayesianModel.Arm) : Prop :=
  a ≠ 2 ∧ (a = 0 → c.n1 < q) ∧ (a = 1 → c.n2 < q)

def gainCoefficient : ℝ := min (gainConstant * lowerConstant) lowerConstant
def comparisonCoefficient : ℝ := min (comparisonConstant * lowerConstant) (lowerConstant / 2)

theorem gainCoefficient_pos : 0 < gainCoefficient := by
  exact lt_min (mul_pos gainConstant_positive lowerConstant_pos) lowerConstant_pos

theorem comparisonCoefficient_pos : 0 < comparisonCoefficient := by
  exact lt_min (mul_pos comparisonConstant_positive lowerConstant_pos) (half_pos lowerConstant_pos)

theorem modelCounts_positive (t : ℕ) (c : Counts) (hc : horizonConsistent t c) :
    ∀ i, 0 < modelCounts c i := by
  intro i
  have h1 := hc.1
  have h2 := hc.2.1
  have h3 := hc.2.2.1
  unfold modelCounts
  split_ifs <;> omega

theorem twoValue_model (k : ℕ) (p : MeanPath) (c : Counts) :
    Draft0.ThreeArmComparison.twoValue k (modelCounts c) (modelMeans p c) =
      Draft0.TwoArmBellman.payoff k (p 1 c.n1) (p 2 c.n2) c := by
  simp only [Draft0.ThreeArmComparison.twoValue, Draft0.ThreeArmComparison.projectCounts,
    modelMeans, modelCounts]
  norm_num
  cases c with
  | mk n1 n2 n3 => exact Draft0.TwoArmBellman.payoff_third_irrelevant k _ _ n1 n2 0 n3

theorem twoAction_model_zero (k : ℕ) (p : MeanPath) (c : Counts) :
    Draft0.ThreeArmComparison.twoAction k (modelCounts c) (modelMeans p c) 0 =
      Draft0.TwoArmBellman.leftAction k (p 1 c.n1) (p 2 c.n2) c := by
  rw [Draft0.ThreeArmComparison.twoAction_zero]
  simp only [Draft0.ThreeArmComparison.projectCounts, modelMeans, modelCounts]
  norm_num
  unfold Draft0.TwoArmBellman.leftAction
  congr 1
  funext z
  cases c with
  | mk n1 n2 n3 => exact Draft0.TwoArmBellman.payoff_third_irrelevant k _ _ (n1 + 1) n2 0 n3

theorem twoAction_model_one (k : ℕ) (p : MeanPath) (c : Counts) :
    Draft0.ThreeArmComparison.twoAction k (modelCounts c) (modelMeans p c) 1 =
      Draft0.TwoArmBellman.rightAction k (p 1 c.n1) (p 2 c.n2) c := by
  rw [Draft0.ThreeArmComparison.twoAction_one]
  simp only [Draft0.ThreeArmComparison.projectCounts, modelMeans, modelCounts]
  norm_num
  unfold Draft0.TwoArmBellman.rightAction
  congr 1
  funext z
  cases c with
  | mk n1 n2 n3 => exact Draft0.TwoArmBellman.payoff_third_irrelevant k _ _ n1 (n2 + 1) 0 n3

theorem slack_bad_state_upper (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : 5 ≤ T) (hc : horizonConsistent t c) (hW : paperW T q 40 p c) :
    Draft0.ThreeArmComparison.stateSlack (modelCounts c) (modelMeans p c) ≤
      2 * invPower 225 T := by
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
  exact Draft0.GaussianTail.bad_state_tail T q c.n1 p hT hc.1 hW.2.2.2.2.2.1
    hW.1 hW.2.1 hW.2.2.2.1 _ hv

theorem forecast_parameters (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 40 p c) :
    0 < accumulatedVariance (T - t) c ∧
      (accumulatedVariance (T - t) c : ℝ) ≤ 2 ∧
      invPower 2 T ≤ (accumulatedVariance (T - t) c : ℝ) ∧
      (terminalGap p c) ^ 2 ≤ 204 * Real.log (T : ℝ) *
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
    rw [invPower, inv_eq_one_div, hV]
    exact lemma17_nonterminal_variance_bound T q t c hT (by omega) ht hpre hc
      hW.2.2.2.2.2.1 hW.2.2.2.2.2.2.1
  have hd : (terminalGap p c) ^ 2 ≤ 204 * Real.log (T : ℝ) *
      (accumulatedVariance (T - t) c : ℝ) := by
    rw [hV]; exact lemma17_fullW_drift_bound T q t 40 p c hT ht hpre hc hW
  exact ⟨hv, hu, hl, hd⟩

theorem nonterminal_horizon_gain (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 40 p c) :
    gainCoefficient * invPower 212 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.payoff (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hpar := forecast_parameters T q t p c hT ht hpre hc hW
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  rw [Draft0.TwoArmGain.payoff_horizon_gain_preterminal q (T - t) _ _ c (by omega)
    h1 h2 hW.2.2.2.2.2.1 hW.2.2.2.2.2.2.1 hs,
    balancedPredictiveLaw_gaussian]
  exact (mul_le_mul_of_nonneg_right (min_le_left _ _) (invPower_nonneg 212 T)).trans
    (nonterminal_hinge_gain_lower T (q - 1) (terminalGap p c) _ (by omega) (by omega)
      (by omega) hpar.1 hpar.2.1 hpar.2.2.1 hpar.2.2.2)

theorem nonterminal_left_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 40 p c) (hsat : c.n1 = q) :
    comparisonCoefficient * invPower 214 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hpar := forecast_parameters T q t p c hT ht hpre hc hW
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  rw [Draft0.TwoArmGain.saturated_left_gap_preterminal q (T - t) _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_gaussian]
  exact (mul_le_mul_of_nonneg_right (min_le_left _ _) (invPower_nonneg 214 T)).trans
    (nonterminal_hinge_difference_lower T q (q - 1) (terminalGap p c) _ (by omega)
      (by omega) (by omega) (by omega) hpar.1 hpar.2.1 hpar.2.2.1 hpar.2.2.2)

theorem nonterminal_right_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q 40 p c) (hsat : c.n2 = q) :
    comparisonCoefficient * invPower 214 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  have hpar := forecast_parameters T q t p c hT ht hpre hc hW
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  rw [Draft0.TwoArmGain.saturated_right_gap_preterminal q (T - t) _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_gaussian]
  exact (mul_le_mul_of_nonneg_right (min_le_left _ _) (invPower_nonneg 214 T)).trans
    (nonterminal_hinge_difference_lower T q (q - 1) (terminalGap p c) _ (by omega)
      (by omega) (by omega) (by omega) hpar.1 hpar.2.1 hpar.2.2.1 hpar.2.2.2)

theorem terminal_horizon_gain (T q : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hT5 : 5 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q 40 p c) :
    gainCoefficient * invPower 212 T ≤
      Draft0.TwoArmBellman.payoff 1 (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.payoff 0 (p 1 c.n1) (p 2 c.n2) c := by
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hq : 2 ≤ q := by omega
  have hs : c.n1 + c.n2 + 0 = 2 * q - 1 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  have hd := Draft0.TerminalDrift.terminal_fullW_drift_bound T q 40 p c hT hT5 hc hW
  have hb := updateVariance_count_bounds q T (by omega) (by omega)
  have hv : 0 < updateVariance (q : ℝ≥0) := by unfold updateVariance; positivity
  have hlower := terminal_hinge_gain_lower T (terminalGap p c) _ hT5 hv
    (by linarith only [hb.2]) hb.1 hd
  have hcount : (0 : ℝ≥0) < (q - 1 : ℕ) := by exact_mod_cast (show 0 < q - 1 by omega)
  have hvar := updateVariance_strictAntitone ((q - 1 : ℕ) : ℝ≥0) (q : ℝ≥0)
    hcount (by exact_mod_cast (show q - 1 < q by omega))
  have hmono := hingeLoss_strictMono_variance |terminalGap p c| (abs_nonneg _)
    (updateVariance (q : ℝ≥0)) (updateVariance ((q - 1 : ℕ) : ℝ≥0)) hvar
  rw [Draft0.TwoArmGain.payoff_horizon_gain_preterminal q 0 _ _ c (by omega)
    h1 h2 hW.2.2.2.2.2.1 hW.2.2.2.2.2.2.1 hs, balancedPredictiveLaw_terminal,
    integral_dirac]
  exact (mul_le_mul (min_le_right _ _) (invPower_antitone (by omega : 208 ≤ 212) (by omega))
    (invPower_nonneg 212 T) lowerConstant_pos.le).trans (hlower.trans hmono.le)

theorem terminal_left_gap (T q : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hT5 : 5 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q 40 p c) (hsat : c.n1 = q) :
    comparisonCoefficient * invPower 214 T ≤
      Draft0.TwoArmBellman.payoff 1 (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.leftAction 0 (p 1 c.n1) (p 2 c.n2) c := by
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hq : 2 ≤ q := by omega
  have hs : c.n1 + c.n2 + 0 = 2 * q - 1 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  have hd := Draft0.TerminalDrift.terminal_fullW_drift_bound T q 40 p c hT hT5 hc hW
  have hlower := terminal_hinge_difference_lower T q (q - 1) (terminalGap p c)
    hT5 (by omega) (by omega) (by omega) hd
  rw [Draft0.TwoArmGain.saturated_left_gap_preterminal q 0 _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_terminal, integral_dirac]
  exact (mul_le_mul (min_le_right _ _) (invPower_antitone (by omega : 211 ≤ 214) (by omega))
    (invPower_nonneg 214 T) (half_pos lowerConstant_pos).le).trans hlower

theorem terminal_right_gap (T q : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hT5 : 5 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q 40 p c) (hsat : c.n2 = q) :
    comparisonCoefficient * invPower 214 T ≤
      Draft0.TwoArmBellman.payoff 1 (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.rightAction 0 (p 1 c.n1) (p 2 c.n2) c := by
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hq : 2 ≤ q := by omega
  have hs : c.n1 + c.n2 + 0 = 2 * q - 1 := by
    rcases hc with ⟨_, _, h3, htotal⟩; omega
  have hd := Draft0.TerminalDrift.terminal_fullW_drift_bound T q 40 p c hT hT5 hc hW
  have hlower := terminal_hinge_difference_lower T q (q - 1) (terminalGap p c)
    hT5 (by omega) (by omega) (by omega) hd
  rw [Draft0.TwoArmGain.saturated_right_gap_preterminal q 0 _ _ c (by omega)
    h1 h2 hsat hs, balancedPredictiveLaw_terminal, integral_dirac]
  exact (mul_le_mul (min_le_right _ _) (invPower_antitone (by omega : 211 ≤ 214) (by omega))
    (invPower_nonneg 214 T) (half_pos lowerConstant_pos).le).trans hlower

theorem all_round_horizon_gain (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hT5 : 5 ≤ T) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (hW : paperW T q 40 p c) :
    gainCoefficient * invPower 212 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.payoff (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  by_cases hpre : t < T
  · exact nonterminal_horizon_gain T q t p c hT ht hpre hc hW
  · have heq : t = T := by omega
    subst t
    simpa only [Nat.sub_self, zero_add] using terminal_horizon_gain T q p c hT hT5 hc hW

theorem all_round_left_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hT5 : 5 ≤ T) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (hW : paperW T q 40 p c) (hsat : c.n1 = q) :
    comparisonCoefficient * invPower 214 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.leftAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  by_cases hpre : t < T
  · exact nonterminal_left_gap T q t p c hT ht hpre hc hW hsat
  · have heq : t = T := by omega
    subst t
    simpa only [Nat.sub_self, zero_add] using terminal_left_gap T q p c hT hT5 hc hW hsat

theorem all_round_right_gap (T q t : ℕ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hT5 : 5 ≤ T) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (hW : paperW T q 40 p c) (hsat : c.n2 = q) :
    comparisonCoefficient * invPower 214 T ≤
      Draft0.TwoArmBellman.payoff (T - t + 1) (p 1 c.n1) (p 2 c.n2) c -
      Draft0.TwoArmBellman.rightAction (T - t) (p 1 c.n1) (p 2 c.n2) c := by
  by_cases hpre : t < T
  · exact nonterminal_right_gap T q t p c hT ht hpre hc hW hsat
  · have heq : t = T := by omega
    subst t
    simpa only [Nat.sub_self, zero_add] using terminal_right_gap T q p c hT hT5 hc hW hsat

/-- A single budget threshold controls all W states and all decision rounds, including T. -/
theorem eventually_all_W_admissible :
    ∃ T0 : ℕ, ∀ T : ℕ, T0 ≤ T → ∀ (q t : ℕ) (p : MeanPath) (c : Counts),
      T = 2 * q + 1 → 4 ≤ t → t ≤ T → horizonConsistent t c →
      paperW T q 40 p c → admissible q c (decision T t p c) := by
  have herr : (fun T => 2 * invPower 225 T) =O[atTop] invPower 225 :=
    (isBigO_refl (invPower 225) atTop).const_mul_left 2
  have hg := eventual_subtraction_positive gainCoefficient_pos (by norm_num : (0 : ℝ) < 1)
    (by omega : 212 < 225) herr (Filter.Eventually.of_forall (fun T => le_rfl))
  have hb := eventual_subtraction_positive comparisonCoefficient_pos (by norm_num : (0 : ℝ) < 1)
    (by omega : 214 < 225) herr (Filter.Eventually.of_forall (fun T => le_rfl))
  obtain ⟨T0, hT0⟩ := Filter.eventually_atTop.mp
    (hg.and (hb.and (eventually_ge_atTop (5 : ℕ))))
  refine ⟨T0, ?_⟩
  intro T hlarge q t p c hT ht horizon hc hW
  have hbudget := hT0 T hlarge
  have hT5 : 5 ≤ T := hbudget.2.2
  have hslack := slack_bad_state_upper T q t p c hT5 hc hW
  have hpositive := modelCounts_positive t c hc
  have hgain := all_round_horizon_gain T q t p c hT hT5 ht horizon hc hW
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
    have hbnd := all_round_left_gap T q t p c hT hT5 ht horizon hc hW hsat
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
    have hbnd := all_round_right_gap T q t p c hT hT5 ht horizon hc hW hsat
    have hgap : Draft0.ThreeArmComparison.stateSlack (modelCounts c) (modelMeans p c) <
        Draft0.ThreeArmComparison.twoValue (T - t + 1) (modelCounts c) (modelMeans p c) -
        Draft0.ThreeArmComparison.twoAction (T - t) (modelCounts c) (modelMeans p c) 1 := by
      rw [twoValue_model, twoAction_model_one]
      linarith only [hslack, hbnd, hbudget.2.1]
    exact Draft0.ThreeArmComparison.deficient_action_never_chosen (T - t)
      (modelCounts c) (modelMeans p c) 1 hpositive hgap hchoose

end

end Draft0.DynamicPolicy
