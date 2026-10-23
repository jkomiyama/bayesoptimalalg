import Draft0.Gaussian
import Draft0.BayesianModel
import Draft0.TwoArmBellman

/-!
Perfect-information slack for adding a third arm to the two-arm Bellman
problem. The slack is the genuine posterior expectation of the positive
part of the third arm's advantage over the first. It is preserved in
expectation by either arm's Gaussian posterior innovation.
-/

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.ThreeArmComparison

open Draft0.Gaussian

noncomputable section

theorem hingeLoss_shift_integrable (center : ℝ) (innovation residual : ℝ≥0) :
    Integrable (fun z : ℝ => hingeLoss (center - z) residual)
      (gaussianReal 0 innovation) := by
  have hconv : gaussianReal 0 innovation ∗ gaussianReal 0 residual =
      gaussianReal 0 (innovation + residual) := by
    simpa using gaussianReal_conv_gaussianReal (m₁ := (0 : ℝ)) (m₂ := (0 : ℝ))
      (v₁ := innovation) (v₂ := residual)
  have hi : Integrable (fun x : ℝ => max (x - center) 0)
      (gaussianReal 0 innovation ∗ gaussianReal 0 residual) := by
    rw [hconv]
    exact hinge_integrable center (innovation + residual)
  have h := (integrable_conv_iff hi.aestronglyMeasurable).mp hi
  have hf : (fun z : ℝ => ∫ r : ℝ, ‖max (z + r - center) 0‖
      ∂gaussianReal 0 residual) = (fun z : ℝ => hingeLoss (center - z) residual) := by
    funext z
    unfold hingeLoss
    congr 1
    funext r
    rw [Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
    congr 1
    ring
  rw [hf] at h
  exact h.2

theorem hingeLoss_shift_tower (center : ℝ) (innovation residual : ℝ≥0) :
    (∫ z : ℝ, hingeLoss (center - z) residual ∂gaussianReal 0 innovation) =
      hingeLoss center (innovation + residual) := by
  have hconv : gaussianReal 0 innovation ∗ gaussianReal 0 residual =
      gaussianReal 0 (innovation + residual) := by
    simpa using gaussianReal_conv_gaussianReal (m₁ := (0 : ℝ)) (m₂ := (0 : ℝ))
      (v₁ := innovation) (v₂ := residual)
  have hi : Integrable (fun x : ℝ => max (x - center) 0)
      (gaussianReal 0 innovation ∗ gaussianReal 0 residual) := by
    rw [hconv]
    exact hinge_integrable center (innovation + residual)
  unfold hingeLoss
  rw [← hconv, integral_conv hi]
  congr 1
  funext z
  congr 1
  funext r
  congr 1
  ring

theorem hingeLoss_add_integrable (center : ℝ) (innovation residual : ℝ≥0) :
    Integrable (fun z : ℝ => hingeLoss (center + z) residual)
      (gaussianReal 0 innovation) := by
  have hsym : (gaussianReal 0 innovation).map (fun z : ℝ => -z) =
      gaussianReal 0 innovation := by simpa using gaussianReal_map_neg (μ := (0 : ℝ))
  have h := hingeLoss_shift_integrable center innovation residual
  rw [← hsym] at h
  have hh := h.comp_measurable (by fun_prop)
  simpa only [Function.comp_def, sub_neg_eq_add] using hh

theorem hingeLoss_add_tower (center : ℝ) (innovation residual : ℝ≥0) :
    (∫ z : ℝ, hingeLoss (center + z) residual ∂gaussianReal 0 innovation) =
      hingeLoss center (innovation + residual) := by
  have hsym : (gaussianReal 0 innovation).map (fun z : ℝ => -z) =
      gaussianReal 0 innovation := by simpa using gaussianReal_map_neg (μ := (0 : ℝ))
  have hi := hingeLoss_shift_integrable center innovation residual
  have hm := hi.aestronglyMeasurable
  rw [← hsym] at hm
  have htower := hingeLoss_shift_tower center innovation residual
  calc
    (∫ z : ℝ, hingeLoss (center + z) residual ∂gaussianReal 0 innovation) =
        ∫ z : ℝ, hingeLoss (center - z) residual
          ∂((gaussianReal 0 innovation).map (fun z : ℝ => -z)) := by
      rw [integral_map (by fun_prop) hm]
      simp only [sub_neg_eq_add]
    _ = hingeLoss center (innovation + residual) := by rw [hsym]; exact htower

theorem positivePart_le_hingeLoss (center : ℝ) (variance : ℝ≥0) :
    max (-center) 0 ≤ hingeLoss center variance := by
  apply max_le
  · have h := hingeLoss_reflection center variance
    have hn := hingeLoss_nonnegative (-center) variance
    linarith
  · exact hingeLoss_nonnegative center variance

/-- Genuine posterior expected positive advantage of arm 3 over arm 1. -/
def slack (mean1 mean3 : ℝ) (count1 count3 : ℕ) : ℝ :=
  hingeLoss (mean1 - mean3) (posteriorVariance (count1 : ℝ≥0) (count3 : ℝ≥0))

theorem slack_nonnegative (mean1 mean3 : ℝ) (count1 count3 : ℕ) :
    0 ≤ slack mean1 mean3 count1 count3 := hingeLoss_nonnegative _ _

theorem slack_bounds_terminal (mean1 mean2 mean3 : ℝ) (count1 count3 : ℕ) :
    max mean1 (max mean2 mean3) ≤
      max mean1 mean2 + slack mean1 mean3 count1 count3 := by
  have hs := positivePart_le_hingeLoss (mean1 - mean3)
    (posteriorVariance (count1 : ℝ≥0) (count3 : ℝ≥0))
  change max (-(mean1 - mean3)) 0 ≤ slack mean1 mean3 count1 count3 at hs
  have hs0 := slack_nonnegative mean1 mean3 count1 count3
  have hm1 := le_max_left mean1 mean2
  have hm2 := le_max_right mean1 mean2
  have hm3 := le_max_left (-(mean1 - mean3)) (0 : ℝ)
  apply max_le
  · linarith
  · apply max_le <;> linarith

theorem slack_first_draw_integrable (mean1 mean3 : ℝ) (count1 count3 : ℕ) :
    Integrable (fun z : ℝ => slack (mean1 + z) mean3 (count1 + 1) count3)
      (gaussianReal 0 (updateVariance (count1 : ℝ≥0))) := by
  have h := hingeLoss_add_integrable (mean1 - mean3) (updateVariance (count1 : ℝ≥0))
    (posteriorVariance ((count1 + 1 : ℕ) : ℝ≥0) (count3 : ℝ≥0))
  unfold slack
  convert h using 1
  funext z
  congr 1
  ring

theorem slack_first_draw_tower (mean1 mean3 : ℝ) (count1 count3 : ℕ)
    (hc1 : 0 < count1) :
    (∫ z : ℝ, slack (mean1 + z) mean3 (count1 + 1) count3
      ∂gaussianReal 0 (updateVariance (count1 : ℝ≥0))) =
        slack mean1 mean3 count1 count3 := by
  have h := hingeLoss_add_tower (mean1 - mean3) (updateVariance (count1 : ℝ≥0))
    (posteriorVariance ((count1 + 1 : ℕ) : ℝ≥0) (count3 : ℝ≥0))
  have hv := posteriorVariance_after_draw (count1 : ℝ≥0) (count3 : ℝ≥0)
    (by exact_mod_cast hc1)
  push_cast at h
  rw [← hv] at h
  unfold slack
  push_cast
  convert h using 1
  congr 1
  funext z
  congr 1
  ring

theorem slack_third_draw_integrable (mean1 mean3 : ℝ) (count1 count3 : ℕ) :
    Integrable (fun z : ℝ => slack mean1 (mean3 + z) count1 (count3 + 1))
      (gaussianReal 0 (updateVariance (count3 : ℝ≥0))) := by
  have h := hingeLoss_shift_integrable (mean1 - mean3) (updateVariance (count3 : ℝ≥0))
    (posteriorVariance (count1 : ℝ≥0) ((count3 + 1 : ℕ) : ℝ≥0))
  unfold slack
  convert h using 1
  funext z
  congr 1
  ring

theorem slack_third_draw_tower (mean1 mean3 : ℝ) (count1 count3 : ℕ)
    (hc3 : 0 < count3) :
    (∫ z : ℝ, slack mean1 (mean3 + z) count1 (count3 + 1)
      ∂gaussianReal 0 (updateVariance (count3 : ℝ≥0))) =
        slack mean1 mean3 count1 count3 := by
  have h := hingeLoss_shift_tower (mean1 - mean3) (updateVariance (count3 : ℝ≥0))
    (posteriorVariance (count1 : ℝ≥0) ((count3 + 1 : ℕ) : ℝ≥0))
  have hv := posteriorVariance_after_draw (count3 : ℝ≥0) (count1 : ℝ≥0)
    (by exact_mod_cast hc3)
  push_cast at h
  have hvdesired : updateVariance (count3 : ℝ≥0) +
      posteriorVariance (count1 : ℝ≥0) ((count3 : ℝ≥0) + 1) =
        posteriorVariance (count1 : ℝ≥0) (count3 : ℝ≥0) := by
    rw [posteriorVariance_comm (count1 : ℝ≥0) ((count3 : ℝ≥0) + 1), ← hv,
      posteriorVariance_comm (count3 : ℝ≥0) (count1 : ℝ≥0)]
  rw [hvdesired] at h
  unfold slack
  push_cast
  convert h using 1
  congr 1
  funext z
  congr 1
  ring

abbrev ModelCounts := Draft0.BayesianModel.Counts
abbrev ModelMeans := Draft0.BayesianModel.Means
abbrev ModelArm := Draft0.BayesianModel.Arm

def projectCounts (c : ModelCounts) : Draft0.Counts.Counts := ⟨c 0, c 1, 0⟩

def twoValue (k : ℕ) (c : ModelCounts) (m : ModelMeans) : ℝ :=
  Draft0.TwoArmBellman.payoff k (m 0) (m 1) (projectCounts c)

def twoAction (k : ℕ) (c : ModelCounts) (m : ModelMeans) (i : ModelArm) : ℝ :=
  ∫ z : ℝ, twoValue k (Draft0.BayesianModel.advance c i)
      (Draft0.BayesianModel.shift m i z)
    ∂gaussianReal 0 (Draft0.BayesianModel.innovation c i)

def stateSlack (c : ModelCounts) (m : ModelMeans) : ℝ := slack (m 0) (m 2) (c 0) (c 2)

theorem twoAction_zero (k : ℕ) (c : ModelCounts) (m : ModelMeans) :
    twoAction k c m 0 = ∫ z : ℝ,
      Draft0.TwoArmBellman.payoff k (m 0 + z) (m 1)
        (Draft0.TwoArmBellman.leftStep (projectCounts c))
      ∂gaussianReal 0 (updateVariance (c 0 : ℝ≥0)) := by
  simp [twoAction, twoValue, projectCounts, Draft0.TwoArmBellman.leftStep,
    Draft0.BayesianModel.advance, Draft0.BayesianModel.shift, Draft0.BayesianModel.innovation]

theorem twoAction_one (k : ℕ) (c : ModelCounts) (m : ModelMeans) :
    twoAction k c m 1 = ∫ z : ℝ,
      Draft0.TwoArmBellman.payoff k (m 0) (m 1 + z)
        (Draft0.TwoArmBellman.rightStep (projectCounts c))
      ∂gaussianReal 0 (updateVariance (c 1 : ℝ≥0)) := by
  simp [twoAction, twoValue, projectCounts, Draft0.TwoArmBellman.rightStep,
    Draft0.BayesianModel.advance, Draft0.BayesianModel.shift, Draft0.BayesianModel.innovation]

theorem twoAction_two (k : ℕ) (c : ModelCounts) (m : ModelMeans) :
    twoAction k c m 2 = twoValue k c m := by
  simp [twoAction, twoValue, projectCounts, Draft0.BayesianModel.advance,
    Draft0.BayesianModel.shift]

theorem twoValue_succ (k : ℕ) (c : ModelCounts) (m : ModelMeans) :
    twoValue (k + 1) c m = max (twoAction k c m 0) (twoAction k c m 1) := by
  rw [twoAction_zero, twoAction_one]
  rfl

theorem twoAction_integrable (k : ℕ) (c : ModelCounts) (m : ModelMeans) (i : ModelArm) :
    Integrable (fun z : ℝ => twoValue k (Draft0.BayesianModel.advance c i)
      (Draft0.BayesianModel.shift m i z))
      (gaussianReal 0 (Draft0.BayesianModel.innovation c i)) := by
  fin_cases i
  · simpa [twoValue, projectCounts, Draft0.BayesianModel.advance,
      Draft0.BayesianModel.shift, Draft0.BayesianModel.innovation,
      Draft0.TwoArmBellman.leftStep] using
      Draft0.TwoArmBellman.left_action_integrable k (m 0) (m 1) (projectCounts c)
  · simpa [twoValue, projectCounts, Draft0.BayesianModel.advance,
      Draft0.BayesianModel.shift, Draft0.BayesianModel.innovation,
      Draft0.TwoArmBellman.rightStep] using
      Draft0.TwoArmBellman.right_action_integrable k (m 0) (m 1) (projectCounts c)
  · simp [twoValue, projectCounts, Draft0.BayesianModel.advance,
      Draft0.BayesianModel.shift]

theorem twoAction_le_next_value (k : ℕ) (c : ModelCounts) (m : ModelMeans) (i : ModelArm) :
    twoAction k c m i ≤ twoValue (k + 1) c m := by
  fin_cases i
  · rw [twoValue_succ]; exact le_max_left _ _
  · rw [twoValue_succ]; exact le_max_right _ _
  · change twoAction k c m 2 ≤ twoValue (k + 1) c m
    rw [twoAction_two]
    exact Draft0.TwoArmBellman.payoff_horizon_mono k (m 0) (m 1) (projectCounts c)

theorem stateSlack_integrable (c : ModelCounts) (m : ModelMeans) (i : ModelArm) :
    Integrable (fun z : ℝ => stateSlack (Draft0.BayesianModel.advance c i)
      (Draft0.BayesianModel.shift m i z))
      (gaussianReal 0 (Draft0.BayesianModel.innovation c i)) := by
  fin_cases i
  · simpa [stateSlack, Draft0.BayesianModel.advance, Draft0.BayesianModel.shift,
      Draft0.BayesianModel.innovation] using slack_first_draw_integrable (m 0) (m 2) (c 0) (c 2)
  · simp [stateSlack, Draft0.BayesianModel.advance, Draft0.BayesianModel.shift]
  · simpa [stateSlack, Draft0.BayesianModel.advance, Draft0.BayesianModel.shift,
      Draft0.BayesianModel.innovation] using slack_third_draw_integrable (m 0) (m 2) (c 0) (c 2)

theorem stateSlack_tower (c : ModelCounts) (m : ModelMeans) (i : ModelArm)
    (hc : ∀ j, 0 < c j) :
    (∫ z : ℝ, stateSlack (Draft0.BayesianModel.advance c i)
      (Draft0.BayesianModel.shift m i z)
      ∂gaussianReal 0 (Draft0.BayesianModel.innovation c i)) = stateSlack c m := by
  fin_cases i
  · simpa [stateSlack, Draft0.BayesianModel.advance, Draft0.BayesianModel.shift,
      Draft0.BayesianModel.innovation] using
      slack_first_draw_tower (m 0) (m 2) (c 0) (c 2) (hc 0)
  · simp [stateSlack, Draft0.BayesianModel.advance, Draft0.BayesianModel.shift]
  · simpa [stateSlack, Draft0.BayesianModel.advance, Draft0.BayesianModel.shift,
      Draft0.BayesianModel.innovation] using
      slack_third_draw_tower (m 0) (m 2) (c 0) (c 2) (hc 2)

theorem positive_advance (c : ModelCounts) (hc : ∀ j, 0 < c j) (i : ModelArm) :
    ∀ j, 0 < Draft0.BayesianModel.advance c i j := by
  intro j
  have h := hc j
  unfold Draft0.BayesianModel.advance
  omega

/-- Exact dynamic-programming perturbation bound, derived by Bellman induction. -/
theorem value_comparison (k : ℕ) (c : ModelCounts) (m : ModelMeans)
    (hc : ∀ j, 0 < c j) :
    twoValue k c m ≤ Draft0.BayesianModel.value k c m ∧
      Draft0.BayesianModel.value k c m ≤ twoValue k c m + stateSlack c m := by
  induction k generalizing c m with
  | zero =>
      constructor
      · change max (m 0) (m 1) ≤ max (m 0) (max (m 1) (m 2))
        exact max_le_max le_rfl (le_max_left _ _)
      · exact slack_bounds_terminal (m 0) (m 1) (m 2) (c 0) (c 2)
  | succ k ih =>
      have ha (i : ModelArm) :
          twoAction k c m i ≤ Draft0.BayesianModel.action k c m i ∧
          Draft0.BayesianModel.action k c m i ≤ twoAction k c m i + stateSlack c m := by
        have hlower := integral_mono (twoAction_integrable k c m i)
          (Draft0.BayesianModel.action_integrable k c m i)
          (fun z => (ih (Draft0.BayesianModel.advance c i)
            (Draft0.BayesianModel.shift m i z) (positive_advance c hc i)).1)
        have hupper := integral_mono (Draft0.BayesianModel.action_integrable k c m i)
          ((twoAction_integrable k c m i).add (stateSlack_integrable c m i))
          (fun z => (ih (Draft0.BayesianModel.advance c i)
            (Draft0.BayesianModel.shift m i z) (positive_advance c hc i)).2)
        simp only [Pi.add_apply] at hupper
        rw [integral_add (twoAction_integrable k c m i) (stateSlack_integrable c m i),
          stateSlack_tower c m i hc] at hupper
        exact ⟨hlower, hupper⟩
      constructor
      · rw [twoValue_succ]
        exact max_le ((ha 0).1.trans (Draft0.BayesianModel.action_le_value k c m 0))
          ((ha 1).1.trans (Draft0.BayesianModel.action_le_value k c m 1))
      · rw [Draft0.BayesianModel.value_succ]
        have hb (i : ModelArm) : Draft0.BayesianModel.action k c m i ≤
            twoValue (k + 1) c m + stateSlack c m :=
          (ha i).2.trans (add_le_add_right (twoAction_le_next_value k c m i) _)
        exact max_le (hb 0) (max_le (hb 1) (hb 2))

/-- Every actual three-arm Bellman action is between its binary value and that value plus slack. -/
theorem action_comparison (k : ℕ) (c : ModelCounts) (m : ModelMeans) (i : ModelArm)
    (hc : ∀ j, 0 < c j) :
    twoAction k c m i ≤ Draft0.BayesianModel.action k c m i ∧
      Draft0.BayesianModel.action k c m i ≤ twoAction k c m i + stateSlack c m := by
  have hlower := integral_mono (twoAction_integrable k c m i)
    (Draft0.BayesianModel.action_integrable k c m i)
    (fun z => (value_comparison k (Draft0.BayesianModel.advance c i)
      (Draft0.BayesianModel.shift m i z) (positive_advance c hc i)).1)
  have hupper := integral_mono (Draft0.BayesianModel.action_integrable k c m i)
    ((twoAction_integrable k c m i).add (stateSlack_integrable c m i))
    (fun z => (value_comparison k (Draft0.BayesianModel.advance c i)
      (Draft0.BayesianModel.shift m i z) (positive_advance c hc i)).2)
  simp only [Pi.add_apply] at hupper
  rw [integral_add (twoAction_integrable k c m i) (stateSlack_integrable c m i),
    stateSlack_tower c m i hc] at hupper
  exact ⟨hlower, hupper⟩

theorem third_never_chosen (k : ℕ) (c : ModelCounts) (m : ModelMeans)
    (hc : ∀ j, 0 < c j)
    (hgain : stateSlack c m < twoValue (k + 1) c m - twoValue k c m) :
    Draft0.BayesianModel.choose k c m ≠ 2 := by
  intro hchoose
  have hopt := Draft0.BayesianModel.choose_optimal k c m
  rw [hchoose] at hopt
  have hb := (action_comparison k c m 2 hc).2
  rw [twoAction_two] at hb
  have hlower := (value_comparison (k + 1) c m hc).1
  linarith

theorem inferior_action_never_chosen (k : ℕ) (c : ModelCounts) (m : ModelMeans)
    (i j : ModelArm) (hc : ∀ a, 0 < c a)
    (hgap : stateSlack c m < twoAction k c m j - twoAction k c m i) :
    Draft0.BayesianModel.choose k c m ≠ i := by
  intro hchoose
  have hopt := Draft0.BayesianModel.choose_optimal k c m
  rw [hchoose] at hopt
  have hb := (action_comparison k c m i hc).2
  have hlower := (action_comparison k c m j hc).1
  have hj := Draft0.BayesianModel.action_le_value k c m j
  linarith

theorem deficient_action_never_chosen (k : ℕ) (c : ModelCounts) (m : ModelMeans)
    (i : ModelArm) (hc : ∀ a, 0 < c a)
    (hgap : stateSlack c m < twoValue (k + 1) c m - twoAction k c m i) :
    Draft0.BayesianModel.choose k c m ≠ i := by
  intro hchoose
  have hopt := Draft0.BayesianModel.choose_optimal k c m
  rw [hchoose] at hopt
  have hb := (action_comparison k c m i hc).2
  have hlower := (value_comparison (k + 1) c m hc).1
  linarith

end

end Draft0.ThreeArmComparison
