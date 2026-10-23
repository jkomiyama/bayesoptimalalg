import Junpei.Dynamic
import Junpei.ExclusionBounds
import Junpei.ExclusionPerm
import Junpei.MarginalTerminal
import Junpei.ForecastTerminal

open MeasureTheory ProbabilityTheory Filter Asymptotics
open scoped NNReal ENNReal

namespace Junpei.PolicyChoice

open Draft0.Counts Draft0.DynamicPolicy Draft0.Rates Draft0.Gaussian
open Draft0.TerminalBounds Draft0.PredictiveBound Draft0.ThreeArmComparison
open Junpei.Exclusion Junpei.ExclusionPerm Junpei.ExclusionBounds Junpei.ExclusionTail
open JunpeiComparison

noncomputable section

def gainC : ℝ := gainConstant * lowerConstant
def comparisonC : ℝ := comparisonConstant * lowerConstant

theorem gainC_pos : 0 < gainC := mul_pos gainConstant_positive lowerConstant_pos
theorem comparisonC_pos : 0 < comparisonC := mul_pos comparisonConstant_positive lowerConstant_pos

theorem lemma6_upper (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : 5 ≤ T) (hCU : 0 < CU) (hcut : 180 < (CU - 6)^2 / 4)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) :
    EBI (T - t) (modelCounts c) (modelMeans p c) 2 ≤ paperG0 T CU :=
  (lemma12_k3 _ _ _ _ (modelCounts_positive t c hc)).trans
    (lemma11_from_equation11 T q t CU p c hT hCU hcut hc hW)

theorem binary_EBI_lower (k : ℕ) (c : Draft0.BayesianModel.Counts)
    (m : Draft0.BayesianModel.Means) (i : Draft0.BayesianModel.Arm)
    (hc : ∀ j, 0 < c j) :
    twoAction k c m i - twoValue k c m - exclusion c m 2 ≤ EBI k c m i := by
  rw [EBI_eq k c m i (hc i)]
  have ha := (action_comparison k c m i hc).1
  have hv := (value_comparison k c m hc).2
  have hs := stateSlack_le_exclusion c m
  linarith only [ha, hv, hs]

theorem binary_action_gap_lower (k : ℕ) (c : Draft0.BayesianModel.Counts)
    (m : Draft0.BayesianModel.Means) (i j : Draft0.BayesianModel.Arm)
    (hc : ∀ a, 0 < c a) :
    twoAction k c m j - twoAction k c m i - exclusion c m 2 ≤
      Draft0.BayesianModel.action k c m j - Draft0.BayesianModel.action k c m i := by
  have hi := (action_comparison k c m i hc).2
  have hj := (action_comparison k c m j hc).1
  have hs := stateSlack_le_exclusion c m
  linarith only [hi, hj, hs]

theorem lemma7_individual_lower (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2*q+1) (ht : 4 ≤ t) (hpre : t < T) (hCU : 10 ≤ CU)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) :
    gainC * invPower 176 T - paperG0 T CU ≤ EBI (T-t) (modelCounts c) (modelMeans p c) 0 ∧
    gainC * invPower 176 T - paperG0 T CU ≤ EBI (T-t) (modelCounts c) (modelMeans p c) 1 := by
  have hg := Junpei.Dynamic.lemma19_individual_gains T q t CU p c hT ht hpre hc hW
  have he := lemma11 T q t CU p c (by omega) hCU hc hW
  have h0 := binary_EBI_lower (T-t) (modelCounts c) (modelMeans p c) 0
    (modelCounts_positive t c hc)
  have h1 := binary_EBI_lower (T-t) (modelCounts c) (modelMeans p c) 1
    (modelCounts_positive t c hc)
  rw [twoAction_model_zero, twoValue_model] at h0
  rw [twoAction_model_one, twoValue_model] at h1
  change gainC * invPower 176 T ≤ _ ∧ gainC * invPower 176 T ≤ _ at hg
  constructor <;> linarith only [hg.1, hg.2, he, h0, h1]

theorem lemma7_saturated_left (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2*q+1) (ht : 4 ≤ t) (hpre : t < T) (hCU : 10 ≤ CU)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) (hsat : c.n1 = q) :
    comparisonC * invPower 178 T - paperG0 T CU ≤
      Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 1 -
      Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 0 := by
  have hg := Junpei.Dynamic.lemma19_saturated_left_gap T q t CU p c hT ht hpre hc hW hsat
  have he := lemma11 T q t CU p c (by omega) hCU hc hW
  have h := binary_action_gap_lower (T-t) (modelCounts c) (modelMeans p c) 0 1
    (modelCounts_positive t c hc)
  rw [twoAction_model_zero, twoAction_model_one] at h
  change comparisonC * invPower 178 T ≤ _ at hg
  linarith only [hg, he, h]

theorem lemma7_saturated_right (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2*q+1) (ht : 4 ≤ t) (hpre : t < T) (hCU : 10 ≤ CU)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) (hsat : c.n2 = q) :
    comparisonC * invPower 178 T - paperG0 T CU ≤
      Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 0 -
      Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 1 := by
  have hg := Junpei.Dynamic.lemma19_saturated_right_gap T q t CU p c hT ht hpre hc hW hsat
  have he := lemma11 T q t CU p c (by omega) hCU hc hW
  have h := binary_action_gap_lower (T-t) (modelCounts c) (modelMeans p c) 1 0
    (modelCounts_positive t c hc)
  rw [twoAction_model_zero, twoAction_model_one] at h
  change comparisonC * invPower 178 T ≤ _ at hg
  linarith only [hg, he, h]

theorem terminal_pair (T q : ℕ) (c : Counts) (hT : T = 2*q+1)
    (hc : horizonConsistent T c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    (c.n1 = q ∧ c.n2 = q-1) ∨ (c.n1 = q-1 ∧ c.n2 = q) := by
  rcases hc with ⟨_, _, h3, hs⟩
  omega

theorem lemma8_individual_lower (T q : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2*q+1) (hlarge : 9 ≤ T) (hCU : 10 ≤ CU)
    (hc : horizonConsistent T c) (hW : newW T q CU p c) :
    Junpei.Marginal.terminalConstant * invPower 22 T - paperG0 T CU ≤
      EBI 0 (modelCounts c) (modelMeans p c) 0 ∧
    Junpei.Marginal.terminalConstant * invPower 22 T - paperG0 T CU ≤
      EBI 0 (modelCounts c) (modelMeans p c) 1 := by
  have hpair := terminal_pair T q c hT hc hW.2.2.2.2.2.1 hW.2.2.2.2.2.2.1
  have hd := Junpei.Forecast.terminal_gap_bound T q CU p c hT hlarge hW hpair
  have hg1 := Junpei.Marginal.lemma15_terminal_uniform22 T q c.n1 (terminalGap p c) 0
    (by omega) hT hc.1 hW.2.2.2.2.2.1 hd
  have hg2 := Junpei.Marginal.lemma15_terminal_uniform22 T q c.n2 (terminalGap p c) 0
    (by omega) hT hc.2.1 hW.2.2.2.2.2.2.1 hd
  rw [oneStepImprovement_eq_hinge] at hg1 hg2
  have he := lemma11 T q T CU p c (by omega) hCU hc hW
  have h0 := binary_EBI_lower 0 (modelCounts c) (modelMeans p c) 0 (modelCounts_positive T c hc)
  have h1 := binary_EBI_lower 0 (modelCounts c) (modelMeans p c) 1 (modelCounts_positive T c hc)
  rw [twoAction_model_zero, twoValue_model, Draft0.TwoArmBellman.leftAction_eq,
    Draft0.TwoArmBellman.payoff_eq] at h0
  rw [twoAction_model_one, twoValue_model, Draft0.TwoArmBellman.rightAction_eq,
    Draft0.TwoArmBellman.payoff_eq] at h1
  simp only [Draft0.TwoArmBellman.cumulativeVariance, add_zero] at h0 h1
  have hz (x : ℝ) : hingeLoss x 0 = max (-x) 0 := by
    simp only [hingeLoss, gaussianReal_zero_var, integral_dirac, zero_sub]
  rw [hz] at h0 h1
  have hs1 := hingeLoss_sub_abs (p 2 c.n2 - p 1 c.n1) (updateVariance (c.n1 : ℝ≥0))
  have hs2 := hingeLoss_sub_abs (p 2 c.n2 - p 1 c.n1) (updateVariance (c.n2 : ℝ≥0))
  have hdabs : |p 2 c.n2 - p 1 c.n1| = |terminalGap p c| := abs_sub_comm _ _
  rw [hdabs] at hs1 hs2
  constructor <;> linarith only [hg1, hg2, he, h0, h1, hs1, hs2]

end
end Junpei.PolicyChoice
