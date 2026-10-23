import Junpei.PolicyChoice
import Junpei.ExclusionAsymptotics

open Filter Asymptotics

namespace Junpei.PolicyChoice

open Draft0.Counts Draft0.DynamicPolicy Draft0.Rates Draft0.ThreeArmComparison
open Junpei.ExclusionPerm Junpei.ExclusionBounds Junpei.ExclusionTail JunpeiComparison

noncomputable section

theorem lemma6_absolute_upper (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : 5 ≤ T) (hCU : 0 < CU) (hcut : 180 < (CU-6)^2/4)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) :
    |EBI (T-t) (modelCounts c) (modelMeans p c) 2| ≤ paperG0 T CU := by
  have hu := lemma6_upper T q t CU p c hT hCU hcut hc hW
  have hl := binary_EBI_lower (T-t) (modelCounts c) (modelMeans p c) 2
    (modelCounts_positive t c hc)
  rw [twoAction_two] at hl
  have he := lemma11_from_equation11 T q t CU p c hT hCU hcut hc hW
  apply abs_le.mpr
  constructor <;> linarith only [hu, hl, he]

/-- Full uniform Omega bounds of Lemmas 7 and 8 under the paper's exact
cutoff condition. The constants and horizon threshold do not depend on the
round, posterior means, counts, or optimal policy. -/
theorem lemmas7_8_uniform (CU : ℝ) (hCU : 0 < CU) (hcut : 180 < (CU-6)^2/4) :
    ∃ T0 : ℕ, ∀ T : ℕ, T0 ≤ T →
      (∀ (q t : ℕ) (p : MeanPath) (c : Counts), T=2*q+1 → 4 ≤ t → t<T →
        horizonConsistent t c → newW T q CU p c →
        ((gainC/2)*invPower 178 T ≤ EBI (T-t) (modelCounts c) (modelMeans p c) 0 ∧
         (gainC/2)*invPower 178 T ≤ EBI (T-t) (modelCounts c) (modelMeans p c) 1) ∧
        (c.n1=q → (comparisonC/2)*invPower 178 T ≤
          Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 1 -
          Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 0) ∧
        (c.n2=q → (comparisonC/2)*invPower 178 T ≤
          Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 0 -
          Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) 1)) ∧
      (∀ (q : ℕ) (p : MeanPath) (c : Counts), T=2*q+1 →
        horizonConsistent T c → newW T q CU p c →
        (Junpei.Marginal.terminalConstant/2)*invPower 22 T ≤ EBI 0 (modelCounts c) (modelMeans p c) 0 ∧
        (Junpei.Marginal.terminalConstant/2)*invPower 22 T ≤ EBI 0 (modelCounts c) (modelMeans p c) 1) := by
  have herr := Junpei.ExclusionAsymptotics.paperG0_bigO180_equation11 CU hcut
  have hg := eventual_subtraction_lower_bound gainC_pos (by norm_num : (0 : ℝ)<1)
    (by norm_num : 176<180) herr (Filter.Eventually.of_forall (fun _ => le_rfl))
  have hb := eventual_subtraction_lower_bound comparisonC_pos (by norm_num : (0 : ℝ)<1)
    (by norm_num : 178<180) herr (Filter.Eventually.of_forall (fun _ => le_rfl))
  have ht := eventual_subtraction_lower_bound Junpei.Marginal.terminalConstant_pos
    (by norm_num : (0 : ℝ)<1) (by norm_num : 22<180) herr
    (Filter.Eventually.of_forall (fun _ => le_rfl))
  obtain ⟨T0, hT0⟩ := Filter.eventually_atTop.mp
    (hg.and (hb.and (ht.and (eventually_ge_atTop (9 : ℕ)))))
  refine ⟨T0, ?_⟩
  intro T hlarge
  have hmargin := hT0 T hlarge
  have hT9 := hmargin.2.2.2
  have h10 := cutoff_condition_implies_ten CU hCU hcut
  have hweak : (gainC/2)*invPower 178 T ≤ (gainC/2)*invPower 176 T :=
    mul_le_mul_of_nonneg_left (invPower_antitone (by norm_num) (by omega)) (half_pos gainC_pos).le
  simp only [one_mul] at hmargin
  constructor
  · intro q t p c hTodd ht4 hpre hc hW
    have hi := lemma7_individual_lower T q t CU p c hTodd ht4 hpre h10 hc hW
    refine ⟨⟨(hweak.trans hmargin.1).trans hi.1, (hweak.trans hmargin.1).trans hi.2⟩, ?_, ?_⟩
    · intro hsat
      exact hmargin.2.1.trans (lemma7_saturated_left T q t CU p c hTodd ht4 hpre h10 hc hW hsat)
    · intro hsat
      exact hmargin.2.1.trans (lemma7_saturated_right T q t CU p c hTodd ht4 hpre h10 hc hW hsat)
  · intro q p c hTodd hc hW
    have hi := lemma8_individual_lower T q CU p c hTodd hT9 h10 hc hW
    exact ⟨hmargin.2.2.1.trans hi.1, hmargin.2.2.1.trans hi.2⟩

end
end Junpei.PolicyChoice
