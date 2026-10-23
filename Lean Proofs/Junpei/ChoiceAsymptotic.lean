import Junpei.PolicyChoice

open Filter Asymptotics

namespace Junpei.PolicyChoice

open Draft0.Counts Draft0.DynamicPolicy Draft0.Rates
open Junpei.ExclusionPerm Junpei.ExclusionBounds Junpei.ExclusionTail JunpeiComparison

noncomputable section

theorem optimal_EBI_max (k : ℕ) (c : Draft0.BayesianModel.Counts)
    (m : Draft0.BayesianModel.Means) (a : Draft0.BayesianModel.Arm)
    (hc : ∀ i, 0 < c i)
    (ha : Draft0.BayesianModel.action k c m a = Draft0.BayesianModel.value (k+1) c m)
    (i : Draft0.BayesianModel.Arm) : EBI k c m i ≤ EBI k c m a := by
  rw [EBI_eq k c m i (hc i), EBI_eq k c m a (hc a), ha]
  exact sub_le_sub_right (Draft0.BayesianModel.action_le_value k c m i) _

/-- One policy-independent horizon threshold excludes arm 3 at every round
and excludes already saturated arms before the final draw. -/
theorem eventually_optimal_choices (CU : ℝ) (hCU : 33 ≤ CU) :
    ∃ T0 : ℕ, ∀ T : ℕ, T0 ≤ T →
      (∀ (q t : ℕ) (p : MeanPath) (c : Counts) (a : Draft0.BayesianModel.Arm),
        T = 2*q+1 → 4 ≤ t → t < T → horizonConsistent t c → newW T q CU p c →
        Draft0.BayesianModel.action (T-t) (modelCounts c) (modelMeans p c) a =
          Draft0.BayesianModel.value (T-t+1) (modelCounts c) (modelMeans p c) →
        admissible q c a) ∧
      (∀ (q : ℕ) (p : MeanPath) (c : Counts) (a : Draft0.BayesianModel.Arm),
        T = 2*q+1 → horizonConsistent T c → newW T q CU p c →
        Draft0.BayesianModel.action 0 (modelCounts c) (modelMeans p c) a =
          Draft0.BayesianModel.value 1 (modelCounts c) (modelMeans p c) → a ≠ 2) := by
  have herr := paperG0_bigO181 CU hCU
  have hg := eventual_subtraction_positive gainC_pos (by norm_num : (0 : ℝ) < 2)
    (by norm_num : 176 < 181) herr (Filter.Eventually.of_forall (fun _ => le_rfl))
  have hb := eventual_subtraction_positive comparisonC_pos (by norm_num : (0 : ℝ) < 1)
    (by norm_num : 178 < 181) herr (Filter.Eventually.of_forall (fun _ => le_rfl))
  have ht := eventual_subtraction_positive Junpei.Marginal.terminalConstant_pos
    (by norm_num : (0 : ℝ) < 2) (by norm_num : 22 < 181) herr
    (Filter.Eventually.of_forall (fun _ => le_rfl))
  obtain ⟨T0, hT0⟩ := Filter.eventually_atTop.mp
    (hg.and (hb.and (ht.and (eventually_ge_atTop (9 : ℕ)))))
  refine ⟨T0, ?_⟩
  intro T hlarge
  have hmargin := hT0 T hlarge
  have hT9 := hmargin.2.2.2
  have hCU10 : 10 ≤ CU := by linarith
  have hcut : 180 < (CU-6)^2/4 := by nlinarith
  constructor
  · intro q t p c a hTodd ht4 hpre hc hW ha
    have he3 := lemma6_upper T q t CU p c (by omega) (by linarith) hcut hc hW
    have he0 := (lemma7_individual_lower T q t CU p c hTodd ht4 hpre hCU10 hc hW).1
    have hthird : a ≠ 2 := by
      intro h
      have hm := optimal_EBI_max (T-t) (modelCounts c) (modelMeans p c) a
        (modelCounts_positive t c hc) ha 0
      rw [h] at hm
      linarith only [he3, he0, hm, hmargin.1]
    refine ⟨hthird, ?_, ?_⟩
    · intro ha0
      by_contra h
      have hsat : c.n1 = q := by have := hW.2.2.2.2.2.1; omega
      have hgap := lemma7_saturated_left T q t CU p c hTodd ht4 hpre hCU10 hc hW hsat
      have hmax := Draft0.BayesianModel.action_le_value (T-t) (modelCounts c) (modelMeans p c) 1
      rw [ha0] at ha
      linarith only [hgap, hmax, ha, hmargin.2.1]
    · intro ha1
      by_contra h
      have hsat : c.n2 = q := by have := hW.2.2.2.2.2.2.1; omega
      have hgap := lemma7_saturated_right T q t CU p c hTodd ht4 hpre hCU10 hc hW hsat
      have hmax := Draft0.BayesianModel.action_le_value (T-t) (modelCounts c) (modelMeans p c) 0
      rw [ha1] at ha
      linarith only [hgap, hmax, ha, hmargin.2.1]
  · intro q p c a hTodd hc hW ha h
    have he3 := lemma6_upper T q T CU p c (by omega) (by linarith) hcut hc hW
    simp only [Nat.sub_self] at he3
    have he0 := (lemma8_individual_lower T q CU p c hTodd hT9 hCU10 hc hW).1
    have hm := optimal_EBI_max 0 (modelCounts c) (modelMeans p c) a
      (modelCounts_positive T c hc) ha 0
    rw [h] at hm
    linarith only [he3, he0, hm, hmargin.2.2.1]

end
end Junpei.PolicyChoice
