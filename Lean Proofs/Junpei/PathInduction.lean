import Junpei.Policy
import Junpei.ChoiceAsymptotic
import Draft0.PathInduction

namespace Junpei.PathInduction

open Draft0.BayesianModel Draft0.PolicyPaths Draft0.DynamicPolicy
open Draft0.PathInduction
open Draft0.MainBridge (invariant)
open JunpeiComparison

noncomputable section

def GoodPath (T q : ℕ) (CU : ℝ) (p : Path) : Prop :=
  Draft0.Drift.eventX T CU (paperPath p) ∧ Draft0.Drift.eventY T q 1 (paperPath p) ∧
  Draft0.Drift.eventY T q 2 (paperPath p) ∧ newZ T q 1 (paperPath p) ∧ newZ T q 2 (paperPath p)

theorem modelCounts_project (c : Counts) : modelCounts (projectCounts c) = c := by
  ext i
  fin_cases i <;> simp [modelCounts, projectCounts]

theorem modelMeans_project (p : Path) (c : Counts) :
    modelMeans (paperPath p) (projectCounts c) = meansAt p c := by
  ext i
  fin_cases i <;> simp [modelMeans, projectCounts, paperPath, meansAt]

theorem GoodPath_W (T q s : ℕ) (CU : ℝ) (p : Path) (c : Counts)
    (hp : GoodPath T q CU p) (hc : invariant q s c) :
    newW T q CU (paperPath p) (projectCounts c) :=
  ⟨hp.1, hp.2.1, hp.2.2.1, hp.2.2.2.1, hp.2.2.2.2, hc.1, hc.2.1, hc.2.2.1.le⟩

theorem projected_consistent (t : ℕ) (c : Counts) (hcpos : ∀ i, 0 < c i)
    (hc3 : c 2 = 1) (hs : c 0 + c 1 + c 2 = t-1) :
    Draft0.Counts.horizonConsistent t (projectCounts c) := by
  exact ⟨hcpos 0, hcpos 1, hc3, hs⟩

theorem counts_invariant_before_terminal (π : Junpei.Policy.StatePolicy) (T q : ℕ)
    (hq : 2 ≤ q) (p : Path) (u : Junpei.Policy.RandomPath)
    (hpolicy : ∀ s, 3 ≤ s → s < T-1 →
      invariant q s (Junpei.Policy.counts π p u s) →
      Draft0.MainBridge.admissible q (Junpei.Policy.counts π p u s)
        (Junpei.Policy.selected π p u (s+1))) :
    ∀ s, 3 ≤ s → s ≤ T-1 → invariant q s (Junpei.Policy.counts π p u s) := by
  intro s
  induction s using Nat.strong_induction_on with
  | h s ih =>
    intro hlo hhi
    by_cases hbase : s = 3
    · subst s
      change (1 : ℕ) ≤ q ∧ 1 ≤ q ∧ 1 = 1 ∧ 1+1+1=3
      omega
    · have hprev : 3 ≤ s-1 := by omega
      have he : s-1+1=s := by omega
      have hc := ih (s-1) (by omega) hprev (by omega)
      have ha := hpolicy (s-1) hprev (by omega) hc
      rw [← he, Junpei.Policy.counts_update π p u (s-1) hprev]
      simpa only [he] using Draft0.MainBridge.advance_invariant q (s-1)
        (Junpei.Policy.counts π p u (s-1)) (Junpei.Policy.selected π p u s) hc
        (by simpa only [he] using ha)

theorem optimal_bad_path_wrong (π : Junpei.Policy.StatePolicy) (T q : ℕ) (CU : ℝ)
    (hT : T=2*q+1) (hlarge : 9 ≤ T) (hCU : 33 ≤ CU)
    (hπ : Junpei.Policy.PointwiseOptimal π T) (p : Path) (u : Junpei.Policy.RandomPath)
    (hp : GoodPath T q CU p)
    (hbefore : ∀ (t : ℕ) (c : Draft0.Counts.Counts) (a : Arm),
      4 ≤ t → t < T → Draft0.Counts.horizonConsistent t c → newW T q CU (paperPath p) c →
      action (T-t) (modelCounts c) (modelMeans (paperPath p) c) a =
        value (T-t+1) (modelCounts c) (modelMeans (paperPath p) c) →
      Draft0.DynamicPolicy.admissible q c a)
    (hlast : ∀ (c : Draft0.Counts.Counts) (a : Arm),
      Draft0.Counts.horizonConsistent T c → newW T q CU (paperPath p) c →
      action 0 (modelCounts c) (modelMeans (paperPath p) c) a =
        value 1 (modelCounts c) (modelMeans (paperPath p) c) → a ≠ 2) :
    Junpei.Policy.recommendation π T p u ≠ 2 := by
  have hq : 2 ≤ q := by omega
  have hinv := counts_invariant_before_terminal π T q hq p u (fun s hs hsT hc => by
    have hc' := projected_consistent (s+1) (Junpei.Policy.counts π p u s)
      (Junpei.Policy.counts_positive π p u s) hc.2.2.1
      (by simpa only [Nat.add_sub_cancel] using hc.2.2.2)
    have hW := GoodPath_W T q s CU p (Junpei.Policy.counts π p u s) hp hc
    have ho := Junpei.Policy.selected_optimal π T hπ p u (s+1) (by omega) (by omega)
    simp only [Nat.add_sub_cancel] at ho
    have ha := hbefore (s+1) (projectCounts (Junpei.Policy.counts π p u s))
      (Junpei.Policy.selected π p u (s+1)) (by omega) (by omega) hc' hW
      (by simpa only [modelCounts_project, modelMeans_project] using ho)
    exact ha)
  have hc := hinv (T-1) (by omega) le_rfl
  have hc' := projected_consistent T (Junpei.Policy.counts π p u (T-1))
    (Junpei.Policy.counts_positive π p u (T-1)) hc.2.2.1 hc.2.2.2
  have hW := GoodPath_W T q (T-1) CU p (Junpei.Policy.counts π p u (T-1)) hp hc
  have ho := Junpei.Policy.selected_optimal π T hπ p u T (by omega) le_rfl
  simp only [Nat.sub_self, zero_add] at ho
  have ha := hlast (projectCounts (Junpei.Policy.counts π p u (T-1)))
    (Junpei.Policy.selected π p u T) hc' hW
    (by simpa only [modelCounts_project, modelMeans_project] using ho)
  have hu := Junpei.Policy.counts_update π p u (T-1) (by omega)
  have ht : T-1+1=T := by omega
  rw [ht] at hu
  have hthird : Junpei.Policy.counts π p u T 2 = 1 := by
    rw [hu]
    simp only [Draft0.BayesianModel.advance, if_neg ha.symm, add_zero]
    exact hc.2.2.1
  have hanchor : ∃ i : Arm, (i=0 ∨ i=1) ∧ Junpei.Policy.counts π p u T i ≤ q := by
    have hcases : Junpei.Policy.selected π p u T = 0 ∨
        Junpei.Policy.selected π p u T = 1 := by
      generalize _he : Junpei.Policy.selected π p u T = a at ha
      fin_cases a
      · exact Or.inl rfl
      · exact Or.inr rfl
      · exact False.elim (ha rfl)
    rcases hcases with h | h
    · refine ⟨1, Or.inr rfl, ?_⟩
      rw [hu, h]
      simpa [Draft0.BayesianModel.advance] using hc.2.1
    · refine ⟨0, Or.inl rfl, ?_⟩
      rw [hu, h]
      simpa [Draft0.BayesianModel.advance] using hc.1
  obtain ⟨i, hi, hcap⟩ := hanchor
  have hlow : |p i (Junpei.Policy.counts π p u T i)| ≤
      6 * Real.sqrt (Real.log (T : ℝ)) := by
    rcases hi with h | h
    · subst i
      simpa [paperPath] using Junpei.ExclusionTail.mean_bound6_newZ T q 1
        (Junpei.Policy.counts π p u T 0) (paperPath p) (by omega)
        (Junpei.Policy.counts_positive π p u T 0) hcap hp.2.1 hp.2.2.2.1
    · subst i
      simpa [paperPath] using Junpei.ExclusionTail.mean_bound6_newZ T q 2
        (Junpei.Policy.counts π p u T 1) (paperPath p) (by omega)
        (Junpei.Policy.counts_positive π p u T 1) hcap hp.2.2.1 hp.2.2.2.2
  have hx : p 2 1 ≤ -CU * Real.sqrt (Real.log (T : ℝ)) := by
    simpa [Draft0.Drift.eventX, paperPath] using hp.1
  have hsqrt : 1 ≤ Real.sqrt (Real.log (T : ℝ)) :=
    Real.one_le_sqrt.mpr (Draft0.Drift.log_horizon_ge_one T (by omega))
  have hsep : p 2 1 < p i (Junpei.Policy.counts π p u T i) := by
    have h := (abs_le.mp hlow).1
    nlinarith only [h, hx, hsqrt, hCU]
  intro hr
  have hg := Junpei.Policy.recommendation_greedy π T hπ p u i
  change p i (Junpei.Policy.counts π p u T i) ≤
    p (Junpei.Policy.recommendation π T p u)
      (Junpei.Policy.counts π p u T (Junpei.Policy.recommendation π T p u)) at hg
  rw [hr, hthird] at hg
  exact not_le_of_gt hsep hg

end
end Junpei.PathInduction
