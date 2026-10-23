import Draft0.PolicyPaths
import Draft0.Drift

namespace Draft0.PathInduction

open BayesianModel PolicyPaths

noncomputable section

def paperPath (p : Path) : Counts.MeanPath :=
  fun i n => if i = 1 then p 0 n else if i = 2 then p 1 n else p 2 n

def projectCounts (c : BayesianModel.Counts) : Counts.Counts := ⟨c 0, c 1, c 2⟩

def goodPath (T q : ℕ) (p : Path) : Prop :=
  Drift.eventX T 40 (paperPath p) ∧ Drift.eventY T q 1 (paperPath p) ∧
  Drift.eventY T q 2 (paperPath p) ∧ Drift.eventZ T q 1 (paperPath p) ∧
  Drift.eventZ T q 2 (paperPath p)

theorem goodPath_W (T q s : ℕ) (p : Path) (c : BayesianModel.Counts)
    (hp : goodPath T q p) (hc : MainBridge.invariant q s c) :
    Drift.paperW T q 40 (paperPath p) (projectCounts c) := by
  have hcaps : c 0 ≤ q ∧ c 1 ≤ q ∧ c 2 = 1 := by
    exact ⟨hc.1, hc.2.1, hc.2.2.1⟩
  exact ⟨hp.1, hp.2.1, hp.2.2.1, hp.2.2.2.1, hp.2.2.2.2,
    hcaps.1, hcaps.2.1, hcaps.2.2.le⟩

/-- Round induction valid for both budget parities, including the last draw. -/
theorem counts_invariant (T q : ℕ) (hq : 2 ≤ q) (_hT : 3 ≤ T) (p : Path)
    (hpolicy : ∀ s, 3 ≤ s → s < T → MainBridge.invariant q s (counts T p s) →
      MainBridge.admissible q (counts T p s) (selected T p (s + 1))) :
    ∀ s, 3 ≤ s → s ≤ T → MainBridge.invariant q s (counts T p s) := by
  intro s
  induction s using Nat.strong_induction_on with
  | h s ih =>
    intro hlo hhi
    by_cases hbase : s = 3
    · subst s
      change (1 : ℕ) ≤ q ∧ 1 ≤ q ∧ 1 = 1 ∧ 1 + 1 + 1 = 3
      omega
    · have hp : 3 ≤ s - 1 := by omega
      have he : s - 1 + 1 = s := by omega
      have hc := ih (s - 1) (by omega) hp (by omega)
      have ha := hpolicy (s - 1) hp (by omega) hc
      rw [← he, counts_update T p (s - 1) hp]
      simpa only [he] using MainBridge.advance_invariant q (s - 1) (counts T p (s - 1))
        (selected T p s) hc (by simpa only [he] using ha)

theorem final_anchor_count (T q : ℕ) (c : BayesianModel.Counts)
    (hT : T = 2 * q ∨ T = 2 * q + 1) (hc : MainBridge.invariant q T c) :
    c 0 = q ∨ c 1 = q := by
  unfold MainBridge.invariant at hc
  omega

theorem goodPath_recommendation_wrong (T q : ℕ) (_hq : 2 ≤ q) (hT5 : 5 ≤ T)
    (hT : T = 2 * q ∨ T = 2 * q + 1) (p : Path) (hp : goodPath T q p)
    (hc : MainBridge.invariant q T (counts T p T)) :
    recommendation T p ≠ 2 := by
  have hanchor := final_anchor_count T q (counts T p T) hT hc
  have hx : p 2 1 ≤ -40 * Real.sqrt (Real.log (T : ℝ)) := by
    simpa [Drift.eventX, paperPath] using hp.1
  have hc3 := hc.2.2.1
  intro hr
  rcases hanchor with h₀ | h₁
  · have hy : |p 0 q| ≤ 1 / (T : ℝ) ^ 2 := by
      simpa [Drift.eventY, paperPath] using hp.2.1
    have hsep := MainBridge.terminal_underestimation_separation T hT5 40 (p 0 q)
      (p 2 1) (by norm_num) hy hx
    have hg := recommendation_greedy T p 0
    rw [hr, h₀, hc3] at hg
    exact not_le_of_gt hsep hg
  · have hy : |p 1 q| ≤ 1 / (T : ℝ) ^ 2 := by
      simpa [Drift.eventY, paperPath] using hp.2.2.1
    have hsep := MainBridge.terminal_underestimation_separation T hT5 40 (p 1 q)
      (p 2 1) (by norm_num) hy hx
    have hg := recommendation_greedy T p 1
    rw [hr, h₁, hc3] at hg
    exact not_le_of_gt hsep hg

end

end Draft0.PathInduction
