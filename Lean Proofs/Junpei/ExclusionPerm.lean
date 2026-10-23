import Junpei.Exclusion

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.ExclusionPerm

open Draft0.BayesianModel Draft0.PosteriorRegret Draft0.ThreeArmComparison
open Junpei.Exclusion

noncomputable section

def renameCounts (e : Equiv.Perm Arm) (c : Counts) : Counts := c ∘ e
def renameMeans (e : Equiv.Perm Arm) (m : Means) : Means := m ∘ e

theorem advance_rename (e : Equiv.Perm Arm) (c : Counts) (i : Arm) :
    advance (renameCounts e c) i = renameCounts e (advance c (e i)) := by
  ext j
  simp [advance, renameCounts, e.injective.eq_iff]

theorem shift_rename (e : Equiv.Perm Arm) (m : Means) (i : Arm) (z : ℝ) :
    shift (renameMeans e m) i z = renameMeans e (shift m (e i) z) := by
  ext j
  simp [shift, renameMeans, e.injective.eq_iff]

theorem terminal_rename (e : Equiv.Perm Arm) (m : Means) :
    terminal (renameMeans e m) = terminal m := by
  have h (n : Means) (i : Arm) : n i ≤ terminal n :=
    (recommend_greedy n i).trans_eq (recommend_max n)
  apply le_antisymm
  · exact max_le (h m (e 0)) (max_le (h m (e 1)) (h m (e 2)))
  · have h0 := h (renameMeans e m) (e.symm 0)
    have h1 := h (renameMeans e m) (e.symm 1)
    have h2 := h (renameMeans e m) (e.symm 2)
    simp only [renameMeans, Function.comp_apply, Equiv.apply_symm_apply] at h0 h1 h2
    exact max_le h0 (max_le h1 h2)

theorem value_rename (k : ℕ) (e : Equiv.Perm Arm) (c : Counts) (m : Means) :
    value k (renameCounts e c) (renameMeans e m) = value k c m := by
  induction k generalizing c m with
  | zero => exact terminal_rename e m
  | succ k ih =>
    have ha (i : Arm) : action k (renameCounts e c) (renameMeans e m) i =
        action k c m (e i) := by
      unfold action
      rw [advance_rename]
      simp_rw [shift_rename, ih]
      rfl
    apply le_antisymm
    · rw [value_succ, ha 0, ha 1, ha 2]
      exact max_le (action_le_value k c m (e 0))
        (max_le (action_le_value k c m (e 1)) (action_le_value k c m (e 2)))
    · have h0 := action_le_value k (renameCounts e c) (renameMeans e m) (e.symm 0)
      have h1 := action_le_value k (renameCounts e c) (renameMeans e m) (e.symm 1)
      have h2 := action_le_value k (renameCounts e c) (renameMeans e m) (e.symm 2)
      rw [ha, Equiv.apply_symm_apply] at h0 h1 h2
      rw [value_succ]
      exact max_le h0 (max_le h1 h2)

theorem action_rename (k : ℕ) (e : Equiv.Perm Arm) (c : Counts) (m : Means) (i : Arm) :
    action k (renameCounts e c) (renameMeans e m) i = action k c m (e i) := by
  unfold action
  rw [advance_rename]
  simp_rw [shift_rename, value_rename]
  rfl

def EBI (k : ℕ) (c : Counts) (m : Means) (i : Arm) : ℝ :=
  loss k c m - actionLoss k c m i

theorem EBI_eq (k : ℕ) (c : Counts) (m : Means) (i : Arm) (hc : 0 < c i) :
    EBI k c m i = action k c m i - value k c m := by
  rw [EBI, actionLoss_eq k c m i hc]
  unfold loss
  ring

/-- Revised Lemma12 for every arm of the three-arm Gaussian model and every horizon. -/
theorem lemma12_k3 (k : ℕ) (c : Counts) (m : Means) (i : Arm)
    (hc : ∀ j, 0 < c j) : EBI k c m i ≤ exclusion c m i := by
  let e : Equiv.Perm Arm := Equiv.swap 2 i
  have he : e 2 = i := Equiv.swap_apply_left 2 i
  have hpositive : ∀ j, 0 < renameCounts e c j := fun j => hc (e j)
  have ha := (action_comparison k (renameCounts e c) (renameMeans e m) 2 hpositive).2
  rw [twoAction_two, action_rename, he] at ha
  have hv := (value_comparison k (renameCounts e c) (renameMeans e m) hpositive).1
  rw [value_rename] at hv
  have hpair : stateSlack (renameCounts e c) (renameMeans e m) =
      pairSlack c m i (e 0) := by
    simp only [stateSlack, slack, renameCounts, renameMeans, Function.comp_apply,
      he, pairSlack, Draft0.Gaussian.posteriorVariance]
    rw [add_comm]
  have hne : i ≠ e 0 := by
    rw [← he]
    exact fun h => (show (2 : Arm) ≠ 0 by decide) (e.injective h)
  have hs := pairSlack_le_exclusion c m i (e 0) hne
  rw [EBI_eq k c m i (hc i)]
  rw [hpair] at ha
  linarith

end

end Junpei.ExclusionPerm
