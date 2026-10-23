import Junpei.Dynamic
import Junpei.ForecastRules

/-! A single globally fixed left-tie rule cannot supply the skip orientation
for both forced-arm comparisons. The counterexample persists at arbitrarily
large odd horizons. The paper's explicit orientation choice is formalized
using one fixed count-only rule within each comparison, chosen for that arm. -/

namespace Junpei.Coupling19

open Draft0.Counts Draft0.TwoArmBellman
open Draft0.TwoArmGain Junpei.Forecast

noncomputable section

theorem fixed_tie_skip_left_counts (q : ℕ) (hq : 2 ≤ q) :
    (balancedRun (2 * q - 3) ⟨1, 1, 1⟩).n1 = q ∧
      (balancedRun (2 * q - 3) ⟨1, 1, 1⟩).n2 = q - 1 := by
  have h := equation22_terminal_counts q (2 * q - 3) ⟨1, 1, 1⟩
    (by omega) (by change 1 ≤ q; omega) (by change 1 ≤ q; omega) (by simp; omega)
  simpa only [show ¬ (1 : ℕ) = q by omega, if_false] using h

theorem fixed_tie_forced_left_counts (q : ℕ) (hq : 2 ≤ q) :
    (balancedRun (2 * q - 3) (leftStep ⟨1, 1, 1⟩)).n1 = q ∧
      (balancedRun (2 * q - 3) (leftStep ⟨1, 1, 1⟩)).n2 = q := by
  have h := Junpei.Dynamic.forced_left_final_counts q (2 * q - 3) ⟨1, 1, 1⟩
    (by omega) (by change 1 ≤ q; omega) (by change 1 ≤ q; omega) (by simp; omega)
  simpa only [show ¬ (1 : ℕ) = q by omega, if_false] using h

/-- A globally fixed LEFT-tie skip allocation cannot always be followed by
one additional LEFT draw to give the optimal forced-left allocation. This
refutes that stronger interpretation, rather than Lemma 19 itself. -/
theorem fixed_tie_same_forced_arm_assertion_false (q : ℕ) (hq : 2 ≤ q) :
    balancedRun (2 * q - 3) (leftStep ⟨1, 1, 1⟩) ≠
      leftStep (balancedRun (2 * q - 3) ⟨1, 1, 1⟩) := by
  intro h
  have hskip := fixed_tie_skip_left_counts q hq
  have hforce := fixed_tie_forced_left_counts q hq
  have he := congrArg Counts.n1 h
  change (balancedRun (2 * q - 3) (leftStep ⟨1, 1, 1⟩)).n1 =
    (balancedRun (2 * q - 3) ⟨1, 1, 1⟩).n1 + 1 at he
  rw [hforce.1, hskip.1] at he
  omega

theorem counts_ext {c d : Counts} (h1 : c.n1 = d.n1) (h2 : c.n2 = d.n2)
    (h3 : c.n3 = d.n3) : c = d := by
  cases c
  cases d
  simp_all

def counterexamplePath (T : ℕ) (CU : ℝ) : Draft0.Counts.MeanPath :=
  fun i _ => if i = 3 then -CU * Real.sqrt (Real.log (T : ℝ)) - 1 else 0

/-- The failing coupling occurs inside the actual W(t) domain of Lemma 19. -/
theorem counterexample_within_lemma19_domain (q : ℕ) (hq : 2 ≤ q) (CU : ℝ) :
    horizonConsistent 4 ⟨1, 1, 1⟩ ∧ 4 < 2 * q + 1 ∧
      (2 * q + 1 - 4 = 2 * q - 3) ∧
      JunpeiComparison.newW (2 * q + 1) q CU (counterexamplePath (2 * q + 1) CU) ⟨1, 1, 1⟩ := by
  refine ⟨by norm_num [horizonConsistent], by omega, by omega, ?_⟩
  have hz (i : ℕ) (hi : i ≠ 3) :
      JunpeiComparison.newZ (2 * q + 1) q i (counterexamplePath (2 * q + 1) CU) := by
    intro n hn hnq
    simp only [counterexamplePath, if_neg hi, sub_self, abs_zero]
    positivity
  refine ⟨?_, ?_, ?_, hz 1 (by norm_num), hz 2 (by norm_num), ?_, ?_, by norm_num⟩
  · dsimp [Draft0.Drift.eventX, counterexamplePath]
    linarith
  · unfold Draft0.Drift.eventY counterexamplePath
    norm_num
    positivity
  · unfold Draft0.Drift.eventY counterexamplePath
    norm_num
    positivity
  · change 1 ≤ q
    omega
  · change 1 ≤ q
    omega

def swapCounts (c : Counts) : Counts := ⟨c.n2, c.n1, c.n3⟩

def rightTie (a b : ℕ) : Bool := decide (a < b)

theorem rightTie_smaller : SmallerCountRule rightTie := by
  intro a b
  unfold rightTie
  by_cases h : a < b <;> simp [h] <;> omega

theorem rightTie_step_swap (c : Counts) :
    ruleStep rightTie c = swapCounts (balancedStep (swapCounts c)) := by
  unfold ruleStep rightTie swapCounts balancedStep
  by_cases h : c.n1 < c.n2
  · have h' : ¬ c.n2 ≤ c.n1 := by omega
    simp [h, h']
  · have h' : c.n2 ≤ c.n1 := by omega
    simp [h, h']

theorem swapCounts_involutive (c : Counts) : swapCounts (swapCounts c) = c := rfl

theorem rightTie_run_swap (k : ℕ) (c : Counts) :
    ruleRun rightTie k c = swapCounts (balancedRun k (swapCounts c)) := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
    rw [ruleRun, ih, rightTie_step_swap, swapCounts_involutive, balancedRun]

theorem oriented_skip_left_counts (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (ruleRun rightTie k c).n1 = (if c.n1 = q then q else q - 1) ∧
      (ruleRun rightTie k c).n2 = (if c.n1 = q then q - 1 else q) := by
  rw [rightTie_run_swap]
  have h := equation22_terminal_counts q k (swapCounts c) hq hcap2 hcap1
    (by simpa [swapCounts, add_comm] using hs)
  change (balancedRun k (swapCounts c)).n2 = _ ∧
    (balancedRun k (swapCounts c)).n1 = _
  exact ⟨h.2, h.1⟩

/-- Repaired orientation: sample the forced LEFT arm once after all common samples. -/
theorem repaired_left_allocation_plus_one (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    balancedRun k (leftStep c) = leftStep (ruleRun rightTie k c) := by
  have hskip := oriented_skip_left_counts q k c hq hcap1 hcap2 hs
  have hforce := Junpei.Dynamic.forced_left_final_counts q k c hq hcap1 hcap2 hs
  have hn3 : (ruleRun rightTie k c).n3 = c.n3 := by
    rw [rightTie_run_swap]
    exact balancedRun_n3 k (swapCounts c)
  apply counts_ext
  · change (balancedRun k (leftStep c)).n1 = (ruleRun rightTie k c).n1 + 1
    rw [hforce.1, hskip.1]
    split_ifs <;> omega
  · change (balancedRun k (leftStep c)).n2 = (ruleRun rightTie k c).n2
    exact hforce.2.trans hskip.2.symm
  · change (balancedRun k (leftStep c)).n3 = (ruleRun rightTie k c).n3
    rw [balancedRun_n3, hn3]
    rfl

/-- Repaired orientation: sample the forced RIGHT arm once after all common samples. -/
theorem repaired_right_allocation_plus_one (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    balancedRun k (rightStep c) = rightStep (balancedRun k c) := by
  have hskip := equation22_terminal_counts q k c hq hcap1 hcap2 hs
  have hforce := Junpei.Dynamic.forced_right_final_counts q k c hq hcap1 hcap2 hs
  apply counts_ext
  · change (balancedRun k (rightStep c)).n1 = (balancedRun k c).n1
    exact hforce.1.trans hskip.1.symm
  · change (balancedRun k (rightStep c)).n2 = (balancedRun k c).n2 + 1
    rw [hforce.2, hskip.2]
    split_ifs <;> omega
  · change (balancedRun k (rightStep c)).n3 = (balancedRun k c).n3
    rw [balancedRun_n3, balancedRun_n3]
    rfl

end
end Junpei.Coupling19
