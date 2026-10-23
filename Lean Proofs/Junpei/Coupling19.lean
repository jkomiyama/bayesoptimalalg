import Junpei.CommonSample
import Junpei.Coupling19Counts

/-! Explicit arm-dependent optimal allocations, permutation of their common
samples, and the conditional last-observation identities used in Lemma 19. -/

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.Coupling19

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain
open Junpei.CommonSample Junpei.Forecast

noncomputable section

def leftTie (a b : ℕ) : Bool := decide (a ≤ b)

theorem leftTie_smaller : SmallerCountRule leftTie := by
  intro a b
  unfold leftTie
  by_cases h : a ≤ b
  · simp [h]
  · simp [h]
    omega

theorem leftTie_step (c : Counts) : ruleStep leftTie c = balancedStep c := by
  simp [ruleStep, leftTie, balancedStep]

theorem leftTie_run (k : ℕ) (c : Counts) : ruleRun leftTie k c = balancedRun k c := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih => rw [ruleRun, leftTie_step, ih, balancedRun]

def schedule (r : ℕ → ℕ → Bool) : ℕ → Counts → List Bool
  | 0, _ => []
  | k + 1, c => r c.n1 c.n2 :: schedule r k (ruleStep r c)

theorem schedule_final (r : ℕ → ℕ → Bool) (k : ℕ) (c : Counts) :
    finalCounts (schedule r k c) c = ruleRun r k c := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
    simp only [schedule, finalCounts, ruleRun]
    have hs : step (r c.n1 c.n2) c = ruleStep r c := by
      unfold step ruleStep
      split <;> rfl
    rw [hs, ih]

theorem schedule_optimal (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (k : ℕ) (c : Counts) (m1 m2 : ℝ) (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    utility (schedule r k c) c m1 m2 = payoff k m1 m2 c := by
  induction k generalizing c m1 m2 with
  | zero => rfl
  | succ k ih =>
    have hm := hr c.n1 c.n2
    cases hd : r c.n1 c.n2
    · simp only [hd, Bool.false_eq_true, ↓reduceIte] at hm
      have hs : ruleStep r c = rightStep c := by simp [ruleStep, hd, rightStep]
      simp only [schedule, hd, utility, Bool.false_eq_true, ↓reduceIte, hs]
      simp_rw [ih (rightStep c) _ _ (by simpa [rightStep] using h1) (by simp [rightStep])]
      change rightAction k m1 m2 c = payoff (k + 1) m1 m2 c
      by_cases he : c.n1 = c.n2
      · have hc : c = ⟨c.n2, c.n2, c.n3⟩ := by
          cases c
          simp_all
        rw [hc]
        exact (JunpeiComparison.Gaussian.equal_count_right_action_optimal
          k c.n2 c.n3 m1 m2 h2).symm
      · exact (smaller_count_right_optimal k m1 m2 c h1 h2 (by omega)).symm
    · simp only [hd, ↓reduceIte] at hm
      have hs : ruleStep r c = leftStep c := by simp [ruleStep, hd, leftStep]
      simp only [schedule, hd, utility, ↓reduceIte, hs]
      simp_rw [ih (leftStep c) _ _ (by simp [leftStep]) (by simpa [leftStep] using h2)]
      exact (smaller_count_left_optimal k m1 m2 c h1 h2 hm).symm

theorem finalCounts_append (s u : List Bool) (c : Counts) :
    finalCounts (s ++ u) c = finalCounts u (finalCounts s c) := by
  induction s generalizing c with
  | nil => rfl
  | cons i s ih => exact ih (step i c)

theorem allocationVariance_append (s u : List Bool) (c : Counts) :
    allocationVariance (s ++ u) c = allocationVariance s c + allocationVariance u (finalCounts s c) := by
  induction s generalizing c with
  | nil => simp [allocationVariance, finalCounts]
  | cons i s ih =>
    simp only [List.cons_append, allocationVariance, finalCounts, ih, add_assoc]

theorem finalCounts_left (s : List Bool) (c : Counts) :
    (finalCounts s c).n1 = c.n1 + s.count true := by
  induction s generalizing c with
  | nil => simp [finalCounts]
  | cons i s ih => cases i <;> simp [finalCounts, ih, step, leftStep, rightStep]; omega

theorem finalCounts_right (s : List Bool) (c : Counts) :
    (finalCounts s c).n2 = c.n2 + s.count false := by
  induction s generalizing c with
  | nil => simp [finalCounts]
  | cons i s ih => cases i <;> simp [finalCounts, ih, step, leftStep, rightStep]; omega

theorem permutation_of_finalCounts (s u : List Bool) (c : Counts)
    (h : finalCounts s c = finalCounts u c) : s.Perm u := by
  apply List.perm_iff_count.mpr
  intro i
  cases i
  · have he := congrArg Counts.n2 h
    rw [finalCounts_right, finalCounts_right] at he
    omega
  · have he := congrArg Counts.n1 h
    rw [finalCounts_left, finalCounts_left] at he
    omega

/-- The forced-first allocation is a permutation of the chosen skip
allocation followed by one final sample from the SAME forced arm. -/
theorem forced_left_is_common_then_last (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (true :: schedule leftTie k (leftStep c)).Perm (schedule rightTie k c ++ [true]) := by
  apply permutation_of_finalCounts _ _ c
  simp only [finalCounts, step, ↓reduceIte, schedule_final, leftTie_run, finalCounts_append]
  exact repaired_left_allocation_plus_one q k c hq hcap1 hcap2 hs

theorem forced_right_is_common_then_last (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (false :: schedule leftTie k (rightStep c)).Perm (schedule leftTie k c ++ [false]) := by
  apply permutation_of_finalCounts _ _ c
  simp only [finalCounts, step, Bool.false_eq_true, ↓reduceIte, schedule_final,
    leftTie_run, finalCounts_append]
  exact repaired_right_allocation_plus_one q k c hq hcap1 hcap2 hs

/-- Both experiments share the entire observed stream. The extra forced-arm
observation is explicitly placed last, without changing any terminal mean. -/
theorem forced_left_shared_samples (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) (m : ℝ × ℝ) (z : Stream) :
    terminalMeans (true :: schedule leftTie k (leftStep c)) c m z =
      terminalMeans (schedule rightTie k c ++ [true]) c m z :=
  terminalMeans_permutation _ _ (forced_left_is_common_then_last q k c hq hcap1 hcap2 hs) c m z

theorem forced_right_shared_samples (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) (m : ℝ × ℝ) (z : Stream) :
    terminalMeans (false :: schedule leftTie k (rightStep c)) c m z =
      terminalMeans (schedule leftTie k c ++ [false]) c m z :=
  terminalMeans_permutation _ _ (forced_right_is_common_then_last q k c hq hcap1 hcap2 hs) c m z

/-- The exact conditional one-step identity, averaged under the common
Gaussian posterior forecast. It holds for every deterministic prefix. -/
theorem common_prefix_last_gain (s : List Bool) (i : Bool) (c : Counts) (m1 m2 : ℝ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    risk s c m1 m2 - risk (s ++ [i]) c m1 m2 =
      ∫ x : ℝ, hingeLoss |x| (updateVariance (sampled i (finalCounts s c)))
        ∂gaussianReal (m1 - m2) (allocationVariance s c) := by
  rw [risk_eq s c m1 m2 h1 h2, risk_eq (s ++ [i]) c m1 m2 h1 h2,
    utility_eq, utility_eq, allocationVariance_append]
  simp only [allocationVariance, add_zero]
  have h := hinge_variance_increment_integral (m2 - m1) (allocationVariance s c)
    (updateVariance (sampled i (finalCounts s c)))
  have he : -(m2 - m1) = m1 - m2 := by ring
  rw [he] at h
  linarith only [h]

/-- The two coupled forced allocations differ only in their last sample.
This is the exact conditional action-gap identity preceding Eq. (25). -/
theorem common_prefix_last_action_gap (s : List Bool) (c : Counts) (m1 m2 : ℝ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    risk (s ++ [true]) c m1 m2 - risk (s ++ [false]) c m1 m2 =
      ∫ x : ℝ, (hingeLoss |x| (updateVariance (finalCounts s c).n2) -
        hingeLoss |x| (updateVariance (finalCounts s c).n1))
        ∂gaussianReal (m1 - m2) (allocationVariance s c) := by
  rw [integral_sub (absolute_hinge_gaussian_integrable _ _ _)
    (absolute_hinge_gaussian_integrable _ _ _)]
  have hleft := common_prefix_last_gain s true c m1 m2 h1 h2
  have hright := common_prefix_last_gain s false c m1 m2 h1 h2
  simp only [sampled, Bool.false_eq_true, ↓reduceIte] at hleft hright
  linarith only [hleft, hright]

theorem ruleRun_n3 (r : ℕ → ℕ → Bool) (k : ℕ) (c : Counts) :
    (ruleRun r k c).n3 = c.n3 := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
    rw [ruleRun, ih]
    unfold ruleStep
    split <;> rfl

theorem ruleRun_right_only (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (k : ℕ) (c : Counts) (h : c.n2 + k ≤ c.n1) :
    ruleRun r k c = ⟨c.n1, c.n2 + k, c.n3⟩ := by
  induction k generalizing c with
  | zero => simp [ruleRun]
  | succ k ih =>
    have hsmall : c.n2 < c.n1 := by omega
    have hchoice : r c.n1 c.n2 = false := by
      have hm := hr c.n1 c.n2
      cases hd : r c.n1 c.n2
      · rfl
      · simp only [hd, ↓reduceIte] at hm
        omega
    have hs : ruleStep r c = rightStep c := by simp [ruleStep, hchoice, rightStep]
    rw [ruleRun, hs, ih (rightStep c) (by simp only [rightStep]; omega)]
    simp only [rightStep]
    congr 1
    omega

/-- The forced terminal allocation is independent of the count-only tie
rule. Therefore the SAME arm-dependent rule may be used within each pair of
coupled experiments, as in the intended reading of the printed proof. -/
theorem forced_left_terminal_counts_independent_rule (r : ℕ → ℕ → Bool)
    (hr : SmallerCountRule r) (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    ruleRun r k (leftStep c) = balancedRun k (leftStep c) := by
  have hn3 := ruleRun_n3 r k (leftStep c)
  have hf := Junpei.Dynamic.forced_left_final_counts q k c hq hcap1 hcap2 hs
  by_cases h : c.n1 = q
  · have hall : (leftStep c).n2 + k ≤ (leftStep c).n1 := by simp only [leftStep]; omega
    rw [ruleRun_right_only r hr k (leftStep c) hall,
      balancedRun_right_only k (leftStep c) hall]
  · have ha : (leftStep c).n1 ≤ q := by simp only [leftStep]; omega
    have hb : (leftStep c).n2 ≤ q := hcap2
    have hsum : (leftStep c).n1 + (leftStep c).n2 + k = 2 * q := by simp only [leftStep]; omega
    have hcaps := ruleRun_caps r hr q k (leftStep c) ha hb (by omega)
    have htotal := ruleRun_sum r k (leftStep c)
    simp only [if_neg h] at hf
    apply counts_ext
    · rw [hf.1]
      omega
    · rw [hf.2]
      omega
    · rw [hn3, balancedRun_n3]

/-- Both sides now use the same right-tie rule, selected for the LEFT-arm
comparison before either experiment. -/
theorem same_rule_left_common_then_last (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) :
    (true :: schedule rightTie k (leftStep c)).Perm (schedule rightTie k c ++ [true]) := by
  apply permutation_of_finalCounts _ _ c
  simp only [finalCounts, step, ↓reduceIte, schedule_final, finalCounts_append]
  rw [forced_left_terminal_counts_independent_rule rightTie rightTie_smaller q k c hq hcap1 hcap2 hs]
  exact repaired_left_allocation_plus_one q k c hq hcap1 hcap2 hs

theorem same_rule_left_shared_samples (q k : ℕ) (c : Counts) (hq : 0 < q)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1) (m : ℝ × ℝ) (z : Stream) :
    terminalMeans (true :: schedule rightTie k (leftStep c)) c m z =
      terminalMeans (schedule rightTie k c ++ [true]) c m z :=
  terminalMeans_permutation _ _ (same_rule_left_common_then_last q k c hq hcap1 hcap2 hs) c m z

end
end Junpei.Coupling19
