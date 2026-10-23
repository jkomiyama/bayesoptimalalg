import Junpei.CommonSampleLaw

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.CommonSample

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain

noncomputable section

def squareWeight (a : ℝ) : ℝ≥0 := ⟨a ^ 2, sq_nonneg a⟩

def weightedSum {N : ℕ} (a : Bool × Fin N → ℝ) (x : Bool × Fin N → ℝ) : ℝ :=
  ∑ j, a j * x j

def weightedVariance {N : ℕ} (a : Bool × Fin N → ℝ) (j : Bool × Fin N) : ℝ≥0 :=
  squareWeight (a j) * updateVariance (j.2.val : ℝ≥0)

theorem weighted_coordinate_law {N : ℕ} (a : Bool × Fin N → ℝ) (j : Bool × Fin N) :
    HasLaw (fun x : Bool × Fin N → ℝ => a j * x j) (gaussianReal 0 (weightedVariance a j)) (streamLaw N) := by
  have he : HasLaw (fun x : Bool × Fin N → ℝ => x j)
      (gaussianReal 0 (updateVariance (j.2.val : ℝ≥0))) (streamLaw N) :=
    (measurePreserving_eval (fun i : Bool × Fin N => gaussianReal 0
      (updateVariance (i.2.val : ℝ≥0))) j).hasLaw
  have hs : HasLaw (fun x : ℝ => a j * x) (gaussianReal 0 (weightedVariance a j))
      (gaussianReal 0 (updateVariance (j.2.val : ℝ≥0))) := by
    constructor
    · exact (by fun_prop : AEMeasurable (fun x : ℝ => a j * x) (gaussianReal 0 (updateVariance (j.2.val : ℝ≥0))))
    · rw [gaussianReal_map_const_mul, mul_zero]
      rfl
  exact hs.fun_comp he

theorem weighted_finset_law {N : ℕ} (a : Bool × Fin N → ℝ) (s : Finset (Bool × Fin N)) :
    HasLaw (fun x : Bool × Fin N → ℝ => ∑ j ∈ s, a j * x j)
      (gaussianReal 0 (∑ j ∈ s, weightedVariance a j)) (streamLaw N) := by
  have hind : iIndepFun (fun j : Bool × Fin N => fun x : Bool × Fin N → ℝ => a j * x j) (streamLaw N) :=
    iIndepFun_pi (fun _ => by fun_prop)
  induction s using Finset.induction_on with
  | empty =>
    constructor
    · simp only [Finset.sum_empty]; exact aemeasurable_const
    · simp [gaussianReal_zero_var]
  | insert j s hj ih =>
    have hInd := hind.indepFun_finset_sum_of_notMem (fun _ => by fun_prop) hj
    have hInd' : IndepFun (fun x : Bool × Fin N → ℝ => a j * x j)
        (fun x : Bool × Fin N → ℝ => ∑ k ∈ s, a k * x k) (streamLaw N) := by
      simpa only [Finset.sum_fn] using hInd.symm
    have h := hInd'.hasLaw_fun_add (weighted_coordinate_law a j) ih
    rw [gaussianReal_conv_gaussianReal, zero_add] at h
    simpa only [Finset.sum_insert hj] using h

theorem weightedSum_law {N : ℕ} (a : Bool × Fin N → ℝ) :
    HasLaw (weightedSum a) (gaussianReal 0 (∑ j, weightedVariance a j)) (streamLaw N) :=
  weighted_finset_law a Finset.univ

def gapCoefficient (N : ℕ) (c d : Counts) (j : Bool × Fin N) : ℝ :=
  if j.1 then if c.n1 ≤ j.2.val ∧ j.2.val < d.n1 then 1 else 0
  else if c.n2 ≤ j.2.val ∧ j.2.val < d.n2 then -1 else 0

theorem gap_weightedSum (N : ℕ) (c d : Counts) (x : Bool × Fin N → ℝ) :
    weightedSum (gapCoefficient N c d) x =
      blockSum N c.n1 d.n1 (fun j => x (true, j)) -
        blockSum N c.n2 d.n2 (fun j => x (false, j)) := by
  unfold weightedSum
  rw [Fintype.sum_prod_type, Fintype.sum_bool]
  simp only [gapCoefficient, ↓reduceIte, Bool.false_eq_true, ite_mul, one_mul, zero_mul, neg_one_mul]
  simp only [blockSum, sub_eq_add_neg, ← Finset.sum_neg_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  split_ifs <;> simp

theorem gap_weightedVariance (N : ℕ) (c d : Counts) :
    (∑ j : Bool × Fin N, weightedVariance (gapCoefficient N c d) j) =
      blockSum N c.n1 d.n1 (fun j => updateVariance (j.val : ℝ≥0)) +
        blockSum N c.n2 d.n2 (fun j => updateVariance (j.val : ℝ≥0)) := by
  rw [Fintype.sum_prod_type, Fintype.sum_bool]
  unfold weightedVariance gapCoefficient
  simp only [↓reduceIte, Bool.false_eq_true]
  congr 1 <;> apply Finset.sum_congr rfl <;> intro j _ <;>
    split_ifs <;> simp [squareWeight]

theorem allocationVariance_eq_blockSum (N : ℕ) (s : List Bool) (c : Counts)
    (hN1 : (finalCounts s c).n1 ≤ N) (hN2 : (finalCounts s c).n2 ≤ N) :
    allocationVariance s c =
      blockSum N c.n1 (finalCounts s c).n1 (fun j => updateVariance (j.val : ℝ≥0)) +
        blockSum N c.n2 (finalCounts s c).n2 (fun j => updateVariance (j.val : ℝ≥0)) := by
  induction s generalizing c with
  | nil => simp [allocationVariance, finalCounts, blockSum_empty]
  | cons i s ih =>
    cases i
    · change (finalCounts s (rightStep c)).n1 ≤ N at hN1
      change (finalCounts s (rightStep c)).n2 ≤ N at hN2
      change updateVariance (c.n2 : ℝ≥0) + allocationVariance s (rightStep c) = _
      rw [ih (rightStep c) hN1 hN2]
      have hm := finalCounts_monotone s (rightStep c)
      have hn : c.n2 < (finalCounts s (rightStep c)).n2 := by
        change c.n1 ≤ (finalCounts s (rightStep c)).n1 ∧ c.n2 + 1 ≤ (finalCounts s (rightStep c)).n2 at hm
        omega
      change updateVariance (c.n2 : ℝ≥0) +
        (blockSum N c.n1 (finalCounts s (rightStep c)).n1 _ +
         blockSum N (c.n2 + 1) (finalCounts s (rightStep c)).n2 _) =
        blockSum N c.n1 (finalCounts s (rightStep c)).n1 _ +
         blockSum N c.n2 (finalCounts s (rightStep c)).n2 _
      rw [blockSum_head N c.n2 (finalCounts s (rightStep c)).n2 _ hn hN2]
      dsimp only
      exact add_left_comm _ _ _
    · change (finalCounts s (leftStep c)).n1 ≤ N at hN1
      change (finalCounts s (leftStep c)).n2 ≤ N at hN2
      change updateVariance (c.n1 : ℝ≥0) + allocationVariance s (leftStep c) = _
      rw [ih (leftStep c) hN1 hN2]
      have hm := finalCounts_monotone s (leftStep c)
      have hn : c.n1 < (finalCounts s (leftStep c)).n1 := by
        change c.n1 + 1 ≤ (finalCounts s (leftStep c)).n1 ∧ c.n2 ≤ (finalCounts s (leftStep c)).n2 at hm
        omega
      change updateVariance (c.n1 : ℝ≥0) +
        (blockSum N (c.n1 + 1) (finalCounts s (leftStep c)).n1 _ +
         blockSum N c.n2 (finalCounts s (leftStep c)).n2 _) =
        blockSum N c.n1 (finalCounts s (leftStep c)).n1 _ +
         blockSum N c.n2 (finalCounts s (leftStep c)).n2 _
      rw [blockSum_head N c.n1 (finalCounts s (leftStep c)).n1 _ hn hN1]
      dsimp only
      exact (add_assoc _ _ _).symm

theorem terminal_gap_law (N : ℕ) (s : List Bool) (c : Counts) (m1 m2 : ℝ)
    (hN1 : (finalCounts s c).n1 ≤ N) (hN2 : (finalCounts s c).n2 ≤ N) :
    HasLaw (fun x : Bool × Fin N → ℝ =>
      (terminalMeans s c (m1, m2) (finiteStream N x)).1 -
        (terminalMeans s c (m1, m2) (finiteStream N x)).2)
      (gaussianReal (m1 - m2) (allocationVariance s c)) (streamLaw N) := by
  have hg := weightedSum_law (gapCoefficient N c (finalCounts s c))
  rw [gap_weightedVariance, ← allocationVariance_eq_blockSum N s c hN1 hN2] at hg
  have hshift : HasLaw (fun z : ℝ => (m1 - m2) + z)
      (gaussianReal (m1 - m2) (allocationVariance s c)) (gaussianReal 0 (allocationVariance s c)) := by
    constructor
    · exact (by fun_prop : AEMeasurable (fun z : ℝ => (m1 - m2) + z) (gaussianReal 0 (allocationVariance s c)))
    · simpa using gaussianReal_map_const_add (μ := (0 : ℝ)) (v := allocationVariance s c) (m1 - m2)
  have h := hshift.fun_comp hg
  convert h using 1
  funext x
  rw [terminalMeans_eq_blockSum N s c (m1, m2) x hN1 hN2, gap_weightedSum]
  dsimp
  ring

end
end Junpei.CommonSample
