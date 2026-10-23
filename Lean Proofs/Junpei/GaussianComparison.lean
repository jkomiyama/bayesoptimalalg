import Draft0.TwoArmBellman

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace JunpeiComparison.Gaussian

open Draft0.Gaussian Draft0.Counts Draft0.TwoArmBellman

noncomputable section

/-- This is the revised exclusion integrand, not the old best-arm contribution. -/
theorem exclusion_loss_pair_bound (a b x : ℝ) :
    max (x - min a b) 0 ≤ max (x - a) 0 + max (x - b) 0 := by
  by_cases h : a ≤ b
  · rw [min_eq_left h]
    exact le_add_of_nonneg_right (le_max_right _ _)
  · rw [min_eq_right (le_of_not_ge h)]
    exact le_add_of_nonneg_left (le_max_right _ _)

/-- The revised exclusion quantity can be positive when arm 3 is not best. -/
theorem exclusion_definition_differs :
    max ((1 : ℝ) - min 0 2) 0 = 1 ∧ ¬ (max (0 : ℝ) 2 < 1) := by
  norm_num

def newG1 (gap : ℝ) (vL vR : ℝ≥0) : ℝ :=
  (gaussianReal 0 vL (Icc (3 * |gap| / 2) (2 * |gap|))).toReal *
    (gaussianReal 0 vR (Ici |gap|)).toReal * |gap|

theorem newG1_nonnegative (gap : ℝ) (vL vR : ℝ≥0) :
    0 ≤ newG1 gap vL vR := by
  unfold newG1
  positivity

theorem newG1_at_zero (vL vR : ℝ≥0) : newG1 0 vL vR = 0 := by
  simp [newG1]

theorem newG1_positive (gap : ℝ) (vL vR : ℝ≥0)
    (hgap : gap ≠ 0) (hl : 0 < vL) (hr : 0 < vR) :
    0 < newG1 gap vL vR := by
  have hd : 0 < |gap| := abs_pos.mpr hgap
  exact mul_pos
    (mul_pos (gaussian_Icc_probability_pos 0 _ _ vL hl (by linarith))
      (gaussian_Ici_probability_pos 0 _ vR hr)) hd

def newG2 (gap : ℝ) (vL vG vR : ℝ≥0) : ℝ :=
  (gaussianReal 0 vL (Icc (|gap| / 4) (|gap| / 3))).toReal *
    (gaussianReal 0 vG (Icc (9 * |gap| / 8) (4 * |gap| / 3))).toReal *
    (gaussianReal 0 vR (Ici (3 * |gap| / 4))).toReal * (2 * |gap| / 3)

theorem newG2_nonnegative (gap : ℝ) (vL vG vR : ℝ≥0) :
    0 ≤ newG2 gap vL vG vR := by
  unfold newG2
  positivity

theorem newG2_at_zero (vL vG vR : ℝ≥0) : newG2 0 vL vG vR = 0 := by
  simp [newG2]

theorem newG2_positive (gap : ℝ) (vL vG vR : ℝ≥0)
    (hgap : gap ≠ 0) (hl : 0 < vL) (hg : 0 < vG) (hr : 0 < vR) :
    0 < newG2 gap vL vG vR := by
  have hd : 0 < |gap| := abs_pos.mpr hgap
  exact mul_pos (mul_pos
    (mul_pos (gaussian_Icc_probability_pos 0 _ _ vL hl (by linarith))
      (gaussian_Icc_probability_pos 0 _ _ vG hg (by linarith)))
    (gaussian_Ici_probability_pos 0 _ vR hr)) (by positivity)

/-- The scalar difference used in new Lemma 15, PDF page 30. -/
def pointwiseImprovement (d l r : ℝ) : ℝ :=
  max (r + l - d) 0 - max (r - |d - l|) 0

theorem lemma15_equal_before_crossing (d l r : ℝ) (hl : l ≤ d) :
    pointwiseImprovement d l r = 0 := by
  unfold pointwiseImprovement
  rw [abs_of_nonneg (sub_nonneg.mpr hl)]
  have he : r + l - d = r - (d - l) := by ring
  rw [he, sub_self]

theorem lemma15_pointwise_nonnegative (d l r : ℝ) :
    0 ≤ pointwiseImprovement d l r := by
  by_cases hl : l ≤ d
  · rw [lemma15_equal_before_crossing d l r hl]
  · unfold pointwiseImprovement
    have hl' : d < l := lt_of_not_ge hl
    rw [abs_of_neg (sub_neg.mpr hl')]
    apply sub_nonneg.mpr
    apply max_le_max
    · linarith
    · exact le_rfl

/-- The changed interval [3d/2,2d] makes the printed event lower bound valid. -/
theorem lemma15_repaired_event_bound (d l r : ℝ) (hd : 0 ≤ d)
    (hlow : 3 * d / 2 ≤ l) (hupp : l ≤ 2 * d) (hr : d ≤ r) :
    d ≤ pointwiseImprovement d l r := by
  have hcross : d ≤ l := by linarith
  unfold pointwiseImprovement
  rw [abs_of_nonpos (sub_nonpos.mpr hcross),
    max_eq_left (by linarith : 0 ≤ r + l - d),
    max_eq_left (by linarith : 0 ≤ r - -(d - l))]
  linarith

/-- The two intervals in new Lemma 16, PDF page 32, satisfy all stated
conditions and give its printed pointwise factor 2d/3. -/
theorem lemma16_repaired_event_bound (d l g r : ℝ) (hd : 0 ≤ d)
    (hlow : d / 4 ≤ l) (hupp : l ≤ d / 3)
    (hglow : 9 * d / 8 ≤ g) (hgupp : g ≤ 4 * d / 3) (hr : 3 * d / 4 ≤ r) :
    2 * d / 3 ≤ pointwiseImprovement (d - l) g r := by
  have hδ : 0 ≤ d - l := by linarith
  have hg1 : 3 * (d - l) / 2 ≤ g := by linarith
  have hg2 : g ≤ 2 * (d - l) := by linarith
  have hrδ : d - l ≤ r := by linarith
  exact (show 2 * d / 3 ≤ d - l by linarith).trans
    (lemma15_repaired_event_bound (d - l) g r hδ hg1 hg2 hrδ)

theorem cumulativeVariance_swap (k n1 n2 n3 : ℕ) :
    cumulativeVariance k ⟨n1, n2, n3⟩ = cumulativeVariance k ⟨n2, n1, n3⟩ := by
  induction k generalizing n1 n2 n3 with
  | zero => rfl
  | succ k ih =>
    simp only [cumulativeVariance, leftStep, rightStep]
    rw [ih (n1 + 1) n2 n3, ih n1 (n2 + 1) n3]
    exact max_comm _ _

/-- New Lemma 17 says every smaller-count arm is optimal. This checks the
equal-count case in addition to draft0's fixed left tie rule. -/
theorem equal_count_actions_equal (k n n3 : ℕ) (m1 m2 : ℝ) :
    leftAction k m1 m2 ⟨n, n, n3⟩ = rightAction k m1 m2 ⟨n, n, n3⟩ := by
  rw [leftAction_eq, rightAction_eq]
  congr 1
  congr 1
  simp only [leftStep, rightStep]
  rw [cumulativeVariance_swap k (n + 1) n n3]

theorem equal_count_right_action_optimal (k n n3 : ℕ) (m1 m2 : ℝ) (hn : 0 < n) :
    payoff (k + 1) m1 m2 ⟨n, n, n3⟩ = rightAction k m1 m2 ⟨n, n, n3⟩ := by
  rw [smaller_count_left_optimal k m1 m2 ⟨n, n, n3⟩ hn hn (by rfl),
    equal_count_actions_equal]

#check lemma15_repaired_event_bound
#check lemma16_repaired_event_bound
#check equal_count_right_action_optimal
#print axioms lemma15_equal_before_crossing
#print axioms lemma15_pointwise_nonnegative
#print axioms lemma15_repaired_event_bound
#print axioms lemma16_repaired_event_bound
#print axioms equal_count_right_action_optimal
#print axioms exclusion_loss_pair_bound
#print axioms newG1_positive
#print axioms newG2_positive

end

end JunpeiComparison.Gaussian
