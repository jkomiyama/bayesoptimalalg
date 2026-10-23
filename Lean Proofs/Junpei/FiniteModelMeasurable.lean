import Junpei.FinitePosterior

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FiniteModelMeasurable

open Junpei.FiniteGaussian Junpei.FinitePosterior

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

def listValue (a0 : A) : List A → Means A → ℝ
  | [], m => m a0
  | a :: as, m => max (m a) (listValue a0 as m)

def listChoice (a0 : A) : List A → Means A → A
  | [], _ => a0
  | a :: as, m => if listValue a0 as m ≤ m a then a else listChoice a0 as m

omit [Fintype A] [Nonempty A] [DecidableEq A] in
theorem listValue_continuous (a0 : A) (as : List A) : Continuous (listValue a0 as) := by
  induction as with
  | nil => exact continuous_apply a0
  | cons a as ih => exact (continuous_apply a).max ih

omit [Fintype A] [Nonempty A] [DecidableEq A] in
theorem listChoice_attains (a0 : A) (as : List A) (m : Means A) :
    m (listChoice a0 as m) = listValue a0 as m := by
  induction as with
  | nil => rfl
  | cons a as ih =>
    change m (if listValue a0 as m ≤ m a then a else listChoice a0 as m) =
      max (m a) (listValue a0 as m)
    split_ifs with h
    · exact (max_eq_left h).symm
    · rw [ih, max_eq_right (le_of_not_ge h)]

omit [DecidableEq A] in
theorem listValue_le_maxAll (a0 : A) (as : List A) (m : Means A) :
    listValue a0 as m ≤ maxAll m := by
  induction as with
  | nil => exact coordinate_le_maxAll m a0
  | cons a as ih => exact max_le (coordinate_le_maxAll m a) ih

omit [Fintype A] [Nonempty A] [DecidableEq A] in
theorem coordinate_le_listValue (a0 : A) (as : List A) (m : Means A)
    (a : A) (ha : a ∈ as) : m a ≤ listValue a0 as m := by
  induction as with
  | nil => simp at ha
  | cons b as ih =>
    rcases List.mem_cons.mp ha with h | h
    · subst a; exact le_max_left _ _
    · exact (ih h).trans (le_max_right _ _)

def firstMax (m : Means A) : A :=
  listChoice (Classical.choice (inferInstance : Nonempty A)) Finset.univ.toList m

omit [DecidableEq A] in
theorem firstMax_attains (m : Means A) : m (firstMax m) = maxAll m := by
  change m (listChoice _ Finset.univ.toList m) = maxAll m
  rw [listChoice_attains _ _ m]
  apply le_antisymm (listValue_le_maxAll _ _ _)
  apply maxAll_le
  intro a
  exact coordinate_le_listValue _ _ _ a (by simp)

section Measurable
variable [MeasurableSpace A]

omit [Nonempty A] [DecidableEq A] in
theorem listChoice_measurable (a0 : A) (as : List A) : Measurable (listChoice a0 as) := by
  induction as with
  | nil => exact measurable_const
  | cons a as ih =>
    exact Measurable.ite
      (isClosed_le (listValue_continuous a0 as) (continuous_apply a)).measurableSet
      measurable_const ih

omit [DecidableEq A] in
theorem firstMax_measurable : Measurable (firstMax (A := A)) :=
  listChoice_measurable _ _

def chooseFirst (k : ℕ) (c : Precisions A) (m : Means A) : A :=
  firstMax (action k c m)

omit [MeasurableSpace A] in
theorem chooseFirst_optimal (k : ℕ) (c : Precisions A) (m : Means A) :
    action k c m (chooseFirst k c m) = value (k + 1) c m := firstMax_attains _

/-- A Borel optimal arm selector for every fixed proper Gaussian posterior
precision vector. Ties are resolved by a fixed finite list. -/
theorem chooseFirst_measurable (k : ℕ) (c : Precisions A) :
    Measurable (chooseFirst k c) := by
  apply firstMax_measurable.comp
  apply measurable_pi_lambda
  intro a
  exact (integral_shift_lipschitz (value_lipschitz k (advance c a))
    a (innovation c a)).continuous.measurable

def optimalFirst : (k : ℕ) → Precisions A → Means A → Strategy (A := A) k
  | 0, _, m => .stop (firstMax m)
  | k + 1, c, m =>
      let i := chooseFirst k c m
      .sample i (fun z => optimalFirst k (advance c i) (shift m i z))

omit [MeasurableSpace A] in
theorem optimalFirst_utility (k : ℕ) (c : Precisions A) (m : Means A) :
    utility (optimalFirst k c m) c m = value k c m := by
  induction k generalizing c m with
  | zero => exact firstMax_attains m
  | succ k ih => simp only [optimalFirst, utility, ih]; exact chooseFirst_optimal k c m

omit [MeasurableSpace A] in
theorem optimalFirst_admissible (k : ℕ) (c : Precisions A) (m : Means A) :
    Admissible (optimalFirst k c m) c m := by
  induction k generalizing c m with
  | zero => trivial
  | succ k ih =>
    constructor
    · intro z; exact ih _ _
    · simp only [optimalFirst_utility]
      exact action_integrable k c m (chooseFirst k c m)

omit [MeasurableSpace A] in
theorem optimalFirst_bayes_optimality {k : ℕ} (s : Strategy (A := A) k)
    (c : Precisions A) (m : Means A) (hs : Admissible s c m) (hc : ∀ a, 0 < c a) :
    expectedRegret (optimalFirst k c m) c m ≤ expectedRegret s c m := by
  rw [expectedRegret_eq _ _ _ (optimalFirst_admissible k c m) hc,
    expectedRegret_eq s c m hs hc, optimalFirst_utility]
  exact sub_le_sub_left (utility_le_value s c m hs) _

end Measurable

end
end Junpei.FiniteModelMeasurable
