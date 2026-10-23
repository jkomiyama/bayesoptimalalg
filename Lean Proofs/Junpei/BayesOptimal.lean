import Junpei.FinitePosterior

namespace Junpei.BayesOptimal

open Junpei.FiniteGaussian Junpei.FinitePosterior

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

def EBI (k : ℕ) (c : Precisions A) (m : Means A) (i : A) : ℝ :=
  loss k c m - actionLoss k c m i

theorem EBI_eq (k : ℕ) (c : Precisions A) (m : Means A) (i : A) (hc : 0<c i) :
    EBI k c m i = action k c m i - value k c m := by
  rw [EBI, actionLoss_eq k c m i hc]
  unfold loss
  ring

theorem argmax_EBI_iff_argmin_actionLoss (k : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    (∀ j, EBI k c m j ≤ EBI k c m i) ↔ (∀ j, actionLoss k c m i ≤ actionLoss k c m j) := by
  simp only [EBI, sub_le_sub_iff_left]

theorem actionLoss_eq_loss_iff (k : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (hc : 0<c i) : actionLoss k c m i = loss (k+1) c m ↔
      action k c m i = value (k+1) c m := by
  rw [actionLoss_eq k c m i hc]
  unfold loss
  exact sub_right_inj

theorem actionLoss_minimal (k : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ i, 0<c i) (j : A) : loss (k+1) c m ≤ actionLoss k c m j := by
  rw [actionLoss_eq k c m j (hc j)]
  unfold loss
  exact sub_le_sub_left (action_le_value k c m j) _

theorem chosen_actionLoss_eq (k : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ i, 0<c i) : actionLoss k c m (choose k c m) = loss (k+1) c m :=
  (actionLoss_eq_loss_iff k c m _ (hc _)).mpr (choose_optimal k c m)

theorem optimal_action_iff_EBI_max (k : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (hc : ∀ j, 0<c j) :
    action k c m i = value (k+1) c m ↔ ∀ j, EBI k c m j ≤ EBI k c m i := by
  simp_rw [EBI_eq k c m _ (hc _)]
  simp only [sub_le_sub_iff_right]
  constructor
  · intro h j
    rw [h]
    exact action_le_value k c m j
  · intro h
    apply le_antisymm (action_le_value k c m i)
    rw [value_succ]
    exact maxAll_le _ _ h

omit [DecidableEq A] in
theorem terminal_recommendation_iff (c : Precisions A) (m : Means A) (i : A) :
    (∀ j, terminalRegret c m i ≤ terminalRegret c m j) ↔ m i = maxAll m := by
  simp_rw [terminalRegret_eq, sub_le_sub_iff_left]
  constructor
  · intro h
    exact le_antisymm (coordinate_le_maxAll m i) (maxAll_le m (m i) h)
  · intro h j
    rw [h]
    exact coordinate_le_maxAll m j

theorem optimal_posterior_loss (k : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ i, 0<c i) : expectedRegret (optimal k c m) c m = loss k c m := by
  rw [expectedRegret_eq _ _ _ (optimal_admissible k c m) hc, optimal_utility]
  rfl

end
end Junpei.BayesOptimal
