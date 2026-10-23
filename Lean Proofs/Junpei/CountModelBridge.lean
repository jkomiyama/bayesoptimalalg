import Draft0.PosteriorRegret
import Junpei.FinitePosterior

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.CountModelBridge

open Draft0.BayesianModel

noncomputable section

/-- The integer-count model is the finite Gaussian model with these precisions. -/
def natPrecisions (c : Counts) : FiniteGaussian.Precisions Arm :=
  fun a => (c a : ℝ≥0)

theorem natPrecisions_positive (c : Counts) (hc : ∀ a, 0 < c a) :
    ∀ a, 0 < natPrecisions c a := by
  intro a
  change (0 : ℝ≥0) < (c a : ℝ≥0)
  exact_mod_cast hc a

theorem advance_natPrecisions (c : Counts) (a : Arm) :
    FiniteGaussian.advance (natPrecisions c) a = natPrecisions (advance c a) := by
  funext b
  by_cases h : b = a <;>
    simp [FiniteGaussian.advance, natPrecisions, advance, h]

theorem shift_eq (m : Means) (a : Arm) (z : ℝ) :
    FiniteGaussian.shift m a z = shift m a z := rfl

theorem innovation_eq (c : Counts) (a : Arm) :
    FiniteGaussian.innovation (natPrecisions c) a = innovation c a := rfl

theorem maxAll_eq_terminal (m : Means) :
    FiniteGaussian.maxAll m = terminal m := by
  apply le_antisymm
  · apply FiniteGaussian.maxAll_le
    intro a
    fin_cases a
    · exact le_max_left _ _
    · exact (le_max_left _ _).trans (le_max_right _ _)
    · exact (le_max_right _ _).trans (le_max_right _ _)
  · unfold terminal
    apply max_le (FiniteGaussian.coordinate_le_maxAll m 0)
    exact max_le (FiniteGaussian.coordinate_le_maxAll m 1)
      (FiniteGaussian.coordinate_le_maxAll m 2)

theorem value_eq (k : ℕ) (c : Counts) (m : Means) :
    FiniteGaussian.value k (natPrecisions c) m = value k c m := by
  induction k generalizing c m with
  | zero => exact maxAll_eq_terminal m
  | succ k ih =>
    rw [FiniteGaussian.value_succ, maxAll_eq_terminal, value_succ]
    have ha (a : Arm) :
        FiniteGaussian.action k (natPrecisions c) m a = action k c m a := by
      unfold FiniteGaussian.action action
      simp only [advance_natPrecisions, shift_eq, innovation_eq, ih]
    simp only [terminal, ha]

theorem action_eq (k : ℕ) (c : Counts) (m : Means) (a : Arm) :
    FiniteGaussian.action k (natPrecisions c) m a = action k c m a := by
  unfold FiniteGaussian.action action
  simp only [advance_natPrecisions, shift_eq, innovation_eq, value_eq]

/-- Equality of generic action loss and generic optimal loss is exactly the
integer-count Bellman optimality predicate used by the main theorem. -/
theorem actionLoss_eq_loss_iff (k : ℕ) (c : Counts) (m : Means) (a : Arm)
    (hc : 0 < c a) :
    FinitePosterior.actionLoss k (natPrecisions c) m a =
      FinitePosterior.loss (k + 1) (natPrecisions c) m ↔
    action k c m a = value (k + 1) c m := by
  rw [FinitePosterior.actionLoss_eq k (natPrecisions c) m a (by
    change (0 : ℝ≥0) < (c a : ℝ≥0)
    exact_mod_cast hc)]
  unfold FinitePosterior.loss
  rw [action_eq, value_eq]
  exact sub_right_inj

end
end Junpei.CountModelBridge
