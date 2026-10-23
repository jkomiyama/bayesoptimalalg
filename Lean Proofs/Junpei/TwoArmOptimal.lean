import Junpei.GaussianComparison
import Junpei.FinitePosterior
import Draft0.PredictiveLaw

/-!
Exact scope of revised Lemma 17 (PDF page 33): positive INTEGER precisions.
The Bellman recursion integrates the actual normal posterior innovations;
its maximum ranges over both possible actions at every history.
-/

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.TwoArmOptimal

open Draft0.Gaussian Draft0.TwoArmBellman Draft0.Counts

noncomputable section

def pairCounts (n1 n2 : ℕ) : Counts := ⟨n1, n2, 0⟩
def pairMeans (m1 m2 : ℝ) : Fin 2 → ℝ := ![m1, m2]
def pairPrecisions (n1 n2 : ℕ) : Fin 2 → ℝ≥0 := ![(n1 : ℝ≥0), (n2 : ℝ≥0)]

theorem maxAll_pair (m1 m2 : ℝ) :
    FiniteGaussian.maxAll (pairMeans m1 m2) = max m1 m2 := by
  apply le_antisymm
  · apply FiniteGaussian.maxAll_le
    intro i
    fin_cases i
    · exact le_max_left _ _
    · exact le_max_right _ _
  · exact max_le (FiniteGaussian.coordinate_le_maxAll (pairMeans m1 m2) 0)
      (FiniteGaussian.coordinate_le_maxAll (pairMeans m1 m2) 1)

theorem advance_pair_left (n1 n2 : ℕ) :
    FiniteGaussian.advance (pairPrecisions n1 n2) 0 = pairPrecisions (n1 + 1) n2 := by
  ext i
  fin_cases i <;> simp [FiniteGaussian.advance, pairPrecisions]

theorem advance_pair_right (n1 n2 : ℕ) :
    FiniteGaussian.advance (pairPrecisions n1 n2) 1 = pairPrecisions n1 (n2 + 1) := by
  ext i
  fin_cases i <;> simp [FiniteGaussian.advance, pairPrecisions]

theorem shift_pair_left (m1 m2 z : ℝ) :
    FiniteGaussian.shift (pairMeans m1 m2) 0 z = pairMeans (m1 + z) m2 := by
  ext i
  fin_cases i <;> simp [FiniteGaussian.shift, pairMeans]

theorem shift_pair_right (m1 m2 z : ℝ) :
    FiniteGaussian.shift (pairMeans m1 m2) 1 z = pairMeans m1 (m2 + z) := by
  ext i
  fin_cases i <;> simp [FiniteGaussian.shift, pairMeans]

theorem value_eq_twoArm (k n1 n2 : ℕ) (m1 m2 : ℝ) :
    FiniteGaussian.value k (pairPrecisions n1 n2) (pairMeans m1 m2) =
      payoff k m1 m2 (pairCounts n1 n2) := by
  induction k generalizing n1 n2 m1 m2 with
  | zero => exact maxAll_pair m1 m2
  | succ k ih =>
    rw [FiniteGaussian.value_succ]
    have ha : FiniteGaussian.action k (pairPrecisions n1 n2) (pairMeans m1 m2) =
        pairMeans (leftAction k m1 m2 (pairCounts n1 n2))
          (rightAction k m1 m2 (pairCounts n1 n2)) := by
      ext i
      fin_cases i
      · change (∫ z : ℝ, FiniteGaussian.value k
            (FiniteGaussian.advance (pairPrecisions n1 n2) 0)
            (FiniteGaussian.shift (pairMeans m1 m2) 0 z)
            ∂gaussianReal 0 (updateVariance n1)) = leftAction k m1 m2 (pairCounts n1 n2)
        rw [advance_pair_left]
        simp_rw [shift_pair_left, ih]
        rfl
      · change (∫ z : ℝ, FiniteGaussian.value k
            (FiniteGaussian.advance (pairPrecisions n1 n2) 1)
            (FiniteGaussian.shift (pairMeans m1 m2) 1 z)
            ∂gaussianReal 0 (updateVariance n2)) = rightAction k m1 m2 (pairCounts n1 n2)
        rw [advance_pair_right]
        simp_rw [shift_pair_right, ih]
        rfl
    rw [ha, maxAll_pair]
    rfl

theorem action_left_eq (k n1 n2 : ℕ) (m1 m2 : ℝ) :
    FiniteGaussian.action k (pairPrecisions n1 n2) (pairMeans m1 m2) 0 =
      leftAction k m1 m2 (pairCounts n1 n2) := by
  change (∫ z : ℝ, FiniteGaussian.value k
      (FiniteGaussian.advance (pairPrecisions n1 n2) 0)
      (FiniteGaussian.shift (pairMeans m1 m2) 0 z)
      ∂gaussianReal 0 (updateVariance n1)) = _
  rw [advance_pair_left]
  simp_rw [shift_pair_left, value_eq_twoArm]
  rfl

theorem action_right_eq (k n1 n2 : ℕ) (m1 m2 : ℝ) :
    FiniteGaussian.action k (pairPrecisions n1 n2) (pairMeans m1 m2) 1 =
      rightAction k m1 m2 (pairCounts n1 n2) := by
  change (∫ z : ℝ, FiniteGaussian.value k
      (FiniteGaussian.advance (pairPrecisions n1 n2) 1)
      (FiniteGaussian.shift (pairMeans m1 m2) 1 z)
      ∂gaussianReal 0 (updateVariance n2)) = _
  rw [advance_pair_right]
  simp_rw [shift_pair_right, value_eq_twoArm]
  rfl

/-- Every minimum-count action achieves the actual adaptive Bellman optimum,
including BOTH actions when the counts are equal. -/
theorem lemma17_action_optimal (k n1 n2 : ℕ) (m1 m2 : ℝ)
    (h1 : 0 < n1) (h2 : 0 < n2) (i : Fin 2)
    (hi : ∀ j, pairPrecisions n1 n2 i ≤ pairPrecisions n1 n2 j) :
    FiniteGaussian.action k (pairPrecisions n1 n2) (pairMeans m1 m2) i =
      FiniteGaussian.value (k + 1) (pairPrecisions n1 n2) (pairMeans m1 m2) := by
  rw [value_eq_twoArm]
  fin_cases i
  · change FiniteGaussian.action k (pairPrecisions n1 n2) (pairMeans m1 m2) 0 = _
    rw [action_left_eq]
    apply (smaller_count_left_optimal k m1 m2 (pairCounts n1 n2) h1 h2 _).symm
    have h := hi 1
    change (n1 : ℝ≥0) ≤ (n2 : ℝ≥0) at h
    exact_mod_cast h
  · change FiniteGaussian.action k (pairPrecisions n1 n2) (pairMeans m1 m2) 1 = _
    rw [action_right_eq]
    have h : n2 ≤ n1 := by
      have hh := hi 0
      change (n2 : ℝ≥0) ≤ (n1 : ℝ≥0) at hh
      exact_mod_cast hh
    by_cases he : n1 = n2
    · subst n2
      exact (JunpeiComparison.Gaussian.equal_count_right_action_optimal
        k n1 0 m1 m2 h1).symm
    · exact (smaller_count_right_optimal k m1 m2 (pairCounts n1 n2) h1 h2
        (show n2 < n1 by omega)).symm

/-- Revised Lemma 17, with L_i interpreted by the actual posterior-regret
integral of FinitePosterior; no optimality inequality is assumed. -/
theorem lemma17 (k n1 n2 : ℕ) (m1 m2 : ℝ)
    (h1 : 0 < n1) (h2 : 0 < n2) (i : Fin 2)
    (hi : ∀ j, pairPrecisions n1 n2 i ≤ pairPrecisions n1 n2 j) (j : Fin 2) :
    FinitePosterior.actionLoss k (pairPrecisions n1 n2) (pairMeans m1 m2) i ≤
      FinitePosterior.actionLoss k (pairPrecisions n1 n2) (pairMeans m1 m2) j := by
  have hc (a : Fin 2) : 0 < pairPrecisions n1 n2 a := by
    fin_cases a
    · change (0 : ℝ≥0) < (n1 : ℝ≥0)
      exact_mod_cast h1
    · change (0 : ℝ≥0) < (n2 : ℝ≥0)
      exact_mod_cast h2
  rw [FinitePosterior.actionLoss_eq _ _ _ _ (hc i),
    FinitePosterior.actionLoss_eq _ _ _ _ (hc j),
    lemma17_action_optimal k n1 n2 m1 m2 h1 h2 i hi]
  exact sub_le_sub_left (FiniteGaussian.action_le_value _ _ _ j) _

/-- The same statement for an arbitrary two-arm mean vector and arbitrary
positive integer precision vector, with no vector-normal-form assumptions. -/
theorem lemma17_every_posterior (k : ℕ) (n : Fin 2 → ℕ) (m : Fin 2 → ℝ)
    (hn : ∀ a, 0 < n a) (i : Fin 2) (hi : ∀ j, n i ≤ n j) (j : Fin 2) :
    FinitePosterior.actionLoss k (fun a => (n a : ℝ≥0)) m i ≤
      FinitePosterior.actionLoss k (fun a => (n a : ℝ≥0)) m j := by
  have hp : (fun a => (n a : ℝ≥0)) = pairPrecisions (n 0) (n 1) := by
    ext a
    fin_cases a <;> rfl
  have hm : m = pairMeans (m 0) (m 1) := by
    ext a
    fin_cases a <;> rfl
  rw [hp, hm]
  apply lemma17 k (n 0) (n 1) (m 0) (m 1) (hn 0) (hn 1) i _ j
  intro a
  rw [← hp]
  change (n i : ℝ≥0) ≤ (n a : ℝ≥0)
  exact_mod_cast hi a

/-- The deterministic smaller-count schedule attains the adaptive optimum. -/
theorem lemma17_balanced_policy (k n1 n2 : ℕ) (m1 m2 : ℝ)
    (h1 : 0 < n1) (h2 : 0 < n2) :
    balancedPayoff k m1 m2 (pairCounts n1 n2) =
      FiniteGaussian.value k (pairPrecisions n1 n2) (pairMeans m1 m2) := by
  rw [value_eq_twoArm]
  exact balanced_policy_optimal _ _ _ _ h1 h2

/-- Static allocation step in the printed proof: reciprocal terminal
variance is minimized by balancing, subject to integer count constraints. -/
theorem lemma17_minimal_terminal_variance (k n1 n2 a b : ℕ)
    (h1 : 0 < n1) (h2 : 0 < n2) (ha : n1 ≤ a) (hb : n2 ≤ b)
    (hs : a + b = n1 + n2 + k) :
    variance (balancedRun k (pairCounts n1 n2)) ≤ posteriorVariance a b :=
  balancedRun_variance_le_allocation _ _ _ _ h1 h2 ha hb hs

/-- The exact cumulative normal variance is initial minus terminal
posterior variance, as in the deterministic-allocation proof. -/
theorem lemma17_variance_identity (k n1 n2 : ℕ) (h1 : 0 < n1) (h2 : 0 < n2) :
    (cumulativeVariance k (pairCounts n1 n2) : ℝ) =
      transitionVariance k (pairCounts n1 n2) :=
  cumulativeVariance_eq_transitionVariance _ _ h1 h2

theorem lemma17_gaussian_forecast (k n1 n2 : ℕ) (delta : ℝ) :
    Draft0.PredictiveLaw.balancedPredictiveLaw k (pairCounts n1 n2) delta =
      gaussianReal delta (Draft0.PredictiveLaw.accumulatedVariance k (pairCounts n1 n2)) :=
  Draft0.PredictiveLaw.balancedPredictiveLaw_gaussian _ _ _

end
end Junpei.TwoArmOptimal
