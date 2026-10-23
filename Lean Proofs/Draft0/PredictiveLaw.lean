import Draft0.Counts
import Draft0.Gaussian

/-!
The forecast law for the corrected deterministic balanced schedule, defined
by convolving the centered Gaussian posterior innovations of Section 5.1.
Its variance is exactly the difference of initial and final posterior
variances. A separate model-identification argument is needed to identify
this measure recursion with the paper's conditional random history.
-/

open MeasureTheory ProbabilityTheory
open scoped MeasureTheory NNReal ENNReal

namespace Draft0.PredictiveLaw

open Draft0.Counts

noncomputable section

def sampledCount (c : Counts) : ℕ := if c.n1 ≤ c.n2 then c.n1 else c.n2

def innovationVariance (c : Counts) : ℝ≥0 :=
  ((sampledCount c : ℝ≥0) * ((sampledCount c : ℝ≥0) + 1))⁻¹

def accumulatedVariance : ℕ → Counts → ℝ≥0
  | 0, _ => 0
  | k + 1, c => innovationVariance c + accumulatedVariance k (balancedStep c)

/-- The actual measure-valued forecast obtained from independent Gaussian innovations. -/
def balancedPredictiveLaw : ℕ → Counts → ℝ → Measure ℝ
  | 0, _, δ => gaussianReal δ 0
  | k + 1, c, δ => gaussianReal 0 (innovationVariance c) ∗
      balancedPredictiveLaw k (balancedStep c) δ

theorem balancedPredictiveLaw_gaussian (k : ℕ) (c : Counts) (δ : ℝ) :
    balancedPredictiveLaw k c δ = gaussianReal δ (accumulatedVariance k c) := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
      rw [balancedPredictiveLaw, ih, gaussianReal_conv_gaussianReal, zero_add]
      rfl

theorem innovationVariance_coe (c : Counts) :
    (innovationVariance c : ℝ) = varianceDecrement (sampledCount c) := by
  simp [innovationVariance, varianceDecrement, one_div]

theorem transitionVariance_succ (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    transitionVariance (k + 1) c =
      varianceDecrement (sampledCount c) + transitionVariance k (balancedStep c) := by
  unfold transitionVariance
  rw [balancedRun]
  unfold sampledCount
  split_ifs with h
  · rw [← varianceDecrement_identity c.n1 h1]
    simp only [balancedStep, if_pos h, Nat.cast_add, Nat.cast_one]
    ring
  · rw [← varianceDecrement_identity c.n2 h2]
    simp only [balancedStep, if_neg h, Nat.cast_add, Nat.cast_one]
    ring

/-- Every innovation variance telescopes, including the off-by-one terminal counts. -/
theorem accumulatedVariance_coe (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    (accumulatedVariance k c : ℝ) = transitionVariance k c := by
  induction k generalizing c with
  | zero => simp [accumulatedVariance, transitionVariance, balancedRun]
  | succ k ih =>
      have hm := balancedStep_monotone c
      have hnext1 : 0 < (balancedStep c).n1 := h1.trans_le hm.1
      have hnext2 : 0 < (balancedStep c).n2 := h2.trans_le hm.2
      rw [accumulatedVariance, NNReal.coe_add, innovationVariance_coe,
        ih (balancedStep c) hnext1 hnext2, transitionVariance_succ k c h1 h2]

/-- A zero-round forecast is a Dirac measure, explaining the terminal degeneration. -/
theorem balancedPredictiveLaw_terminal (c : Counts) (δ : ℝ) :
    balancedPredictiveLaw 0 c δ = Measure.dirac δ := by
  rw [balancedPredictiveLaw, gaussianReal_zero_var]

end

end Draft0.PredictiveLaw
