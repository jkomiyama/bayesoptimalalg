import Junpei.ProseCorollariesFootnote14

open Filter Asymptotics
open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Footnote14Scale

noncomputable section

open Junpei.ProseCorollaries.Footnote14

theorem standardDeviation_eq_paper_scale (T : ℕ) :
    standardDeviation T = (T : ℝ) ^ (-(3 : ℝ) / 2) := by
  unfold standardDeviation
  rw [Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_mul (Nat.cast_nonneg T),
    ← Real.rpow_neg (Nat.cast_nonneg T)]
  congr 1
  norm_num

theorem logarithmicGap_not_bigO_paper_scale :
    ¬ (logarithmicGap =O[atTop] (fun T : ℕ => (T : ℝ) ^ (-(3 : ℝ) / 2))) := by
  simpa only [← standardDeviation_eq_paper_scale] using logarithmicGap_not_bigO_standardDeviation

def PolynomialTailLowerBound (gap : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, 0 < c ∧ ∃ p : ℕ, ∀ T : ℕ, 5 ≤ T →
    c * Draft0.Rates.invPower p T ≤
      (gaussianReal 0 (((T : ℝ≥0) ^ 3)⁻¹)).real (Ici (gap T))

/-- Direct negation of the footnote's asserted necessary condition, even
when every threshold is nonnegative. -/
theorem polynomial_tail_does_not_require_paper_scale :
    ¬ (∀ gap : ℕ → ℝ, (∀ T, 0 ≤ gap T) → PolynomialTailLowerBound gap →
      gap =O[atTop] (fun T : ℕ => (T : ℝ) ^ (-(3 : ℝ) / 2))) := by
  intro h
  apply logarithmicGap_not_bigO_paper_scale
  apply h logarithmicGap
  · intro T
    unfold logarithmicGap standardDeviation
    positivity
  · exact ⟨Draft0.BadEvent.underestimationConstant,
      Draft0.BadEvent.underestimationConstant_positive, 4, logarithmicGap_polynomial_tail⟩

end
end Junpei.Footnote14Scale
