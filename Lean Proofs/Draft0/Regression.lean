import Draft0.Gaussian
import Draft0.Counts
import Draft0.MainBridge

/-!
Previously failing parameter choices, checked against the corrected statements.
Every declaration is a Lean proof. These cases supplement the general theorems.
-/

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Draft0.Regression

theorem corrected_lemma13_old_witness :
    Draft0.Gaussian.posteriorLoss 0 1 =
      (Real.sqrt (2 * Real.pi))⁻¹ * Draft0.Gaussian.unnormalizedLoss 0 1 := by
  simpa using Draft0.Gaussian.lemma13_normalized 0 1 (by norm_num)

theorem corrected_lemma14_zero_gap_positive :
    0 < Draft0.Gaussian.oneStepImprovement 0 (1 / 2) (3 / 2) := by
  exact Draft0.Gaussian.oneStepImprovement_positive 0 (1 / 2) (3 / 2) (by norm_num)

theorem corrected_lemma14_zero_gap_bound_positive :
    0 < Draft0.Gaussian.correctedImprovementLowerBound 0 (1 / 25) (1 / 2) := by
  exact Draft0.Gaussian.correctedImprovementLowerBound_positive 0 (1 / 25)
    (1 / 2) (by norm_num) (by norm_num)

theorem corrected_inverse_old_witness :
    (2 : ℝ)⁻¹ = 2 / (((2 * 2 + 1 : ℕ) : ℝ) - 1) := by
  exact Draft0.Counts.appendixB_exact_inverse 2 (by norm_num)

theorem corrected_counts_old_preterminal_witness :
    Draft0.Counts.balancedRun (7 - 6) ⟨1, 3, 1⟩ = ⟨2, 3, 1⟩ := by
  norm_num [Draft0.Counts.balancedRun, Draft0.Counts.balancedStep]

theorem corrected_counts_old_terminal_witness :
    Draft0.Counts.balancedRun (5 - 5) ⟨1, 2, 1⟩ = ⟨1, 2, 1⟩ := by
  rfl

theorem corrected_counts_final_sample :
    Draft0.Counts.balancedRun 1 ⟨1, 2, 1⟩ = ⟨2, 2, 1⟩ := by
  norm_num [Draft0.Counts.balancedRun, Draft0.Counts.balancedStep]

theorem corrected_lemma17_terminal_mass {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] :
    (μ {_ω : Ω | Draft0.Counts.gapBand 5 0}).toReal = 0 := by
  have he : {_ω : Ω | Draft0.Counts.gapBand 5 0} = ∅ := by
    ext _ω
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    exact Draft0.Counts.lemma17_terminal_band_zero 5 (by norm_num)
  rw [he]
  simp

theorem corrected_regret_scale_old_witness :
    (1 / 2 : ℝ) * (1 / 2 : ℝ) = (1 / 4 : ℝ) := by
  norm_num

end Draft0.Regression
