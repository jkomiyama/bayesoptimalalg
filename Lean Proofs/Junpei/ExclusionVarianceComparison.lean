import Junpei.ExclusionBounds

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.ExclusionVarianceComparison

noncomputable section

open Draft0.Gaussian Draft0.BayesianModel Junpei.Exclusion

theorem hingeLoss_antitone_center (v : ℝ≥0) : Antitone (fun a : ℝ => hingeLoss a v) := by
  intro a b hab
  apply integral_mono (hinge_integrable b v) (hinge_integrable a v)
  intro x
  exact max_le_max (by linarith) le_rfl

theorem hingeLoss_mono_variance (a : ℝ) (ha : 0 ≤ a) {v w : ℝ≥0} (hvw : v ≤ w) :
    hingeLoss a v ≤ hingeLoss a w := by
  rcases eq_or_lt_of_le hvw with h | h
  · rw [h]
  · exact (hingeLoss_strictMono_variance a ha v w h).le

/-- Exact positive-part comparison used in the printed Lemma 11 proof:
the means are ordered and the posterior variances are at most one. -/
theorem pair_comparison (c : Counts) (m : Means) (i j : Arm)
    (upper lower : ℝ) (hgap : upper ≤ lower)
    (hi : m i ≤ upper) (hj : lower ≤ m j)
    (hci : 1 ≤ c i) (hcj : 1 ≤ c j) :
    pairSlack c m i j ≤ hingeLoss (lower - upper) 2 := by
  have hcent : lower - upper ≤ m j - m i := by linarith
  have hvar : (c i : ℝ≥0)⁻¹ + (c j : ℝ≥0)⁻¹ ≤ 2 := by
    exact_mod_cast Junpei.ExclusionBounds.variance_le_two_counts _ _ hci hcj
  exact (hingeLoss_antitone_center _ hcent).trans
    (hingeLoss_mono_variance _ (by linarith) hvar)

theorem independent_unit_positivePart (upper lower : ℝ) :
    (∫ x : ℝ, ∫ y : ℝ, max ((upper + x) - (lower + y)) 0
      ∂gaussianReal 0 1 ∂gaussianReal 0 1) = hingeLoss (lower - upper) 2 := by
  have h := positive_pair_double_integral upper lower 1 1
  norm_num at h
  exact h

/-- The displayed R^(3) ≤ 2 E[(x-y)+] inequality, with the independent
N(upper,1), N(lower,1) variables realized by translated standard Gaussians. -/
theorem lemma11_independent_comparison (c : Counts) (m : Means) (upper lower : ℝ)
    (hgap : upper ≤ lower) (h3 : m 2 ≤ upper) (h1 : lower ≤ m 0) (h2 : lower ≤ m 1)
    (hc : ∀ a, 1 ≤ c a) :
    exclusion c m 2 ≤ 2 * (∫ x : ℝ, ∫ y : ℝ,
      max ((upper + x) - (lower + y)) 0 ∂gaussianReal 0 1 ∂gaussianReal 0 1) := by
  rw [independent_unit_positivePart]
  have hleft := pair_comparison c m 2 0 upper lower hgap h3 h1 (hc 2) (hc 0)
  have hright := pair_comparison c m 2 1 upper lower hgap h3 h2 (hc 2) (hc 1)
  have hsum := exclusion_third_le_pairSlacks c m
  linarith

end
end Junpei.ExclusionVarianceComparison
