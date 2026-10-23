import Junpei.MarginalRates

open MeasureTheory ProbabilityTheory Set Filter Asymptotics
open scoped MeasureTheory NNReal ENNReal

namespace Junpei.Marginal

noncomputable section

open Draft0.Gaussian Draft0.Rates Draft0.TerminalBounds
open JunpeiComparison.Gaussian

def g1Count (gap : ℝ) (ni nj : ℕ) : ℝ :=
  newG1 gap (updateVariance (ni : ℝ≥0)) (posteriorVariance ((ni : ℝ≥0) + 1) nj)

def g2Count (gap : ℝ) (ni nj : ℕ) : ℝ :=
  newG2 gap (updateVariance (ni : ℝ≥0))
    (updateVariance (nj : ℝ≥0) - updateVariance (ni : ℝ≥0))
    (posteriorVariance ni ((nj : ℝ≥0) + 1))

theorem posteriorVariance_count_bounds (ni nj T : ℕ) (hT : 1 ≤ T)
    (hi : 1 ≤ ni) (hj : 1 ≤ nj) (hiT : ni ≤ T) (_hjT : nj ≤ T) :
    invPower 2 T ≤ (posteriorVariance (ni : ℝ≥0) nj : ℝ) ∧
      (posteriorVariance (ni : ℝ≥0) nj : ℝ) ≤ 2 := by
  have hir : 1 ≤ (ni : ℝ) := by exact_mod_cast hi
  have hjr : 1 ≤ (nj : ℝ) := by exact_mod_cast hj
  have hTr : 1 ≤ (T : ℝ) := by exact_mod_cast hT
  have hiTr : (ni : ℝ) ≤ T := by exact_mod_cast hiT
  have hilu : (ni : ℝ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hir
  have hjlu : (nj : ℝ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hjr
  have hlow : invPower 2 T ≤ (ni : ℝ)⁻¹ := by
    unfold invPower
    apply (inv_le_inv₀ (by positivity) (by linarith only [hir])).mpr
    nlinarith only [hiTr, hTr]
  simp only [posteriorVariance, NNReal.coe_add, NNReal.coe_inv, NNReal.coe_natCast]
  constructor
  · exact hlow.trans (le_add_of_nonneg_right (by positivity))
  · linarith only [hilu, hjlu]

theorem updateVariance_positive_nat (n : ℕ) (hn : 1 ≤ n) :
    0 < updateVariance (n : ℝ≥0) := by
  have hp : 0 < (n : ℝ≥0) := by exact_mod_cast (show 0 < n by omega)
  unfold updateVariance
  positivity

theorem lemma15_g1_count_lower (gap : ℝ) (ni nj : ℕ) :
    g1Count gap ni nj ≤ oneStepImprovement gap (updateVariance (ni : ℝ≥0))
      (posteriorVariance ((ni : ℝ≥0) + 1) nj) :=
  lemma15_g1_lower gap _ _

/-- Revised Lemma 15's named g1, with the stronger T^-4 rate before weakening. -/
theorem lemma15_g1_uniform4 (T ni nj : ℕ) (gap : ℝ)
    (hT : 5 ≤ T) (hi : 1 ≤ ni) (hj : 1 ≤ nj) (hiT : ni < T) (hjT : nj < T)
    (hgapL : invPower 2 T ≤ |gap|) (hgapU : |gap| ≤ 2 * invPower 2 T) :
    g1Constant * invPower 4 T ≤ g1Count gap ni nj := by
  have hv := updateVariance_count_bounds ni T hi hiT
  have hr := posteriorVariance_count_bounds (ni + 1) nj T (by omega) (by omega) hj
    (by omega) (by omega)
  unfold g1Count
  apply g1_uniform4_variance T gap _ _ hT hgapL hgapU (updateVariance_positive_nat ni hi)
    (by linarith only [hv.2])
  · exact (invPower_antitone (by norm_num : 2 ≤ 3) (by omega : 1 ≤ T)).trans hv.1
  · simpa only [Nat.cast_add, Nat.cast_one] using hr.1

/-- Revised Lemma 15's exact uniform Omega(T^-6) lower bound. -/
theorem lemma15_g1_uniform6 (T ni nj : ℕ) (gap : ℝ)
    (hT : 5 ≤ T) (hi : 1 ≤ ni) (hj : 1 ≤ nj) (hiT : ni < T) (hjT : nj < T)
    (hgapL : invPower 2 T ≤ |gap|) (hgapU : |gap| ≤ 2 * invPower 2 T) :
    g1Constant * invPower 6 T ≤ g1Count gap ni nj := by
  exact (mul_le_mul_of_nonneg_left
    (invPower_antitone (by norm_num : 4 ≤ 6) (by omega : 1 ≤ T)) g1Constant_pos.le).trans
    (lemma15_g1_uniform4 T ni nj gap hT hi hj hiT hjT hgapL hgapU)

theorem varianceDifference_count_bounds (T ni nj : ℕ)
    (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T) :
    0 < updateVariance (nj : ℝ≥0) - updateVariance (ni : ℝ≥0) ∧
      ((updateVariance (nj : ℝ≥0) - updateVariance (ni : ℝ≥0) : ℝ≥0) : ℝ) ≤ 2 ∧
      invPower 3 T ≤ ((updateVariance (nj : ℝ≥0) - updateVariance (ni : ℝ≥0) : ℝ≥0) : ℝ) := by
  have hjp : 0 < (nj : ℝ≥0) := by exact_mod_cast (show 0 < nj by omega)
  have hji : (nj : ℝ≥0) < ni := by exact_mod_cast hstrict
  have hv := updateVariance_strictAntitone (nj : ℝ≥0) ni hjp hji
  have hjb := updateVariance_count_bounds nj T hj (by omega)
  have hl := updateVariance_difference_lower ni nj T hj hstrict hiT
  constructor
  · exact tsub_pos_iff_lt.mpr hv
  rw [NNReal.coe_sub hv.le]
  constructor
  · have hip : 0 ≤ (updateVariance (ni : ℝ≥0) : ℝ) := (updateVariance _).property
    linarith only [hjb.2, hip]
  · exact hl

/-- The exact revised g2 bound on the actual terminal action-loss difference. -/
theorem lemma16_g2_count_lower (gap : ℝ) (ni nj : ℕ)
    (hj : 1 ≤ nj) (hstrict : nj < ni) :
    g2Count gap ni nj ≤
      expectedLossAfterDraw gap (updateVariance (ni : ℝ≥0))
        (posteriorVariance ((ni : ℝ≥0) + 1) nj) -
      expectedLossAfterDraw gap (updateVariance (nj : ℝ≥0))
        (posteriorVariance ((nj : ℝ≥0) + 1) ni) := by
  have hip : 0 < (ni : ℝ≥0) := by exact_mod_cast (show 0 < ni by omega)
  have hjp : 0 < (nj : ℝ≥0) := by exact_mod_cast (show 0 < nj by omega)
  have hji : (nj : ℝ≥0) < ni := by exact_mod_cast hstrict
  have hv := updateVariance_strictAntitone (nj : ℝ≥0) ni hjp hji
  rw [terminal_actionLoss_difference_eq gap _ _ hjp hip]
  unfold g2Count
  simpa only [add_tsub_cancel_of_le hv.le] using
    lemma16_g2_lower gap (updateVariance (ni : ℝ≥0))
      (updateVariance (nj : ℝ≥0) - updateVariance (ni : ℝ≥0))
      (posteriorVariance ni ((nj : ℝ≥0) + 1))

theorem lemma16_g2_uniform6 (T ni nj : ℕ) (gap : ℝ)
    (hT : 5 ≤ T) (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T)
    (hgapL : invPower 2 T ≤ |gap|) (hgapU : |gap| ≤ 2 * invPower 2 T) :
    g2Constant * invPower 6 T ≤ g2Count gap ni nj := by
  have hi : 1 ≤ ni := by omega
  have hv := updateVariance_count_bounds ni T hi hiT
  have hg := varianceDifference_count_bounds T ni nj hj hstrict hiT
  have hr := posteriorVariance_count_bounds ni (nj + 1) T (by omega) hi (by omega)
    (by omega) (by omega)
  unfold g2Count
  apply g2_uniform6_variance T gap _ _ _ hT hgapL hgapU (updateVariance_positive_nat ni hi)
    (by linarith only [hv.2])
  · exact (invPower_antitone (by norm_num : 2 ≤ 3) (by omega : 1 ≤ T)).trans hv.1
  · exact hg.1
  · exact hg.2.1
  · exact hg.2.2
  · simpa only [Nat.cast_add, Nat.cast_one] using hr.1

/-- Revised Lemma 16's exact uniform Omega(T^-8) lower bound on its named g2. -/
theorem lemma16_g2_uniform8 (T ni nj : ℕ) (gap : ℝ)
    (hT : 5 ≤ T) (hj : 1 ≤ nj) (hstrict : nj < ni) (hiT : ni < T)
    (hgapL : invPower 2 T ≤ |gap|) (hgapU : |gap| ≤ 2 * invPower 2 T) :
    g2Constant * invPower 8 T ≤ g2Count gap ni nj := by
  exact (mul_le_mul_of_nonneg_left
    (invPower_antitone (by norm_num : 6 ≤ 8) (by omega : 1 ≤ T)) g2Constant_pos.le).trans
    (lemma16_g2_uniform6 T ni nj gap hT hj hstrict hiT hgapL hgapU)

end

end Junpei.Marginal
