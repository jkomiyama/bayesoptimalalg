import Junpei.MarginalCounts

open MeasureTheory ProbabilityTheory Set
open scoped MeasureTheory NNReal ENNReal

namespace Junpei.Marginal

noncomputable section

open Draft0.Gaussian Draft0.TerminalBounds
open JunpeiComparison.Gaussian

theorem g2_selected_integral_lower (gap : ℝ) (vL vG vR : ℝ≥0) :
    newG2 gap vL vG vR ≤ ∫ l : ℝ in Icc (|gap| / 4) (|gap| / 3),
      hingeLoss |(|gap| - l)| vG ∂gaussianReal 0 vL := by
  have hevent (l : ℝ) (hl : l ∈ Icc (|gap| / 4) (|gap| / 3)) :
      (2 * |gap| / 3) *
        (gaussianReal 0 vG (Icc (9 * |gap| / 8) (4 * |gap| / 3))).toReal *
        (gaussianReal 0 vR (Ici (3 * |gap| / 4))).toReal ≤ hingeLoss |(|gap| - l)| vG := by
    have hg_event (g : ℝ) (hg : g ∈ Icc (9 * |gap| / 8) (4 * |gap| / 3)) :
        (2 * |gap| / 3) * (gaussianReal 0 vR (Ici (3 * |gap| / 4))).toReal ≤
          max (g - (|gap| - l)) 0 := by
      apply scalar_event_to_hinge
      intro r hr
      exact lemma16_repaired_event_bound |gap| l g r (abs_nonneg _) hl.1 hl.2
        hg.1 hg.2 hr
    rw [abs_of_nonneg (by linarith [hl.2, abs_nonneg gap] : 0 ≤ |gap| - l)]
    calc
      _ = ((2 * |gap| / 3) * (gaussianReal 0 vR (Ici (3 * |gap| / 4))).toReal) *
          (gaussianReal 0 vG (Icc (9 * |gap| / 8) (4 * |gap| / 3))).toReal := by ring
      _ ≤ ∫ g : ℝ in Icc (9 * |gap| / 8) (4 * |gap| / 3),
          max (g - (|gap| - l)) 0 ∂gaussianReal 0 vG :=
        setIntegral_ge_of_const_le measurableSet_Icc (measure_ne_top _ _) hg_event
          (hinge_integrable (|gap| - l) vG).restrict
      _ ≤ hingeLoss (|gap| - l) vG :=
        setIntegral_le_integral (hinge_integrable (|gap| - l) vG)
          (ae_of_all _ (fun _ => le_max_right _ _))
  calc
    newG2 gap vL vG vR = ((2 * |gap| / 3) *
        (gaussianReal 0 vG (Icc (9 * |gap| / 8) (4 * |gap| / 3))).toReal *
        (gaussianReal 0 vR (Ici (3 * |gap| / 4))).toReal) *
        (gaussianReal 0 vL (Icc (|gap| / 4) (|gap| / 3))).toReal := by
      unfold newG2
      ring
    _ ≤ _ := setIntegral_ge_of_const_le measurableSet_Icc (measure_ne_top _ _) hevent
      (absolute_shift_hinge_integrable |gap| vL vG).restrict

/-- Strictness of the revised Lemma 16 bound itself. The Gaussian contribution
outside the selected l interval is positive, even at tied means. -/
theorem lemma16_g2_strict (gap : ℝ) (vL vG vR : ℝ≥0)
    (hl : 0 < vL) (hg : 0 < vG) :
    newG2 gap vL vG vR < hingeLoss |gap| (vL + vG) - hingeLoss |gap| vL := by
  let S : Set ℝ := Icc (|gap| / 4) (|gap| / 3)
  let f : ℝ → ℝ := fun l => hingeLoss |(|gap| - l)| vG
  have hi : Integrable f (gaussianReal 0 vL) := absolute_shift_hinge_integrable |gap| vL vG
  have hcomp : 0 < ∫ l : ℝ in Sᶜ, f l ∂gaussianReal 0 vL := by
    apply (integral_pos_iff_support_of_nonneg (fun l => hingeLoss_nonnegative _ _) hi.restrict).mpr
    have hs : Function.support f = univ := by
      apply Set.eq_univ_of_forall
      intro l
      exact ne_of_gt (hingeLoss_positive _ _ hg)
    rw [hs, Measure.restrict_apply MeasurableSet.univ, univ_inter]
    have hp := gaussian_Icc_probability_pos 0 (|gap| + 1) (|gap| + 2) vL hl (by linarith)
    have hm := (ENNReal.toReal_pos_iff.mp hp).1
    apply lt_of_lt_of_le hm (measure_mono ?_)
    intro l hmem
    change l ∉ Icc (|gap| / 4) (|gap| / 3)
    intro hbad
    linarith only [hmem.1, hbad.2, abs_nonneg gap]
  have hsum := integral_add_compl (μ := gaussianReal 0 vL)
    (show MeasurableSet S from measurableSet_Icc) hi
  have hsel := g2_selected_integral_lower gap vL vG vR
  have hwhole : newG2 gap vL vG vR < ∫ l : ℝ, f l ∂gaussianReal 0 vL := by
    change newG2 gap vL vG vR ≤ ∫ l : ℝ in S, f l ∂gaussianReal 0 vL at hsel
    linarith only [hsum, hsel, hcomp]
  rw [hinge_increment_integral_shift]
  exact hwhole

/-- Revised Lemma 16, with strict inequality and no restriction on the gap. -/
theorem lemma16_g2_count_strict (gap : ℝ) (ni nj : ℕ)
    (hj : 1 ≤ nj) (hstrict : nj < ni) :
    g2Count gap ni nj <
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
    lemma16_g2_strict gap (updateVariance (ni : ℝ≥0))
      (updateVariance (nj : ℝ≥0) - updateVariance (ni : ℝ≥0))
      (posteriorVariance ni ((nj : ℝ≥0) + 1))
      (updateVariance_positive_nat ni (by omega)) (tsub_pos_iff_lt.mpr hv)

end

end Junpei.Marginal
