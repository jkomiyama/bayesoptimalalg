import Junpei.GaussianComparison
import Draft0.TwoArmGain
import Draft0.TerminalBounds

open MeasureTheory ProbabilityTheory Set Filter Asymptotics
open scoped MeasureTheory NNReal ENNReal

namespace Junpei.Marginal

noncomputable section

open Draft0.Gaussian Draft0.TwoArmBellman Draft0.Rates
open JunpeiComparison.Gaussian

/-- Revised Lemma 14: terminal Gaussian simple regret with its density normalization. -/
theorem lemma14_normalized (m1 m2 : ℝ) (v1 v2 : ℝ≥0) (hv : 0 < v1 + v2) :
    twoArmPosteriorLoss m1 m2 v1 v2 =
      (Real.sqrt (2 * Real.pi * (v1 + v2)))⁻¹ *
        unnormalizedLoss (m1 - m2) (v1 + v2) :=
  twoArmPosteriorLoss_normalized m1 m2 v1 v2 hv

theorem pointwiseImprovement_eq (d l r : ℝ) :
    pointwiseImprovement d l r =
      max (r - (d - l)) 0 - max (r - |d - l|) 0 := by
  unfold pointwiseImprovement
  congr 2
  ring

theorem scalar_integrable (d l : ℝ) (v : ℝ≥0) :
    Integrable (pointwiseImprovement d l) (gaussianReal 0 v) := by
  have he : pointwiseImprovement d l =
      fun r : ℝ => max (r - (d - l)) 0 - max (r - |d - l|) 0 := by
    funext r
    exact pointwiseImprovement_eq d l r
  rw [he]
  exact (hinge_integrable (d - l) v).sub (hinge_integrable |d - l| v)

theorem scalar_integral (d l : ℝ) (v : ℝ≥0) :
    (∫ r : ℝ, pointwiseImprovement d l r ∂gaussianReal 0 v) = max (l - d) 0 := by
  simp_rw [pointwiseImprovement_eq]
  rw [integral_sub (hinge_integrable (d - l) v) (hinge_integrable |d - l| v)]
  change hingeLoss (d - l) v - hingeLoss |d - l| v = _
  rw [hingeLoss_sub_abs]
  congr 1
  ring

theorem scalar_event_to_hinge (d l u a : ℝ) (v : ℝ≥0)
    (h : ∀ r ∈ Ici a, u ≤ pointwiseImprovement d l r) :
    u * (gaussianReal 0 v (Ici a)).toReal ≤ max (l - d) 0 := by
  calc
    _ ≤ ∫ r : ℝ in Ici a, pointwiseImprovement d l r ∂gaussianReal 0 v :=
      setIntegral_ge_of_const_le measurableSet_Ici (measure_ne_top _ _) h
        (scalar_integrable d l v).restrict
    _ ≤ ∫ r : ℝ, pointwiseImprovement d l r ∂gaussianReal 0 v :=
      setIntegral_le_integral (scalar_integrable d l v)
        (ae_of_all _ (lemma15_pointwise_nonnegative d l))
    _ = _ := scalar_integral d l v

/-- The exact probability-product lower bound printed in revised Lemma 15. -/
theorem lemma15_g1_lower (gap : ℝ) (vL vR : ℝ≥0) :
    newG1 gap vL vR ≤ oneStepImprovement gap vL vR := by
  rw [oneStepImprovement_eq_hinge]
  have hevent (l : ℝ) (hl : l ∈ Icc (3 * |gap| / 2) (2 * |gap|)) :
      |gap| * (gaussianReal 0 vR (Ici |gap|)).toReal ≤ max (l - |gap|) 0 := by
    apply scalar_event_to_hinge
    intro r hr
    exact lemma15_repaired_event_bound |gap| l r (abs_nonneg _) hl.1 hl.2 hr
  calc
    newG1 gap vL vR = (|gap| * (gaussianReal 0 vR (Ici |gap|)).toReal) *
        (gaussianReal 0 vL (Icc (3 * |gap| / 2) (2 * |gap|))).toReal := by
      unfold newG1
      ring
    _ ≤ ∫ l : ℝ in Icc (3 * |gap| / 2) (2 * |gap|), max (l - |gap|) 0
        ∂gaussianReal 0 vL :=
      setIntegral_ge_of_const_le measurableSet_Icc (measure_ne_top _ _) hevent
        (hinge_integrable |gap| vL).restrict
    _ ≤ hingeLoss |gap| vL :=
      setIntegral_le_integral (hinge_integrable |gap| vL)
        (ae_of_all _ (fun _ => le_max_right _ _))

theorem absolute_shift_hinge_integrable (d : ℝ) (a b : ℝ≥0) :
    Integrable (fun l : ℝ => hingeLoss |d - l| b) (gaussianReal 0 a) := by
  have he (l : ℝ) : hingeLoss |d - l| b =
      hingeLoss (d - l) b - max (l - d) 0 := by
    have h := hingeLoss_sub_abs (d - l) b
    have hn : -(d - l) = l - d := by ring
    rw [hn] at h
    linarith only [h]
  simp_rw [he]
  exact (hinge_smoothing_integrable d a b).sub (hinge_integrable d a)

theorem hinge_increment_integral_shift (d : ℝ) (a b : ℝ≥0) :
    hingeLoss d (a + b) - hingeLoss d a =
      ∫ l : ℝ, hingeLoss |d - l| b ∂gaussianReal 0 a := by
  have he (l : ℝ) : hingeLoss |d - l| b =
      hingeLoss (d - l) b - max (l - d) 0 := by
    have h := hingeLoss_sub_abs (d - l) b
    have hn : -(d - l) = l - d := by ring
    rw [hn] at h
    linarith only [h]
  simp_rw [he]
  rw [integral_sub (hinge_smoothing_integrable d a b) (hinge_integrable d a),
    hinge_smoothing]
  rfl

/-- The exact probability-product lower bound printed in revised Lemma 16,
before substituting the count-dependent variance difference. -/
theorem lemma16_g2_lower (gap : ℝ) (vL vG vR : ℝ≥0) :
    newG2 gap vL vG vR ≤
      hingeLoss |gap| (vL + vG) - hingeLoss |gap| vL := by
  rw [hinge_increment_integral_shift]
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
    _ ≤ ∫ l : ℝ in Icc (|gap| / 4) (|gap| / 3), hingeLoss |(|gap| - l)| vG
        ∂gaussianReal 0 vL :=
      setIntegral_ge_of_const_le measurableSet_Icc (measure_ne_top _ _) hevent
        (absolute_shift_hinge_integrable |gap| vL vG).restrict
    _ ≤ ∫ l : ℝ, hingeLoss |(|gap| - l)| vG ∂gaussianReal 0 vL :=
      setIntegral_le_integral (absolute_shift_hinge_integrable |gap| vL vG)
        (ae_of_all _ (fun _ => hingeLoss_nonnegative _ _))

end

end Junpei.Marginal
