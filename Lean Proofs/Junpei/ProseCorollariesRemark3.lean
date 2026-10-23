import Junpei.Dynamic
import Junpei.TwoArmOptimal

set_option autoImplicit false
set_option warningAsError true

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.ProseCorollaries

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain
open Junpei.TwoArmOptimal

noncomputable section

/-- Remark 3: when neither initial count is at the cap, either first action
has the same posterior loss. This holds for every real mean pair, without W. -/
theorem remark3_equal_action_losses (q k n1 n2 : ℕ) (m1 m2 : ℝ)
    (h1 : 0 < n1) (h2 : 0 < n2) (hcap1 : n1 < q) (hcap2 : n2 < q)
    (hs : n1 + n2 + k = 2 * q - 1) :
    Junpei.FinitePosterior.actionLoss k (pairPrecisions n1 n2) (pairMeans m1 m2) 0 =
      Junpei.FinitePosterior.actionLoss k (pairPrecisions n1 n2) (pairMeans m1 m2) 1 := by
  let c := pairCounts n1 n2
  have hf1 := Junpei.Dynamic.forced_left_final_counts q k c (by omega)
    (show c.n1 ≤ q from hcap1.le) (show c.n2 ≤ q from hcap2.le) hs
  have hf2 := Junpei.Dynamic.forced_right_final_counts q k c (by omega)
    (show c.n1 ≤ q from hcap1.le) (show c.n2 ≤ q from hcap2.le) hs
  simp only [show ¬ c.n1 = q from Nat.ne_of_lt hcap1,
    show ¬ c.n2 = q from Nat.ne_of_lt hcap2, if_false] at hf1 hf2
  have hl := leftAction_variance_decomposition k c h1 h2
  have hr := rightAction_variance_decomposition k c h1 h2
  rw [Draft0.TwoArmBellman.variance, hf1.1, hf1.2] at hl
  rw [Draft0.TwoArmBellman.variance, hf2.1, hf2.2] at hr
  have hv : updateVariance c.n1 + cumulativeVariance k (leftStep c) =
      updateVariance c.n2 + cumulativeVariance k (rightStep c) :=
    add_right_cancel (hl.trans hr.symm)
  have ha : leftAction k m1 m2 c = rightAction k m1 m2 c := by
    rw [leftAction_eq, rightAction_eq, hv]
  have hp0 : 0 < pairPrecisions n1 n2 0 := by simpa [pairPrecisions] using h1
  have hp1 : 0 < pairPrecisions n1 n2 1 := by simpa [pairPrecisions] using h2
  rw [Junpei.FinitePosterior.actionLoss_eq _ _ _ _ hp0,
    Junpei.FinitePosterior.actionLoss_eq _ _ _ _ hp1,
    action_left_eq, action_right_eq]
  exact congrArg (fun x => Junpei.FinitePosterior.potential
    (pairPrecisions n1 n2) (pairMeans m1 m2) - x) ha

/-- The actual count-balancing continuation has the literal terminal pair
(q,q) with the remaining budget of Remark 3. -/
theorem remark3_balanced_terminal_counts (q S n1 n2 : ℕ)
    (hcap1 : n1 ≤ q) (hcap2 : n2 ≤ q) (hs : n1 + n2 + S = 2 * q) :
    (balancedRun S (pairCounts n1 n2)).n1 = q ∧
      (balancedRun S (pairCounts n1 n2)).n2 = q :=
  balancedRun_final_counts q S (pairCounts n1 n2) hcap1 hcap2 hs

theorem hinge_difference_integral_pos (a b v : ℝ≥0) (d : ℝ) (hab : a < b) :
    0 < ∫ x : ℝ, hingeLoss |x| b - hingeLoss |x| a ∂gaussianReal d v := by
  have hp (x : ℝ) : 0 < hingeLoss |x| b - hingeLoss |x| a :=
    sub_pos.mpr (hingeLoss_strictMono_variance |x| (abs_nonneg _) a b hab)
  apply (integral_pos_iff_support_of_nonneg (fun x => (hp x).le)
    ((absolute_hinge_gaussian_integrable d v b).sub
      (absolute_hinge_gaussian_integrable d v a))).mpr
  have hsupport : Function.support (fun x : ℝ => hingeLoss |x| b - hingeLoss |x| a) =
      Set.univ := Set.eq_univ_of_forall (fun x => (hp x).ne')
  rw [hsupport]
  simp

/-- Remark 3's saturated-left inequality, at every mean pair. No event or
asymptotic probability hypothesis is needed. -/
theorem remark3_saturated_left_strict (q k n2 : ℕ) (m1 m2 : ℝ)
    (h2 : 0 < n2) (hs : q + n2 + k = 2 * q - 1) :
    Junpei.FinitePosterior.actionLoss k (pairPrecisions q n2) (pairMeans m1 m2) 1 <
      Junpei.FinitePosterior.actionLoss k (pairPrecisions q n2) (pairMeans m1 m2) 0 := by
  have hq : 2 ≤ q := by omega
  let c := pairCounts q n2
  have hc1 : 0 < c.n1 := by change 0 < q; omega
  have hc2 : 0 < c.n2 := h2
  have hct : c.n2 < c.n1 := by change n2 < q; omega
  have hgap := saturated_left_action_gap_integral k m1 m2 c hc1 hc2
    (show c.n2 + k ≤ c.n1 by change n2 + k ≤ q; omega)
  have hf := saturated_left_prefix_counts q k c (by omega) rfl hs
  rw [balancedInnovation_preterminal q k c (by omega) (by rfl)
    (show c.n2 ≤ q by change n2 ≤ q; omega) hs, hf.1,
    Draft0.PredictiveLaw.balancedPredictiveLaw_gaussian] at hgap
  have hv := updateVariance_strictAntitone ((q - 1 : ℕ) : ℝ≥0) (q : ℝ≥0)
    (by exact_mod_cast (show 0 < q - 1 by omega))
    (by exact_mod_cast (show q - 1 < q by omega))
  have hp := hinge_difference_integral_pos _ _
    (Draft0.PredictiveLaw.accumulatedVariance k c) (m1 - m2) hv
  rw [← hgap] at hp
  have hright := smaller_count_right_optimal k m1 m2 c hc1 hc2 hct
  have h0 : 0 < pairPrecisions q n2 0 := by simpa [pairPrecisions] using hc1
  have h1 : 0 < pairPrecisions q n2 1 := by simpa [pairPrecisions] using h2
  rw [Junpei.FinitePosterior.actionLoss_eq _ _ _ _ h1,
    Junpei.FinitePosterior.actionLoss_eq _ _ _ _ h0,
    action_right_eq, action_left_eq]
  change _ - rightAction k m1 m2 c < _ - leftAction k m1 m2 c
  rw [← hright]
  linarith

/-- The symmetric saturated-right assertion of Remark 3. -/
theorem remark3_saturated_right_strict (q k n1 : ℕ) (m1 m2 : ℝ)
    (h1 : 0 < n1) (hs : n1 + q + k = 2 * q - 1) :
    Junpei.FinitePosterior.actionLoss k (pairPrecisions n1 q) (pairMeans m1 m2) 0 <
      Junpei.FinitePosterior.actionLoss k (pairPrecisions n1 q) (pairMeans m1 m2) 1 := by
  have hq : 2 ≤ q := by omega
  let c := pairCounts n1 q
  have hc1 : 0 < c.n1 := h1
  have hc2 : 0 < c.n2 := by change 0 < q; omega
  have hct : c.n1 ≤ c.n2 := by change n1 ≤ q; omega
  have hgap := saturated_right_action_gap_integral k m1 m2 c hc1 hc2
    (show c.n1 + k ≤ c.n2 by change n1 + k ≤ q; omega)
  have hf := saturated_right_prefix_counts q k c (by omega) rfl hs
  rw [balancedInnovation_preterminal q k c (by omega)
    (show c.n1 ≤ q by change n1 ≤ q; omega) (by rfl) hs, hf.2,
    Draft0.PredictiveLaw.balancedPredictiveLaw_gaussian] at hgap
  have hv := updateVariance_strictAntitone ((q - 1 : ℕ) : ℝ≥0) (q : ℝ≥0)
    (by exact_mod_cast (show 0 < q - 1 by omega))
    (by exact_mod_cast (show q - 1 < q by omega))
  have hp := hinge_difference_integral_pos _ _
    (Draft0.PredictiveLaw.accumulatedVariance k c) (m1 - m2) hv
  rw [← hgap] at hp
  have hleft := smaller_count_left_optimal k m1 m2 c hc1 hc2 hct
  have h0 : 0 < pairPrecisions n1 q 0 := by simpa [pairPrecisions] using h1
  have h1' : 0 < pairPrecisions n1 q 1 := by simpa [pairPrecisions] using hc2
  rw [Junpei.FinitePosterior.actionLoss_eq _ _ _ _ h0,
    Junpei.FinitePosterior.actionLoss_eq _ _ _ _ h1',
    action_left_eq, action_right_eq]
  change _ - leftAction k m1 m2 c < _ - rightAction k m1 m2 c
  rw [← hleft]
  linarith

end
end Junpei.ProseCorollaries
