import Junpei.Exclusion
import Junpei.ExclusionTail
import Draft0.DynamicPolicy

open MeasureTheory ProbabilityTheory Filter Asymptotics
open scoped NNReal ENNReal

namespace Junpei.ExclusionBounds

open Draft0.Gaussian Draft0.Drift Draft0.Rates Draft0.DynamicPolicy
open Junpei.Exclusion Junpei.ExclusionTail

noncomputable section

theorem variance_le_two_counts (n m : ℕ) (hn : 1 ≤ n) (hm : 1 ≤ m) :
    ((n : ℝ≥0)⁻¹ + (m : ℝ≥0)⁻¹ : ℝ) ≤ 2 := by
  simp only [NNReal.coe_inv, NNReal.coe_natCast]
  have hnR : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hmR : (1 : ℝ) ≤ m := by exact_mod_cast hm
  linarith [inv_le_one_of_one_le₀ hnR, inv_le_one_of_one_le₀ hmR]

theorem cutoff_condition_implies_ten (CU : ℝ) (hCU : 0 < CU)
    (hcondition : 180 < (CU - 6) ^ 2 / 4) : 10 ≤ CU := by nlinarith

/-- Revised Lemma11 with its actual min-based exclusion loss and exact prefactor. -/
theorem lemma11 (T q t : ℕ) (CU : ℝ) (p : Draft0.Counts.MeanPath)
    (c : Draft0.Counts.Counts) (hT : 5 ≤ T) (hCU : 10 ≤ CU)
    (hc : Draft0.Counts.horizonConsistent t c) (hW : JunpeiComparison.newW T q CU p c) :
    exclusion (modelCounts c) (modelMeans p c) 2 ≤ paperG0 T CU := by
  have hl := pair_tail_newW T q 1 c.n1 CU p hT hCU hc.1 hW.2.2.2.2.2.1
    hW.1 hW.2.1 hW.2.2.2.1
    ((c.n3 : ℝ≥0)⁻¹ + (c.n1 : ℝ≥0)⁻¹)
    (variance_le_two_counts _ _ (by have := hc.2.2.1; omega) hc.1)
  have hr := pair_tail_newW T q 2 c.n2 CU p hT hCU hc.2.1 hW.2.2.2.2.2.2.1
    hW.1 hW.2.2.1 hW.2.2.2.2.1
    ((c.n3 : ℝ≥0)⁻¹ + (c.n2 : ℝ≥0)⁻¹)
    (variance_le_two_counts _ _ (by have := hc.2.2.1; omega) hc.2.1)
  have hs := exclusion_third_le_pairSlacks (modelCounts c) (modelMeans p c)
  have he0 : pairSlack (modelCounts c) (modelMeans p c) 2 0 =
      hingeLoss (p 1 c.n1 - p 3 1) ((c.n3 : ℝ≥0)⁻¹ + (c.n1 : ℝ≥0)⁻¹) := by
    simp [pairSlack, modelCounts, modelMeans, hc.2.2.1]
  have he1 : pairSlack (modelCounts c) (modelMeans p c) 2 1 =
      hingeLoss (p 2 c.n2 - p 3 1) ((c.n3 : ℝ≥0)⁻¹ + (c.n2 : ℝ≥0)⁻¹) := by
    simp [pairSlack, modelCounts, modelMeans, hc.2.2.1]
  rw [he0, he1] at hs
  calc
    _ ≤ _ := hs
    _ ≤ (1 / Real.sqrt Real.pi) * (T : ℝ) ^ (-((CU - 6) ^ 2 / 4)) +
        (1 / Real.sqrt Real.pi) * (T : ℝ) ^ (-((CU - 6) ^ 2 / 4)) := add_le_add hl hr
    _ = paperG0 T CU := by unfold paperG0; ring

theorem lemma11_from_equation11 (T q t : ℕ) (CU : ℝ) (p : Draft0.Counts.MeanPath)
    (c : Draft0.Counts.Counts) (hT : 5 ≤ T) (hCU : 0 < CU)
    (hcondition : 180 < (CU - 6) ^ 2 / 4)
    (hc : Draft0.Counts.horizonConsistent t c) (hW : JunpeiComparison.newW T q CU p c) :
    exclusion (modelCounts c) (modelMeans p c) 2 ≤ paperG0 T CU :=
  lemma11 T q t CU p c hT (cutoff_condition_implies_ten CU hCU hcondition) hc hW

theorem paperG0_nonnegative (T : ℕ) (CU : ℝ) : 0 ≤ paperG0 T CU := by
  unfold paperG0
  positivity

theorem paperG0_le_inverse181 (T : ℕ) (CU : ℝ) (hT : 5 ≤ T) (hCU : 33 ≤ CU) :
    paperG0 T CU ≤ (2 / Real.sqrt Real.pi) * invPower 181 T := by
  have hTpos : (0 : ℝ) < T := by exact_mod_cast (show 0 < T by omega)
  have hlog : 0 ≤ Real.log (T : ℝ) := by have := log_horizon_ge_one T hT; linarith
  have hrate : (181 : ℝ) ≤ (CU - 6) ^ 2 / 4 := by nlinarith
  have hscaled := mul_le_mul_of_nonneg_right hrate hlog
  have he : Real.exp (-(181 : ℝ) * Real.log (T : ℝ)) = invPower 181 T := by
    rw [neg_mul, Real.exp_neg]
    change (Real.exp ((181 : ℕ) * Real.log (T : ℝ)))⁻¹ = invPower 181 T
    rw [Real.exp_nat_mul, Real.exp_log hTpos]
    rfl
  unfold paperG0
  rw [Real.rpow_def_of_pos hTpos]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  rw [← he]
  apply Real.exp_le_exp.mpr
  nlinarith only [hscaled]

theorem paperG0_bigO181 (CU : ℝ) (hCU : 33 ≤ CU) :
    (fun T : ℕ => paperG0 T CU) =O[atTop] invPower 181 := by
  apply IsBigOWith.isBigO (c := 2 / Real.sqrt Real.pi)
  apply IsBigOWith.of_bound
  filter_upwards [eventually_ge_atTop (5 : ℕ)] with T hT
  rw [Real.norm_of_nonneg (paperG0_nonnegative T CU),
    Real.norm_of_nonneg (invPower_nonneg 181 T)]
  exact paperG0_le_inverse181 T CU hT hCU

theorem paperG0_littleO180 (CU : ℝ) (hCU : 33 ≤ CU) :
    (fun T : ℕ => paperG0 T CU) =o[atTop] invPower 180 :=
  error_littleO (by norm_num : 180 < 181) (paperG0_bigO181 CU hCU)

end

end Junpei.ExclusionBounds
