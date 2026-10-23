import Draft0.GaussianTail
import Junpei.CountComparison
import Mathlib.Analysis.Real.Pi.Bounds

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.ExclusionTail

set_option maxHeartbeats 1000000

open Draft0.Gaussian Draft0.Counts Draft0.Drift Draft0.Rates

noncomputable section

theorem mean_bound6_newZ (T q i n : ℕ) (p : MeanPath) (hT : 5 ≤ T)
    (hn : 1 ≤ n) (hnq : n ≤ q) (hY : eventY T q i p)
    (hZ : JunpeiComparison.newZ T q i p) :
    |p i n| ≤ 6 * Real.sqrt (Real.log (T : ℝ)) := by
  have hlog := log_horizon_ge_one T hT
  have hroot : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have hs := reciprocalSqSum_bound n q hn hnq
  have hnR : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hinv : 1 / (n : ℝ) ≤ 1 := (div_le_one (by positivity)).mpr hnR
  have hsbound : reciprocalSqSum n q ≤ 2 := by
    have hqinv : 0 ≤ 1 / (q : ℝ) := by positivity
    linarith
  have hsroot : Real.sqrt (reciprocalSqSum n q) ≤ 3 / 2 := by
    apply (Real.sqrt_le_left (by norm_num : (0 : ℝ) ≤ 3 / 2)).mpr
    linarith
  have hz := hZ n hn hnq
  have htri : |p i n| ≤ |p i n - p i q| + |p i q| := by
    simpa only [sub_add_cancel] using abs_add_le (p i n - p i q) (p i q)
  have hsmall : 1 / (T : ℝ) ^ 2 ≤ 1 := by
    have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT
    apply (div_le_one (by positivity)).mpr
    nlinarith
  have hscaled := mul_le_mul_of_nonneg_left hsroot
    (show 0 ≤ 3 * Real.sqrt (Real.log (T : ℝ)) by positivity)
  unfold eventY at hY
  nlinarith

/-- The paper's exact prefactor is obtained by retaining the 1/t MGF factor. -/
theorem hingeLoss_le_paper_prefactor (a b : ℝ) (v : ℝ≥0)
    (hb : 4 ≤ b) (ha : b ≤ a) (hv : (v : ℝ) ≤ 2) :
    hingeLoss a v ≤ (1 / Real.sqrt Real.pi) * Real.exp (-b ^ 2 / 4) := by
  have hbpos : 0 < b := by linarith
  have hpi : Real.sqrt Real.pi ≤ 2 := by
    apply (Real.sqrt_le_left (by norm_num : (0 : ℝ) ≤ 2)).mpr
    linarith [Real.pi_lt_four]
  have hroot : 0 < Real.sqrt Real.pi := Real.sqrt_pos.mpr Real.pi_pos
  have hexp : -(b / 2) * a + (v : ℝ) * (b / 2) ^ 2 / 2 ≤ -b ^ 2 / 4 := by
    have hm := mul_le_mul_of_nonneg_left ha (half_pos hbpos).le
    nlinarith [sq_nonneg b]
  have hfactor : 1 / (b / 2) ≤ 1 / Real.sqrt Real.pi := by
    apply one_div_le_one_div_of_le hroot
    linarith
  calc
    hingeLoss a v ≤ Real.exp (-(b / 2) * a + (v : ℝ) * (b / 2) ^ 2 / 2) / (b / 2) :=
      Draft0.GaussianTail.hingeLoss_le_mgf a v (b / 2) (half_pos hbpos)
    _ ≤ Real.exp (-b ^ 2 / 4) / (b / 2) :=
      div_le_div_of_nonneg_right (Real.exp_le_exp.mpr hexp) (half_pos hbpos).le
    _ ≤ (1 / Real.sqrt Real.pi) * Real.exp (-b ^ 2 / 4) := by
      calc
        _ = (1 / (b / 2)) * Real.exp (-b ^ 2 / 4) := by ring
        _ ≤ _ := mul_le_mul_of_nonneg_right hfactor (Real.exp_pos (-b ^ 2 / 4)).le

def paperG0 (T : ℕ) (CU : ℝ) : ℝ :=
  (2 / Real.sqrt Real.pi) * (T : ℝ) ^ (-((CU - 6) ^ 2 / 4))

theorem pair_tail_newW (T q i n : ℕ) (CU : ℝ) (p : MeanPath) (hT : 5 ≤ T)
    (hCU : 10 ≤ CU) (hn : 1 ≤ n) (hnq : n ≤ q)
    (hX : eventX T CU p) (hY : eventY T q i p)
    (hZ : JunpeiComparison.newZ T q i p) (v : ℝ≥0) (hv : (v : ℝ) ≤ 2) :
    hingeLoss (p i n - p 3 1) v ≤
      (1 / Real.sqrt Real.pi) * (T : ℝ) ^ (-((CU - 6) ^ 2 / 4)) := by
  have hlog := log_horizon_ge_one T hT
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith
  have hsqrt : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have hmean := mean_bound6_newZ T q i n p hT hn hnq hY hZ
  have hlo := (abs_le.mp hmean).1
  have hgap : (CU - 6) * Real.sqrt (Real.log (T : ℝ)) ≤ p i n - p 3 1 := by
    unfold eventX at hX
    nlinarith
  have hb : 4 ≤ (CU - 6) * Real.sqrt (Real.log (T : ℝ)) := by nlinarith
  have h := hingeLoss_le_paper_prefactor _ _ v hb hgap hv
  have he : Real.exp (-((CU - 6) * Real.sqrt (Real.log (T : ℝ))) ^ 2 / 4) =
      (T : ℝ) ^ (-((CU - 6) ^ 2 / 4)) := by
    rw [Real.rpow_def_of_pos (by exact_mod_cast (show 0 < T by omega))]
    congr 1
    rw [mul_pow, Real.sq_sqrt hlog0]
    ring
  simpa only [he] using h

end

end Junpei.ExclusionTail
