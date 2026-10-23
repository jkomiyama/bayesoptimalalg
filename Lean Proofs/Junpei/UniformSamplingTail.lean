import Junpei.TailRatio
import Junpei.RegretExponent

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

open Junpei.TailRatio

noncomputable section

theorem normalization_pos : 0 < normalization := by
  unfold normalization
  positivity

theorem tail_sandwich (x : ℝ) (hx : 2 ≤ x) :
    normalization / (2 * x) * Real.exp (-x^2 / 2) ≤ normalTail x ∧
      normalTail x ≤ normalization * Real.exp (-x^2 / 2) := by
  have hxp : 0 < x := by linarith
  have hcoef : 1 / (2 * x) ≤ 1 / x - 1 / x^3 := by
    have hsq : 2 ≤ x^2 := by nlinarith
    apply (div_le_iff₀ (by positivity : 0 < 2 * x)).mpr
    field_simp [ne_of_gt hxp]
    nlinarith
  constructor
  · have hm := mul_le_mul_of_nonneg_right hcoef (phi_nonnegative x)
    rw [phi_eq] at hm
    have he : normalization / (2 * x) * Real.exp (-x^2 / 2) =
        (1 / (2*x)) * (normalization * Real.exp (-x^2/2)) := by ring
    rw [he]
    exact hm.trans (by simpa only [phi_eq] using mills_lower x hxp)
  · have hm := mills_upper x hxp
    have hcoef' : 1 / x ≤ 1 := (div_le_one hxp).mpr (by linarith)
    exact hm.trans (by
      rw [← phi_eq]
      exact (mul_le_mul_of_nonneg_right hcoef' (phi_nonnegative x)).trans_eq (one_mul _))

theorem normalTail_pos_of_two_le (x : ℝ) (hx : 2 ≤ x) : 0 < normalTail x := by
  exact (mul_pos (div_pos normalization_pos (by linarith)) (Real.exp_pos _)).trans_le
    (tail_sandwich x hx).1

theorem normalTail_pos (x : ℝ) : 0 < normalTail x := by
  have hp := normalTail_pos_of_two_le (max x 2) (le_max_right _ _)
  have hm : normalTail (max x 2) ≤ normalTail x :=
    measureReal_mono (fun _ hy => (le_max_left x 2).trans_lt hy)
  exact hp.trans_le hm

theorem tail_log_bounds (x : ℝ) (hx : 2 ≤ x) :
    x^2 / 2 - Real.log normalization ≤ -Real.log (normalTail x) ∧
      -Real.log (normalTail x) ≤
        x^2 / 2 + Real.log 2 + Real.log x - Real.log normalization := by
  have hxp : 0 < x := by linarith
  have hp := normalTail_pos_of_two_le x hx
  have hs := tail_sandwich x hx
  have hl := Real.log_le_log
    (mul_pos (div_pos normalization_pos (by positivity)) (Real.exp_pos _)) hs.1
  have hu := Real.log_le_log hp hs.2
  rw [Real.log_mul (ne_of_gt (div_pos normalization_pos (by positivity)))
    (Real.exp_ne_zero _), Real.log_div normalization_pos.ne'
    (by positivity), Real.log_mul (by norm_num) hxp.ne', Real.log_exp] at hl
  rw [Real.log_mul normalization_pos.ne' (Real.exp_ne_zero _), Real.log_exp] at hu
  constructor <;> linarith

/-- Mills' bounds yield the exact quadratic logarithmic asymptotic of the
actual standard Gaussian tail. -/
theorem tail_log_sq_tendsto :
    Tendsto (fun x : ℝ => -Real.log (normalTail x) / x^2) atTop (𝓝 (1/2)) := by
  have hsq : Tendsto (fun x : ℝ => x^2) atTop atTop := by
    simpa only [pow_two] using Tendsto.atTop_mul_atTop₀ tendsto_id tendsto_id
  have hc (a : ℝ) : Tendsto (fun x : ℝ => a / x^2) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop hsq
  have hlog : Tendsto (fun x : ℝ => Real.log x / x) atTop (𝓝 0) := by
    simpa using Real.tendsto_pow_log_div_mul_add_atTop 1 0 1 one_ne_zero
  have hlogsq : Tendsto (fun x : ℝ => Real.log x / x^2) atTop (𝓝 0) := by
    convert hlog.mul tendsto_inv_atTop_zero using 1
    · funext x
      simp only [pow_two, div_eq_mul_inv]
      ring
    · simp
  have hlo : Tendsto (fun x : ℝ => (1/2 : ℝ) - Real.log normalization / x^2)
      atTop (𝓝 (1/2)) := by
    simpa using tendsto_const_nhds.sub (hc (Real.log normalization))
  have hhi : Tendsto (fun x : ℝ => (1/2 : ℝ) + Real.log 2 / x^2 +
      Real.log x / x^2 - Real.log normalization / x^2) atTop (𝓝 (1/2)) := by
    simpa using ((tendsto_const_nhds.add (hc (Real.log 2))).add hlogsq).sub
      (hc (Real.log normalization))
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo hhi
  · filter_upwards [eventually_ge_atTop (2 : ℝ)] with x hx
    have hb := div_le_div_of_nonneg_right (tail_log_bounds x hx).1 (sq_nonneg x)
    have hxn : x ≠ 0 := by linarith
    have he : x^2 / 2 / x^2 = (1/2 : ℝ) := by field_simp
    simpa only [sub_div, he] using hb
  · filter_upwards [eventually_ge_atTop (2 : ℝ)] with x hx
    have hb := div_le_div_of_nonneg_right (tail_log_bounds x hx).2 (sq_nonneg x)
    have hxn : x ≠ 0 := by linarith
    have he : x^2 / 2 / x^2 = (1/2 : ℝ) := by field_simp
    simpa only [add_div, sub_div, he] using hb

theorem gaussian_tail_rate (x : ℕ → ℝ) (q : ℝ)
    (hx : Tendsto x atTop atTop)
    (hq : Tendsto (fun T => (x T)^2 / (T : ℝ)) atTop (𝓝 q)) :
    Tendsto (fun T => -Real.log (normalTail (x T)) / (T : ℝ)) atTop (𝓝 (q / 2)) := by
  have h := (tail_log_sq_tendsto.comp hx).mul hq
  have h' : Tendsto (fun T => (-Real.log (normalTail (x T)) / (x T)^2) *
      ((x T)^2 / (T : ℝ))) atTop (𝓝 (q / 2)) := by
    have he : (1/2 : ℝ) * q = q / 2 := by ring
    simpa only [Function.comp_def, he] using h
  apply h'.congr'
  filter_upwards [hx.eventually (eventually_gt_atTop (0 : ℝ))] with T hT
  have he : (-Real.log (normalTail (x T)) / (x T)^2) * ((x T)^2 / (T : ℝ)) =
      -Real.log (normalTail (x T)) / (T : ℝ) := by
    field_simp [ne_of_gt hT]
  exact he

end
end Junpei.UniformSampling
