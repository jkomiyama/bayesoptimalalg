import Draft0.Gaussian
import Draft0.Rates
import Draft0.Drift

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Draft0.GaussianTail

noncomputable section

/-- Exponential domination of the Gaussian positive-part expectation. -/
theorem hingeLoss_le_mgf (a : ℝ) (v : ℝ≥0) (t : ℝ) (ht : 0 < t) :
    Gaussian.hingeLoss a v ≤ Real.exp (-t * a + v * t ^ 2 / 2) / t := by
  have hp (x : ℝ) : max (x - a) 0 ≤ Real.exp (t * (x - a)) / t := by
    apply (le_div_iff₀ ht).mpr
    by_cases hx : 0 ≤ x - a
    · rw [max_eq_left hx]
      have he := Real.add_one_le_exp (t * (x - a))
      nlinarith
    · rw [max_eq_right (le_of_not_ge hx), zero_mul]
      exact (Real.exp_pos _).le
  have hexp : Integrable (fun x : ℝ => Real.exp (t * (x - a)) / t)
      (gaussianReal 0 v) := by
    have he := (integrable_exp_mul_gaussianReal (μ := (0 : ℝ)) (v := v) t).const_mul
      (Real.exp (-t * a) / t)
    convert he using 1
    ext x
    rw [div_mul_eq_mul_div, ← Real.exp_add]
    congr 2
    ring
  calc
    Gaussian.hingeLoss a v ≤ ∫ x : ℝ, Real.exp (t * (x - a)) / t
        ∂gaussianReal 0 v :=
      integral_mono (Gaussian.hinge_integrable a v) hexp hp
    _ = Real.exp (-t * a) / t * ∫ x : ℝ, Real.exp (t * x)
        ∂gaussianReal 0 v := by
      rw [← integral_const_mul]
      congr 1
      funext x
      rw [div_mul_eq_mul_div, ← Real.exp_add]
      congr 2
      ring
    _ = Real.exp (-t * a + v * t ^ 2 / 2) / t := by
      have hmgf := congrFun (mgf_fun_id_gaussianReal (μ := (0 : ℝ)) (v := v)) t
      simp only [mgf, zero_mul, zero_add] at hmgf
      rw [hmgf, div_mul_eq_mul_div, ← Real.exp_add]

/-- A uniform bound for posterior variances at most two. -/
theorem hingeLoss_le_exp (a : ℝ) (v : ℝ≥0) (ha : 1 ≤ a) (hv : (v : ℝ) ≤ 2) :
    Gaussian.hingeLoss a v ≤ 2 * Real.exp (-a ^ 2 / 4) := by
  have hapos : 0 < a := lt_of_lt_of_le zero_lt_one ha
  have he : - (a / 2) * a + (v : ℝ) * (a / 2) ^ 2 / 2 ≤ -a ^ 2 / 4 := by
    nlinarith [sq_nonneg a]
  calc
    Gaussian.hingeLoss a v ≤
        Real.exp (- (a / 2) * a + (v : ℝ) * (a / 2) ^ 2 / 2) / (a / 2) :=
      hingeLoss_le_mgf a v (a / 2) (half_pos hapos)
    _ ≤ Real.exp (-a ^ 2 / 4) / (a / 2) :=
      div_le_div_of_nonneg_right (Real.exp_le_exp.mpr he) (half_pos hapos).le
    _ ≤ 2 * Real.exp (-a ^ 2 / 4) := by
      apply (div_le_iff₀ (half_pos hapos)).mpr
      nlinarith [Real.exp_pos (-a ^ 2 / 4)]

/-- The safe exponent 225 suffices for both corrected Bellman margins. -/
theorem hingeLoss_le_inverse_power (T : ℕ) (hT : 5 ≤ T) (a : ℝ) (v : ℝ≥0)
    (ha : 30 * Real.sqrt (Real.log (T : ℝ)) ≤ a) (hv : (v : ℝ) ≤ 2) :
    Gaussian.hingeLoss a v ≤ 2 * Rates.invPower 225 T := by
  have hTr : (1 : ℝ) ≤ T := by exact_mod_cast (show 1 ≤ T by omega)
  have hTpos : (0 : ℝ) < T := by exact_mod_cast (show 0 < T by omega)
  have hlog : 0 ≤ Real.log (T : ℝ) := Real.log_nonneg hTr
  have hloglb : (1 / 4 : ℝ) < Real.log (T : ℝ) := by
    have ht : (2 : ℝ) ≤ T := by exact_mod_cast (show 2 ≤ T by omega)
    exact (by linarith [Real.log_two_gt_d9] : (1 / 4 : ℝ) < Real.log 2).trans_le
      (Real.log_le_log (by norm_num) ht)
  have hs : (1 / 2 : ℝ) < Real.sqrt (Real.log (T : ℝ)) := by
    apply (Real.lt_sqrt (by norm_num)).mpr
    nlinarith
  have ha1 : 1 ≤ a := by linarith
  have hs2 := Real.sq_sqrt hlog
  have hexp : -a ^ 2 / 4 ≤ -(225 : ℝ) * Real.log (T : ℝ) := by
    nlinarith [Real.sqrt_nonneg (Real.log (T : ℝ))]
  calc
    Gaussian.hingeLoss a v ≤ 2 * Real.exp (-a ^ 2 / 4) := hingeLoss_le_exp a v ha1 hv
    _ ≤ 2 * Real.exp (-(225 : ℝ) * Real.log (T : ℝ)) := by gcongr
    _ = 2 * Rates.invPower 225 T := by
      congr 1
      rw [neg_mul, Real.exp_neg]
      change (Real.exp ((225 : ℕ) * Real.log (T : ℝ)))⁻¹ = Rates.invPower 225 T
      rw [Real.exp_nat_mul, Real.exp_log hTpos]
      rfl

/-- The printed Y and Z conditions also give a uniform bound on each mean. -/
theorem mean_bound (T q i n : ℕ) (p : Counts.MeanPath) (hT : 5 ≤ T)
    (hn : 1 ≤ n) (hnq : n ≤ q) (hY : Drift.eventY T q i p)
    (hZ : Drift.eventZ T q i p) :
    |p i n| ≤ 9 * Real.sqrt (Real.log (T : ℝ)) := by
  have hlog := Drift.log_horizon_ge_one T hT
  have hsqrt : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := by
    exact (Real.le_sqrt (by norm_num) (by linarith)).mpr (by simpa using hlog)
  have hsum := Drift.reciprocalSqSum_bound n q hn hnq
  have hnR : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hinv : 1 / (n : ℝ) ≤ 1 := (div_le_one (by positivity)).mpr hnR
  have hS : Drift.reciprocalSqSum n q ≤ 4 := by
    have hqinv : 0 ≤ 1 / (q : ℝ) := by positivity
    linarith
  have hs : Real.sqrt (Drift.reciprocalSqSum n q) ≤ 2 := by
    exact (Real.sqrt_le_left (by norm_num)).mpr (by norm_num at *; exact hS)
  have hZ' := hZ n hn hnq
  have hTr : (1 : ℝ) ≤ T := by exact_mod_cast (show 1 ≤ T by omega)
  have hsmall : 1 / (T : ℝ) ^ 2 ≤ 1 := by
    apply (div_le_one (by positivity)).mpr
    nlinarith
  have htri : |p i n| ≤ |p i n - p i q| + |p i q| := by
    simpa only [sub_add_cancel] using abs_add_le (p i n - p i q) (p i q)
  have hprod := mul_le_mul_of_nonneg_left hs
    (show 0 ≤ 4 * Real.sqrt (Real.log (T : ℝ)) by positivity)
  unfold Drift.eventY at hY
  nlinarith

/-- Replacing the disputed regret-component lemma by a perfect-information
slack preserves a strictly faster decay than the corrected lower margins. -/
theorem bad_state_tail (T q n : ℕ) (p : Counts.MeanPath) (hT : 5 ≤ T)
    (hn : 1 ≤ n) (hnq : n ≤ q) (hX : Drift.eventX T 40 p)
    (hY : Drift.eventY T q 1 p) (hZ : Drift.eventZ T q 1 p)
    (v : ℝ≥0) (hv : (v : ℝ) ≤ 2) :
    Gaussian.hingeLoss (p 1 n - p 3 1) v ≤ 2 * Rates.invPower 225 T := by
  apply hingeLoss_le_inverse_power T hT _ v ?_ hv
  have hm := mean_bound T q 1 n p hT hn hnq hY hZ
  have hlo := (abs_le.mp hm).1
  unfold Drift.eventX at hX
  have hs : 0 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.sqrt_nonneg _
  linarith

/-- A larger underestimation cutoff gives room for the even-budget drift
estimate while retaining polynomial frequentist bad-event probability. -/
theorem hingeLoss_le_inverse_power400 (T : ℕ) (hT : 5 ≤ T) (a : ℝ) (v : ℝ≥0)
    (ha : 40 * Real.sqrt (Real.log (T : ℝ)) ≤ a) (hv : (v : ℝ) ≤ 2) :
    Gaussian.hingeLoss a v ≤ 2 * Rates.invPower 400 T := by
  have hTpos : (0 : ℝ) < T := by exact_mod_cast (show 0 < T by omega)
  have hlog := Drift.log_horizon_ge_one T hT
  have hs : 1 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.one_le_sqrt.mpr hlog
  have ha1 : 1 ≤ a := by linarith
  have hs2 := Real.sq_sqrt (show 0 ≤ Real.log (T : ℝ) by linarith)
  have hexp : -a ^ 2 / 4 ≤ -(400 : ℝ) * Real.log (T : ℝ) := by
    nlinarith [Real.sqrt_nonneg (Real.log (T : ℝ))]
  calc
    Gaussian.hingeLoss a v ≤ 2 * Real.exp (-a ^ 2 / 4) := hingeLoss_le_exp a v ha1 hv
    _ ≤ 2 * Real.exp (-(400 : ℝ) * Real.log (T : ℝ)) := by gcongr
    _ = 2 * Rates.invPower 400 T := by
      congr 1
      rw [neg_mul, Real.exp_neg]
      change (Real.exp ((400 : ℕ) * Real.log (T : ℝ)))⁻¹ = Rates.invPower 400 T
      rw [Real.exp_nat_mul, Real.exp_log hTpos]
      rfl

theorem bad_state_tail50 (T q n : ℕ) (p : Counts.MeanPath) (hT : 5 ≤ T)
    (hn : 1 ≤ n) (hnq : n ≤ q) (hX : Drift.eventX T 50 p)
    (hY : Drift.eventY T q 1 p) (hZ : Drift.eventZ T q 1 p)
    (v : ℝ≥0) (hv : (v : ℝ) ≤ 2) :
    Gaussian.hingeLoss (p 1 n - p 3 1) v ≤ 2 * Rates.invPower 400 T := by
  apply hingeLoss_le_inverse_power400 T hT _ v ?_ hv
  have hm := mean_bound T q 1 n p hT hn hnq hY hZ
  have hlo := (abs_le.mp hm).1
  unfold Drift.eventX at hX
  have hs : 0 ≤ Real.sqrt (Real.log (T : ℝ)) := Real.sqrt_nonneg _
  linarith

end

end Draft0.GaussianTail
