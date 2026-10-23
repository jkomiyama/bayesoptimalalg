import Junpei.CountComparison

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal
open Draft0.Counts Draft0.Drift Draft0.PredictiveBound

namespace Junpei.Forecast

noncomputable section

def finalVariance (n1 n2 f1 f2 : ℕ) : ℝ :=
  (1 / (n1 : ℝ) - 1 / (f1 : ℝ)) + (1 / (n2 : ℝ) - 1 / (f2 : ℝ))

theorem finalVariance_nonnegative (n1 n2 f1 f2 : ℕ)
    (hn1 : 1 ≤ n1) (hn2 : 1 ≤ n2) (h1 : n1 ≤ f1) (h2 : n2 ≤ f2) :
    0 ≤ finalVariance n1 n2 f1 f2 := by
  exact add_nonneg (inverse_difference_nonnegative _ _ (by omega) h1)
    (inverse_difference_nonnegative _ _ (by omega) h2)

theorem finalVariance_le_two (n1 n2 f1 f2 : ℕ)
    (hn1 : 1 ≤ n1) (hn2 : 1 ≤ n2) : finalVariance n1 n2 f1 f2 ≤ 2 := by
  have h1 : 1 / (n1 : ℝ) ≤ 1 := by
    apply (div_le_iff₀ (by exact_mod_cast (show 0 < n1 by omega))).mpr
    simpa only [one_mul] using (show (1 : ℝ) ≤ n1 by exact_mod_cast hn1)
  have h2 : 1 / (n2 : ℝ) ≤ 1 := by
    apply (div_le_iff₀ (by exact_mod_cast (show 0 < n2 by omega))).mpr
    simpa only [one_mul] using (show (1 : ℝ) ≤ n2 by exact_mod_cast hn2)
  have hf1 : 0 ≤ 1 / (f1 : ℝ) := by positivity
  have hf2 : 0 ≤ 1 / (f2 : ℝ) := by positivity
  unfold finalVariance
  linarith

theorem finalVariance_residual_lower (q n1 n2 f1 f2 : ℕ)
    (hn1 : 1 ≤ n1) (hn2 : 1 ≤ n2) (h1 : n1 ≤ f1) (h2 : n2 ≤ f2)
    (hf1 : f1 ≤ q) (hf2 : f2 ≤ q) (hprogress : n1 < f1 ∨ n2 < f2) :
    1 / ((q : ℝ) * ((q : ℝ) - 1)) ≤ finalVariance n1 n2 f1 f2 := by
  unfold finalVariance
  rcases hprogress with h | h
  · have ha := inverse_difference_preterminal_lower_bound n1 f1 q hn1 h hf1
    have hb := inverse_difference_nonnegative n2 f2 (by omega) h2
    linarith
  · have ha := inverse_difference_preterminal_lower_bound n2 f2 q hn2 h hf2
    have hb := inverse_difference_nonnegative n1 f1 (by omega) h1
    linarith

theorem finalVariance_lower (T n1 n2 f1 f2 : ℕ)
    (hn1 : 1 ≤ n1) (hn2 : 1 ≤ n2) (h1 : n1 ≤ f1) (h2 : n2 ≤ f2)
    (hf1 : f1 ≤ T) (hf2 : f2 ≤ T) (hprogress : n1 < f1 ∨ n2 < f2) :
    1 / (T : ℝ) ^ 2 ≤ finalVariance n1 n2 f1 f2 := by
  unfold finalVariance
  rcases hprogress with h | h
  · have ha := inverse_difference_lower_bound n1 f1 T (by omega) h hf1
    have hb := inverse_difference_nonnegative n2 f2 (by omega) h2
    linarith
  · have ha := inverse_difference_lower_bound n2 f2 T (by omega) h hf2
    have hb := inverse_difference_nonnegative n1 f1 (by omega) h1
    linarith

theorem finalVariance_anchor (q n1 n2 f1 f2 : ℕ) (hq : 2 ≤ q)
    (hn1 : 1 ≤ n1) (hn2 : 1 ≤ n2) (h1 : n1 ≤ f1) (h2 : n2 ≤ f2)
    (hf : (f1 = q ∧ f2 = q - 1) ∨ (f1 = q - 1 ∧ f2 = q))
    (hprogress : n1 < f1 ∨ n2 < f2) :
    (1 / (n1 : ℝ) - 1 / (q : ℝ)) + (1 / (n2 : ℝ) - 1 / (q : ℝ)) ≤
      2 * finalVariance n1 n2 f1 f2 := by
  have hf1 : f1 ≤ q := by rcases hf with h | h <;> omega
  have hf2 : f2 ≤ q := by rcases hf with h | h <;> omega
  have hl := finalVariance_residual_lower q n1 n2 f1 f2 hn1 hn2 h1 h2 hf1 hf2 hprogress
  have hq0 : (q : ℝ) ≠ 0 := by exact_mod_cast (show q ≠ 0 by omega)
  have hqsub : (q : ℝ) - 1 ≠ 0 := by
    have hqr : (2 : ℝ) ≤ q := by exact_mod_cast hq
    linarith
  have hcast : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - 1 := by
    rw [Nat.cast_sub (show 1 ≤ q by omega), Nat.cast_one]
  have hid : (1 / (n1 : ℝ) - 1 / (q : ℝ)) + (1 / (n2 : ℝ) - 1 / (q : ℝ)) =
      finalVariance n1 n2 f1 f2 + 1 / ((q : ℝ) * ((q : ℝ) - 1)) := by
    unfold finalVariance
    rcases hf with h | h
    · rw [h.1, h.2, hcast]
      field_simp
      ring
    · rw [h.1, h.2, hcast]
      field_simp
      ring
  linarith

theorem general_drift160 (T q t : Nat) (CU : Real) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 <= t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : JunpeiComparison.newW T q CU p c)
    (f1 f2 : Nat) (hn1 : c.n1 <= f1) (hn2 : c.n2 <= f2)
    (hf : (f1 = q ∧ f2 = q - 1) ∨ (f1 = q - 1 ∧ f2 = q)) :
    (terminalGap p c) ^ 2 <= 160 * Real.log (T : Real) * finalVariance c.n1 c.n2 f1 f2 := by
  rcases hc with ⟨hc1, hc2, hc3, htotal⟩
  rcases hW with ⟨_hX, hY1, hY2, hZ1, hZ2, h1, h2, _h3⟩
  have hq : 2 ≤ q := by omega
  have hT5 : 5 ≤ T := by omega
  have hTreal : 5 ≤ (T : ℝ) := by exact_mod_cast hT5
  have hlog := log_horizon_ge_one T hT5
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith
  have hS1 := reciprocalSqSum_nonnegative c.n1 q
  have hS2 := reciprocalSqSum_nonnegative c.n2 q
  have hB1 := reciprocalSqSum_bound c.n1 q hc1 h1
  have hB2 := reciprocalSqSum_bound c.n2 q hc2 h2
  have hf1 : f1 <= T := by rcases hf with h | h <;> omega
  have hf2 : f2 <= T := by rcases hf with h | h <;> omega
  have hprogress : c.n1 < f1 ∨ c.n2 < f2 := by rcases hf with h | h <;> omega
  have hanchor := finalVariance_anchor q c.n1 c.n2 f1 f2 hq hc1 hc2 hn1 hn2 hf hprogress
  have hSsum : reciprocalSqSum c.n1 q + reciprocalSqSum c.n2 q ≤
      4 * finalVariance c.n1 c.n2 f1 f2 := by linarith
  have hV0 := finalVariance_nonnegative c.n1 c.n2 f1 f2 hc1 hc2 hn1 hn2
  have hVl := finalVariance_lower T c.n1 c.n2 f1 f2 hc1 hc2 hn1 hn2 hf1 hf2 hprogress
  have hZ1' := hZ1 c.n1 hc1 h1
  have hZ2' := hZ2 c.n2 hc2 h2
  have hanchors : |p 1 q - p 2 q| ≤ 2 / (T : ℝ) ^ 2 := by
    have habs : |p 1 q - p 2 q| ≤ |p 1 q| + |p 2 q| := by
      simpa only [← sub_eq_add_neg, abs_neg] using abs_add_le (p 1 q) (-p 2 q)
    unfold eventY at hY1 hY2
    calc
      |p 1 q - p 2 q| ≤ |p 1 q| + |p 2 q| := habs
      _ ≤ 1 / (T : ℝ) ^ 2 + 1 / (T : ℝ) ^ 2 := add_le_add hY1 hY2
      _ = 2 / (T : ℝ) ^ 2 := by ring
  have htri1 := abs_sub_le (p 1 c.n1) (p 1 q) (p 2 c.n2)
  have htri2 := abs_sub_le (p 1 q) (p 2 q) (p 2 c.n2)
  rw [abs_sub_comm (p 2 q) (p 2 c.n2)] at htri2
  let A : ℝ := 3 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (reciprocalSqSum c.n1 q)
  let B : ℝ := 3 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (reciprocalSqSum c.n2 q)
  let C : ℝ := 2 / (T : ℝ) ^ 2
  have hA0 : 0 ≤ A := by dsimp [A]; positivity
  have hB0 : 0 ≤ B := by dsimp [B]; positivity
  have hC0 : 0 ≤ C := by dsimp [C]; positivity
  have habs : |terminalGap p c| ≤ A + B + C := by
    dsimp [A, B, C]
    unfold terminalGap
    linarith only [htri1, htri2, hanchors, hZ1', hZ2']
  have hsq : (terminalGap p c) ^ 2 ≤ (A + B + C) ^ 2 := by
    have h := (sq_le_sq₀ (abs_nonneg (terminalGap p c))
      (show 0 ≤ A + B + C by positivity)).mpr habs
    simpa only [sq_abs] using h
  have hthree : (A + B + C) ^ 2 ≤ 3 * (A ^ 2 + B ^ 2 + C ^ 2) := by
    nlinarith only [sq_nonneg (A - B), sq_nonneg (A - C), sq_nonneg (B - C)]
  have hAsq : A ^ 2 = 9 * Real.log (T : ℝ) * reciprocalSqSum c.n1 q := by
    dsimp [A]
    rw [mul_pow, mul_pow, Real.sq_sqrt hlog0, Real.sq_sqrt hS1]
    norm_num
  have hBsq : B ^ 2 = 9 * Real.log (T : ℝ) * reciprocalSqSum c.n2 q := by
    dsimp [B]
    rw [mul_pow, mul_pow, Real.sq_sqrt hlog0, Real.sq_sqrt hS2]
    norm_num
  have hCsq : C ^ 2 = 4 / (T : ℝ) ^ 4 := by
    dsimp [C]
    rw [div_pow, ← pow_mul]
    norm_num
  have hSscaled := mul_le_mul_of_nonneg_left hSsum hlog0
  have hinv : 1 / (T : ℝ) ^ 4 ≤ 1 / (T : ℝ) ^ 2 := by
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    have hsquare : 1 ≤ (T : ℝ) ^ 2 := by nlinarith only [hTreal]
    have hpow := mul_le_mul_of_nonneg_right hsquare (sq_nonneg (T : ℝ))
    nlinarith only [hpow]
  have hVscaled : finalVariance c.n1 c.n2 f1 f2 ≤
      Real.log (T : ℝ) * finalVariance c.n1 c.n2 f1 f2 := by
    simpa using mul_le_mul_of_nonneg_right hlog hV0
  have hinvFinal : 1 / (T : ℝ) ^ 4 ≤
      Real.log (T : ℝ) * finalVariance c.n1 c.n2 f1 f2 :=
    hinv.trans (hVl.trans hVscaled)
  have hinvScaled : 4 / (T : ℝ) ^ 4 ≤
      4 * (Real.log (T : ℝ) * finalVariance c.n1 c.n2 f1 f2) := by
    calc
      4 / (T : ℝ) ^ 4 = 4 * (1 / (T : ℝ) ^ 4) := by ring
      _ ≤ 4 * (Real.log (T : ℝ) * finalVariance c.n1 c.n2 f1 f2) :=
        mul_le_mul_of_nonneg_left hinvFinal (by norm_num)
  rw [hAsq, hBsq, hCsq] at hthree
  nlinarith only [hsq, hthree, hSscaled, hinvScaled, mul_nonneg hlog0 hV0]


theorem general_forecast_probability170 (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : JunpeiComparison.newW T q CU p c)
    (f1 f2 : ℕ) (hn1 : c.n1 ≤ f1) (hn2 : c.n2 ≤ f2)
    (hf : (f1 = q ∧ f2 = q - 1) ∨ (f1 = q - 1 ∧ f2 = q))
    (v : ℝ≥0) (hvcoe : (v : ℝ) = finalVariance c.n1 c.n2 f1 f2) :
    lowerConstant * Draft0.Rates.invPower 170 T ≤
      (gaussianReal (terminalGap p c) v
        (Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T))).toReal := by
  have hprogress : c.n1 < f1 ∨ c.n2 < f2 := by
    rcases hc with ⟨_, _, hc3, htotal⟩
    rcases hf with h | h <;> omega
  have hf1 : f1 ≤ T := by rcases hf with h | h <;> omega
  have hf2 : f2 ≤ T := by rcases hf with h | h <;> omega
  have hlower : Draft0.Rates.invPower 2 T ≤ (v : ℝ) := by
    rw [Draft0.Rates.invPower, inv_eq_one_div, hvcoe]
    exact finalVariance_lower T c.n1 c.n2 f1 f2 hc.1 hc.2.1 hn1 hn2 hf1 hf2 hprogress
  have hv : 0 < v := by
    have hpos : 0 < Draft0.Rates.invPower 2 T := Draft0.Rates.invPower_pos 2 T (by omega)
    exact hpos.trans_le hlower
  have hupper : (v : ℝ) ≤ 2 := by
    rw [hvcoe]
    exact finalVariance_le_two c.n1 c.n2 f1 f2 hc.1 hc.2.1
  have hδ : (terminalGap p c) ^ 2 ≤ 160 * Real.log (T : ℝ) * (v : ℝ) := by
    rw [hvcoe]
    exact general_drift160 T q t CU p c hT ht hpre hc hW f1 f2 hn1 hn2 hf
  have hm := gaussian_interval_probability_lower_nat 160 T (by omega) hv hupper hlower hδ
  exact (mul_le_mul_of_nonneg_left
    (Draft0.Rates.invPower_antitone (by norm_num : 162 ≤ 170) (by omega : 1 ≤ T))
    lowerConstant_pos.le).trans hm

end

end Junpei.Forecast
