import Draft0.Drift
import Draft0.TerminalBounds

namespace Draft0.TerminalDrift

open Draft0.Counts Draft0.Drift Draft0.Gaussian Draft0.TerminalBounds
open scoped BigOperators NNReal

noncomputable section

theorem terminal_anchorSum_le_updateVariance (q : ℕ) (c : Counts)
    (hq : 2 ≤ q) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 = 2 * q - 1) :
    reciprocalSqSum c.n1 q + reciprocalSqSum c.n2 q ≤
      6 * (updateVariance (q : ℝ≥0) : ℝ) := by
  have hqr : 2 ≤ (q : ℝ) := by exact_mod_cast hq
  have hqp : 0 < (q : ℝ) := by linarith
  have hqsp : 0 < (q : ℝ) - 1 := by linarith
  have hq0 : (q : ℝ) ≠ 0 := by linarith
  have hqsub0 : (q : ℝ) - 1 ≠ 0 := by linarith
  have hqadd0 : (q : ℝ) + 1 ≠ 0 := by linarith
  have hqsub : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - 1 := by
    simp only [Nat.cast_sub (show 1 ≤ q by omega), Nat.cast_one]
  have hVar : (updateVariance (q : ℝ≥0) : ℝ) =
      1 / ((q : ℝ) * ((q : ℝ) + 1)) := by
    simp only [updateVariance, NNReal.coe_inv, NNReal.coe_mul, NNReal.coe_add,
      NNReal.coe_natCast, NNReal.coe_one, one_div]
  have hdiff : 1 / ((q : ℝ) - 1) - 1 / (q : ℝ) =
      1 / ((q : ℝ) * ((q : ℝ) - 1)) := by
    field_simp
    ring
  have hratio : 2 * (1 / ((q : ℝ) - 1) - 1 / (q : ℝ)) ≤
      6 * (updateVariance (q : ℝ≥0) : ℝ) := by
    calc
      _ = 2 / ((q : ℝ) * ((q : ℝ) - 1)) := by rw [hdiff]; ring
      _ ≤ 6 / ((q : ℝ) * ((q : ℝ) + 1)) := by
        apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
        nlinarith
      _ = 6 * (updateVariance (q : ℝ≥0) : ℝ) := by rw [hVar]; ring
  have hbound := reciprocalSqSum_bound (q - 1) q (by omega) (by omega)
  rw [hqsub] at hbound
  have hc : (c.n1 = q ∧ c.n2 = q - 1) ∨ (c.n1 = q - 1 ∧ c.n2 = q) := by omega
  rcases hc with ⟨hc1, hc2⟩ | ⟨hc1, hc2⟩
  · rw [hc1, hc2]
    simpa only [reciprocalSqSum, Finset.Ico_self, Finset.sum_empty, zero_add] using hbound.trans hratio
  · rw [hc1, hc2]
    simpa only [reciprocalSqSum, Finset.Ico_self, Finset.sum_empty, add_zero] using hbound.trans hratio

/-- Terminal replacement for the drift-to-variance estimate: the terminal
forecast variance is zero, but the sampled-arm innovation variance remains
positive and controls the full W-event drift. -/
theorem terminal_fullW_drift_bound (T q : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (hT5 : 5 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q CU p c) :
    (terminalGap p c) ^ 2 ≤
      204 * Real.log (T : ℝ) * (updateVariance (q : ℝ≥0) : ℝ) := by
  rcases hc with ⟨hc1, hc2, hc3, htotal⟩
  rcases hW with ⟨_, hY1, hY2, hZ1, hZ2, h1, h2, _⟩
  have hq : 2 ≤ q := by omega
  have hsum : c.n1 + c.n2 = 2 * q - 1 := by omega
  have hlog := log_horizon_ge_one T hT5
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith
  have hS1 := reciprocalSqSum_nonnegative c.n1 q
  have hS2 := reciprocalSqSum_nonnegative c.n2 q
  have hSsum := terminal_anchorSum_le_updateVariance q c hq h1 h2 hsum
  have hZ1' := hZ1 c.n1 hc1 h1
  have hZ2' := hZ2 c.n2 hc2 h2
  have hanchors : |p 1 q - p 2 q| ≤ 2 / (T : ℝ) ^ 2 := by
    have htri := abs_sub (p 1 q) (p 2 q)
    unfold eventY at hY1 hY2
    calc
      _ ≤ |p 1 q| + |p 2 q| := htri
      _ ≤ 1 / (T : ℝ) ^ 2 + 1 / (T : ℝ) ^ 2 := add_le_add hY1 hY2
      _ = _ := by ring
  have htri1 := abs_sub_le (p 1 c.n1) (p 1 q) (p 2 c.n2)
  have htri2 := abs_sub_le (p 1 q) (p 2 q) (p 2 c.n2)
  rw [abs_sub_comm (p 2 q) (p 2 c.n2)] at htri2
  let A : ℝ := 4 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (reciprocalSqSum c.n1 q)
  let B : ℝ := 4 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (reciprocalSqSum c.n2 q)
  let C : ℝ := 2 / (T : ℝ) ^ 2
  have hA0 : 0 ≤ A := by dsimp [A]; positivity
  have hB0 : 0 ≤ B := by dsimp [B]; positivity
  have hC0 : 0 ≤ C := by dsimp [C]; positivity
  have habs : |terminalGap p c| ≤ A + B + C := by
    dsimp [A, B, C]
    unfold terminalGap
    linarith only [htri1, htri2, hanchors, hZ1', hZ2']
  have hsq : (terminalGap p c) ^ 2 ≤ (A + B + C) ^ 2 := by
    simpa only [sq_abs] using
      (sq_le_sq₀ (abs_nonneg (terminalGap p c)) (by positivity)).mpr habs
  have hcq : c.n1 = q ∨ c.n2 = q := by omega
  have hAB : A * B = 0 := by
    rcases hcq with hq1 | hq2
    · simp [A, hq1, reciprocalSqSum]
    · simp [B, hq2, reciprocalSqSum]
  have htwo : (A + B + C) ^ 2 ≤ 2 * (A ^ 2 + B ^ 2 + C ^ 2) := by
    nlinarith [sq_nonneg (A + B - C)]
  have hAsq : A ^ 2 = 16 * Real.log (T : ℝ) * reciprocalSqSum c.n1 q := by
    dsimp [A]
    rw [mul_pow, mul_pow, Real.sq_sqrt hlog0, Real.sq_sqrt hS1]
    norm_num
  have hBsq : B ^ 2 = 16 * Real.log (T : ℝ) * reciprocalSqSum c.n2 q := by
    dsimp [B]
    rw [mul_pow, mul_pow, Real.sq_sqrt hlog0, Real.sq_sqrt hS2]
    norm_num
  have hCsq : C ^ 2 = 4 / (T : ℝ) ^ 4 := by
    dsimp [C]
    rw [div_pow, ← pow_mul]
    norm_num
  have hSscaled := mul_le_mul_of_nonneg_left hSsum hlog0
  have hTreal : 5 ≤ (T : ℝ) := by exact_mod_cast hT5
  have hinv : 1 / (T : ℝ) ^ 4 ≤ 1 / (T : ℝ) ^ 2 := by
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith [sq_nonneg ((T : ℝ) ^ 2 - 1)]
  have hcount := updateVariance_count_bounds q T (by omega) (by omega)
  have hcountLower : 1 / (T : ℝ) ^ 2 ≤ (updateVariance (q : ℝ≥0) : ℝ) := by
    simpa only [Draft0.Rates.invPower, inv_eq_one_div] using hcount.1
  have hv0 : 0 ≤ (updateVariance (q : ℝ≥0) : ℝ) := (updateVariance (q : ℝ≥0)).property
  have hVscaled : (updateVariance (q : ℝ≥0) : ℝ) ≤
      Real.log (T : ℝ) * (updateVariance (q : ℝ≥0) : ℝ) := by
    simpa using mul_le_mul_of_nonneg_right hlog hv0
  have hinvFinal : 1 / (T : ℝ) ^ 4 ≤
      Real.log (T : ℝ) * (updateVariance (q : ℝ≥0) : ℝ) :=
    hinv.trans (hcountLower.trans hVscaled)
  have hinvScaled : 4 / (T : ℝ) ^ 4 ≤
      4 * (Real.log (T : ℝ) * (updateVariance (q : ℝ≥0) : ℝ)) := by
    calc
      _ = 4 * (1 / (T : ℝ) ^ 4) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hinvFinal (by norm_num)
  rw [hAsq, hBsq, hCsq] at htwo
  nlinarith only [hsq, htwo, hSscaled, hinvScaled, mul_nonneg hlog0 hv0]

end

end Draft0.TerminalDrift
