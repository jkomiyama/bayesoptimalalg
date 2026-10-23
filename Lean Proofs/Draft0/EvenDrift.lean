import Draft0.EvenCounts
import Draft0.TerminalDrift

namespace Draft0.EvenDrift

open Draft0.Counts Draft0.Drift Draft0.Gaussian Draft0.TerminalBounds Draft0.EvenCounts
open scoped NNReal

noncomputable section

theorem fullW_gap_squared_anchor_bound (T q : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT5 : 5 ≤ T) (hc1 : 1 ≤ c.n1) (hc2 : 1 ≤ c.n2)
    (hW : paperW T q CU p c) :
    (terminalGap p c) ^ 2 ≤ 3 *
      (16 * Real.log (T : ℝ) * (reciprocalSqSum c.n1 q + reciprocalSqSum c.n2 q) +
        4 / (T : ℝ) ^ 4) := by
  rcases hW with ⟨_, hY1, hY2, hZ1, hZ2, h1, h2, _⟩
  have hlog0 : 0 ≤ Real.log (T : ℝ) := (by linarith [log_horizon_ge_one T hT5])
  have hS1 := reciprocalSqSum_nonnegative c.n1 q
  have hS2 := reciprocalSqSum_nonnegative c.n2 q
  have hZ1' := hZ1 c.n1 hc1 h1
  have hZ2' := hZ2 c.n2 hc2 h2
  have hanchors : |p 1 q - p 2 q| ≤ 2 / (T : ℝ) ^ 2 := by
    unfold eventY at hY1 hY2
    calc
      _ ≤ |p 1 q| + |p 2 q| := abs_sub _ _
      _ ≤ 1 / (T : ℝ) ^ 2 + 1 / (T : ℝ) ^ 2 := add_le_add hY1 hY2
      _ = _ := by ring
  have htri1 := abs_sub_le (p 1 c.n1) (p 1 q) (p 2 c.n2)
  have htri2 := abs_sub_le (p 1 q) (p 2 q) (p 2 c.n2)
  rw [abs_sub_comm (p 2 q) (p 2 c.n2)] at htri2
  let A : ℝ := 4 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (reciprocalSqSum c.n1 q)
  let B : ℝ := 4 * Real.sqrt (Real.log (T : ℝ)) * Real.sqrt (reciprocalSqSum c.n2 q)
  let C : ℝ := 2 / (T : ℝ) ^ 2
  have habs : |terminalGap p c| ≤ A + B + C := by
    dsimp [A, B, C]
    unfold terminalGap
    linarith only [htri1, htri2, hanchors, hZ1', hZ2']
  have hsq : (terminalGap p c) ^ 2 ≤ (A + B + C) ^ 2 := by
    simpa only [sq_abs] using
      (sq_le_sq₀ (abs_nonneg (terminalGap p c)) (by dsimp [A, B, C]; positivity)).mpr habs
  have hthree : (A + B + C) ^ 2 ≤ 3 * (A ^ 2 + B ^ 2 + C ^ 2) := by
    nlinarith only [sq_nonneg (A - B), sq_nonneg (A - C), sq_nonneg (B - C)]
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
  rw [hAsq, hBsq, hCsq] at hthree
  nlinarith only [hsq, hthree]

/-- Nonterminal drift for even budgets, including the initial saturated
orientations. The coefficient 300 retains both q-anchor residuals. -/
theorem even_fullW_drift_bound (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q CU p c) :
    (terminalGap p c) ^ 2 ≤ 300 * Real.log (T : ℝ) * transitionVariance (T - t) c := by
  rcases hc with ⟨hc1, hc2, hc3, htotal⟩
  have h1 := hW.2.2.2.2.2.1
  have h2 := hW.2.2.2.2.2.2.1
  have hq : 3 ≤ q := by omega
  have hT5 : 5 ≤ T := by omega
  have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT5
  have hk : 0 < T - t := by omega
  have hs : c.n1 + c.n2 + (T - t) = 2 * q - 2 := by omega
  have hbase := fullW_gap_squared_anchor_bound T q CU p c hT5 hc1 hc2 hW
  have hS1 := reciprocalSqSum_bound c.n1 q hc1 h1
  have hS2 := reciprocalSqSum_bound c.n2 q hc2 h2
  have hanchor := even_to_anchor_variance_bound q (T - t) c hq hk hc1 hc2 h1 h2 hs
  have hSsum : reciprocalSqSum c.n1 q + reciprocalSqSum c.n2 q ≤
      6 * transitionVariance (T - t) c := by linarith
  have hlog := log_horizon_ge_one T hT5
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith
  have hV0 := transitionVariance_nonnegative (T - t) c (by omega) (by omega)
  have hcaps := even_prefix_caps q (T - t) c (by omega) h1 h2 hs
  have hVl := transitionVariance_ge_inverse_square_of_caps (T - t) T c hk
    (by omega) (by omega) (by omega) (by omega)
  have hSscaled := mul_le_mul_of_nonneg_left hSsum hlog0
  have hinv : 1 / (T : ℝ) ^ 4 ≤ 1 / (T : ℝ) ^ 2 := by
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith [sq_nonneg ((T : ℝ) ^ 2 - 1)]
  have hVscaled : transitionVariance (T - t) c ≤
      Real.log (T : ℝ) * transitionVariance (T - t) c := by
    simpa using mul_le_mul_of_nonneg_right hlog hV0
  have hinvFinal := hinv.trans (hVl.trans hVscaled)
  have hinvScaled : 4 / (T : ℝ) ^ 4 ≤ 4 *
      (Real.log (T : ℝ) * transitionVariance (T - t) c) := by
    calc
      _ = 4 * (1 / (T : ℝ) ^ 4) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hinvFinal (by norm_num)
  nlinarith only [hbase, hSscaled, hinvScaled]

theorem even_terminal_anchorSum_le_updateVariance (q : ℕ) (c : Counts)
    (hq : 6 ≤ q) (hc1 : 1 ≤ c.n1) (hc2 : 1 ≤ c.n2)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) (hs : c.n1 + c.n2 = 2 * q - 2) :
    reciprocalSqSum c.n1 q + reciprocalSqSum c.n2 q ≤
      7 * (updateVariance (q : ℝ≥0) : ℝ) := by
  have hB1 := reciprocalSqSum_bound c.n1 q hc1 h1
  have hB2 := reciprocalSqSum_bound c.n2 q hc2 h2
  have hres := even_terminal_anchor_residual_upper q 0 c (by omega) h1 h2 (by simpa using hs)
  simp only [balancedRun] at hres
  have hscaled := mul_le_mul_of_nonneg_left hres (by norm_num : (0 : ℝ) ≤ 2)
  have hsum : reciprocalSqSum c.n1 q + reciprocalSqSum c.n2 q ≤
      4 / ((q : ℝ) * ((q : ℝ) - 2)) := by
    have he : 2 * (2 / ((q : ℝ) * ((q : ℝ) - 2))) =
        4 / ((q : ℝ) * ((q : ℝ) - 2)) := by ring
    rw [he] at hscaled
    linarith
  have hqr : 6 ≤ (q : ℝ) := by exact_mod_cast hq
  have hqp : 0 < (q : ℝ) := by linarith
  have hq2p : 0 < (q : ℝ) - 2 := by linarith
  have hratio : 4 / ((q : ℝ) * ((q : ℝ) - 2)) ≤
      7 / ((q : ℝ) * ((q : ℝ) + 1)) := by
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith
  have hVar : (updateVariance (q : ℝ≥0) : ℝ) =
      1 / ((q : ℝ) * ((q : ℝ) + 1)) := by
    simp only [updateVariance, NNReal.coe_inv, NNReal.coe_mul, NNReal.coe_add,
      NNReal.coe_natCast, NNReal.coe_one, one_div]
  calc
    _ ≤ 4 / ((q : ℝ) * ((q : ℝ) - 2)) := hsum
    _ ≤ 7 / ((q : ℝ) * ((q : ℝ) + 1)) := hratio
    _ = _ := by rw [hVar]; ring

/-- Even-budget terminal drift bound; T≥12 suffices for the eventual main
theorem. The innovation variance at count q bounds both terminal orientations. -/
theorem even_terminal_fullW_drift_bound (T q : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q) (hT12 : 12 ≤ T)
    (hc : horizonConsistent T c) (hW : paperW T q CU p c) :
    (terminalGap p c) ^ 2 ≤
      350 * Real.log (T : ℝ) * (updateVariance (q : ℝ≥0) : ℝ) := by
  rcases hc with ⟨hc1, hc2, hc3, htotal⟩
  have h1 := hW.2.2.2.2.2.1
  have h2 := hW.2.2.2.2.2.2.1
  have hq : 6 ≤ q := by omega
  have hT5 : 5 ≤ T := by omega
  have hs : c.n1 + c.n2 = 2 * q - 2 := by omega
  have hbase := fullW_gap_squared_anchor_bound T q CU p c hT5 hc1 hc2 hW
  have hSsum := even_terminal_anchorSum_le_updateVariance q c hq hc1 hc2 h1 h2 hs
  have hlog := log_horizon_ge_one T hT5
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith
  have hSscaled := mul_le_mul_of_nonneg_left hSsum hlog0
  have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT5
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
  have hinvFinal := hinv.trans (hcountLower.trans hVscaled)
  have hinvScaled : 4 / (T : ℝ) ^ 4 ≤
      4 * (Real.log (T : ℝ) * (updateVariance (q : ℝ≥0) : ℝ)) := by
    calc
      _ = 4 * (1 / (T : ℝ) ^ 4) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hinvFinal (by norm_num)
  nlinarith only [hbase, hSscaled, hinvScaled, mul_nonneg hlog0 hv0]

end

end Draft0.EvenDrift
