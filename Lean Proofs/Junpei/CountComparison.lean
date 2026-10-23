import Draft0.CorrectedLemma17
import Draft0.BadEvent

set_option warningAsError true

open MeasureTheory ProbabilityTheory Set
open Draft0.Counts Draft0.Drift Draft0.PredictiveLaw Draft0.PredictiveBound

namespace JunpeiComparison

noncomputable section

-- Exact coefficient 3 printed on PDF page 13.
def newZ (T q i : ℕ) (p : MeanPath) : Prop :=
  ∀ n : ℕ, 1 ≤ n → n ≤ q →
    |p i n - p i q| ≤ 3 * Real.sqrt (Real.log (T : ℝ)) *
      Real.sqrt (reciprocalSqSum n q)

def newW (T q : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts) : Prop :=
  eventX T CU p ∧ eventY T q 1 p ∧ eventY T q 2 p ∧
    newZ T q 1 p ∧ newZ T q 2 p ∧
    c.n1 ≤ q ∧ c.n2 ≤ q ∧ c.n3 ≤ 1

-- PDF pages 34-35 state a multiset rather than a fixed orientation.
theorem forecast_terminal_multiset (T q t : ℕ) (c : Counts)
    (hT : T = 2 * q + 1) (hq : 0 < q) (ht : 4 ≤ t) (horizon : t ≤ T)
    (hc : horizonConsistent t c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    ((balancedRun (T - t) c).n1 = q ∧ (balancedRun (T - t) c).n2 = q - 1) ∨
    ((balancedRun (T - t) c).n1 = q - 1 ∧ (balancedRun (T - t) c).n2 = q) := by
  have he := equation22_generic_horizon T q t c hT hq ht horizon hc h1 h2
  by_cases hs : c.n2 = q
  · right
    simpa [hs] using he
  · left
    simpa [hs] using he

theorem new_equation23_drift160 (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) :
    (terminalGap p c) ^ 2 ≤ 160 * Real.log (T : ℝ) * transitionVariance (T - t) c := by
  rcases hc with ⟨hc1, hc2, hc3, htotal⟩
  rcases hW with ⟨_hX, hY1, hY2, hZ1, hZ2, h1, h2, _h3⟩
  have hq : 2 ≤ q := by omega
  have hT5 : 5 ≤ T := by omega
  have hTreal : 5 ≤ (T : ℝ) := by exact_mod_cast hT5
  have hk : 0 < T - t := by omega
  have hsum : c.n1 + c.n2 + (T - t) = 2 * q - 1 := by omega
  have hlog := log_horizon_ge_one T hT5
  have hlog0 : 0 ≤ Real.log (T : ℝ) := by linarith
  have hS1 := reciprocalSqSum_nonnegative c.n1 q
  have hS2 := reciprocalSqSum_nonnegative c.n2 q
  have hB1 := reciprocalSqSum_bound c.n1 q hc1 h1
  have hB2 := reciprocalSqSum_bound c.n2 q hc2 h2
  have hanchor := to_anchor_variance_bound q (T - t) c hq hk hc1 hc2 h1 h2 hsum
  have hSsum : reciprocalSqSum c.n1 q + reciprocalSqSum c.n2 q ≤
      4 * transitionVariance (T - t) c := by linarith
  have hV0 := transitionVariance_nonnegative (T - t) c (by omega) (by omega)
  have hVl := lemma17_nonterminal_variance_bound T q t c hT (by omega) ht hpre
    ⟨hc1, hc2, hc3, htotal⟩ h1 h2
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
  have hVscaled : transitionVariance (T - t) c ≤
      Real.log (T : ℝ) * transitionVariance (T - t) c := by
    simpa using mul_le_mul_of_nonneg_right hlog hV0
  have hinvFinal : 1 / (T : ℝ) ^ 4 ≤
      Real.log (T : ℝ) * transitionVariance (T - t) c :=
    hinv.trans (hVl.trans hVscaled)
  have hinvScaled : 4 / (T : ℝ) ^ 4 ≤
      4 * (Real.log (T : ℝ) * transitionVariance (T - t) c) := by
    calc
      4 / (T : ℝ) ^ 4 = 4 * (1 / (T : ℝ) ^ 4) := by ring
      _ ≤ 4 * (Real.log (T : ℝ) * transitionVariance (T - t) c) :=
        mul_le_mul_of_nonneg_left hinvFinal (by norm_num)
  rw [hAsq, hBsq, hCsq] at hthree
  nlinarith only [hsq, hthree, hSscaled, hinvScaled, mul_nonneg hlog0 hV0]

theorem new_lemma18_fixed_tie162 (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) :
    lowerConstant * Draft0.Rates.invPower 162 T ≤
      (balancedPredictiveLaw (T - t) c (terminalGap p c)
        (Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T))).toReal := by
  have hq : 0 < q := by omega
  have hT5 : 5 ≤ T := by omega
  have hk : 0 < T - t := by omega
  have h1 : 0 < c.n1 := by have := hc.1; omega
  have h2 : 0 < c.n2 := by have := hc.2.1; omega
  have hcap1 : c.n1 ≤ q := hW.2.2.2.2.2.1
  have hcap2 : c.n2 ≤ q := hW.2.2.2.2.2.2.1
  have hVcoe := accumulatedVariance_coe (T - t) c h1 h2
  have hv : 0 < accumulatedVariance (T - t) c := by
    have hVr : 0 < (accumulatedVariance (T - t) c : ℝ) := by
      rw [hVcoe]
      exact transitionVariance_positive (T - t) c hk h1 h2
    exact hVr
  have hupper : (accumulatedVariance (T - t) c : ℝ) ≤ 2 := by
    rw [hVcoe]
    exact transitionVariance_le_two (T - t) c hc.1 hc.2.1
  have hlower : Draft0.Rates.invPower 2 T ≤
      (accumulatedVariance (T - t) c : ℝ) := by
    rw [Draft0.Rates.invPower, inv_eq_one_div, hVcoe]
    exact lemma17_nonterminal_variance_bound T q t c hT hq ht hpre hc hcap1 hcap2
  have hδ : (terminalGap p c) ^ 2 ≤
      160 * Real.log (T : ℝ) * (accumulatedVariance (T - t) c : ℝ) := by
    rw [hVcoe]
    exact new_equation23_drift160 T q t CU p c hT ht hpre hc hW
  rw [balancedPredictiveLaw_gaussian]
  exact gaussian_interval_probability_lower_nat 160 T hT5 hv hupper hlower hδ


-- A direct Lean certificate for the paper's conservative T^-170 bound,
-- for the certified count-only rule that resolves every tie in favor of arm 1.
theorem new_lemma18_fixed_tie170 (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : newW T q CU p c) :
    lowerConstant * Draft0.Rates.invPower 170 T ≤
      (balancedPredictiveLaw (T - t) c (terminalGap p c)
        (Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T))).toReal := by
  have hm := new_lemma18_fixed_tie162 T q t CU p c hT ht hpre hc hW
  have hpow := Draft0.Rates.invPower_antitone (by norm_num : 162 ≤ 170)
    (show 1 ≤ T by omega)
  exact (mul_le_mul_of_nonneg_left hpow lowerConstant_pos.le).trans hm

-- The normalized exponent 164 printed on PDF page 35 is also checked.
theorem new_density_exponent164 {t δ x v : ℝ}
    (ht : 1 ≤ t) (hlog : 1 ≤ Real.log t) (hv : 0 < v)
    (hvlower : (t ^ 2)⁻¹ ≤ v)
    (hδ : δ ^ 2 ≤ 160 * Real.log t * v)
    (hx : x ∈ Icc ((t ^ 2)⁻¹) (2 * (t ^ 2)⁻¹)) :
    (x - δ) ^ 2 / (2 * v) ≤ 164 * Real.log t := by
  have hqnonneg : 0 ≤ (t ^ 2)⁻¹ := inv_nonneg.mpr (sq_nonneg t)
  have hqle : (t ^ 2)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ (one_le_pow₀ ht)
  have hxnonneg : 0 ≤ x := hqnonneg.trans hx.1
  have hxx : x ^ 2 ≤ 4 * v := by
    have hqq : ((t ^ 2)⁻¹) ^ 2 ≤ (t ^ 2)⁻¹ := by nlinarith
    have hsq : x ^ 2 ≤ (2 * (t ^ 2)⁻¹) ^ 2 :=
      (sq_le_sq₀ hxnonneg (by positivity)).mpr hx.2
    nlinarith
  have hquad : (x - δ) ^ 2 ≤ (160 * Real.log t + 4) * (2 * v) := by
    nlinarith [sq_nonneg (x + δ)]
  have hquot : (x - δ) ^ 2 / (2 * v) ≤ 160 * Real.log t + 4 :=
    (div_le_iff₀ (mul_pos (by norm_num) hv)).mpr hquad
  linarith

end

end JunpeiComparison

#print axioms JunpeiComparison.forecast_terminal_multiset
#print axioms JunpeiComparison.new_equation23_drift160
#print axioms JunpeiComparison.new_lemma18_fixed_tie170
#print axioms JunpeiComparison.new_density_exponent164
