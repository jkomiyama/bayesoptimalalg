import Draft0.Counts
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.BigOperators.Intervals

namespace Draft0.Drift

set_option maxHeartbeats 1000000

open Draft0.Counts
open scoped BigOperators

noncomputable def reciprocalSqSum (n q : ℕ) : ℝ :=
  ∑ m ∈ Finset.Ico n q, 1 / (m : ℝ) ^ 2

theorem reciprocalSqSum_nonnegative (n q : ℕ) : 0 ≤ reciprocalSqSum n q := by
  unfold reciprocalSqSum
  exact Finset.sum_nonneg (fun _ _ => by positivity)

theorem reciprocal_square_le_two_decrement (n : ℕ) (hn : 1 ≤ n) :
    1 / (n : ℝ) ^ 2 ≤ 2 * (1 / (n : ℝ) - 1 / ((n : ℝ) + 1)) := by
  have hnr : 1 ≤ (n : ℝ) := by exact_mod_cast hn
  have hp : 0 < (n : ℝ) := by linarith
  have hp1 : 0 < (n : ℝ) + 1 := by positivity
  rw [varianceDecrement_identity n (by omega)]
  unfold varianceDecrement
  rw [← mul_div_assoc]
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith

theorem reciprocalSqSum_bound (n q : ℕ) (hn : 1 ≤ n) (hnq : n ≤ q) :
    reciprocalSqSum n q ≤ 2 * (1 / (n : ℝ) - 1 / (q : ℝ)) := by
  induction q with
  | zero => omega
  | succ q ih =>
      by_cases hnq' : n ≤ q
      · unfold reciprocalSqSum
        rw [Finset.sum_Ico_succ_top hnq']
        have hprev := ih hnq'
        have hstep := reciprocal_square_le_two_decrement q (by omega)
        unfold reciprocalSqSum at hprev
        push_cast
        linarith
      · have hnqeq : n = q + 1 := by omega
        subst n
        simp [reciprocalSqSum]

theorem inverse_difference_preterminal_lower_bound (a b q : ℕ)
    (ha : 1 ≤ a) (hab : a < b) (hbq : b ≤ q) :
    1 / ((q : ℝ) * ((q : ℝ) - 1)) ≤ 1 / (a : ℝ) - 1 / (b : ℝ) := by
  have har : 0 < (a : ℝ) := by exact_mod_cast (show 0 < a by omega)
  have hbr : 0 < (b : ℝ) := by exact_mod_cast (show 0 < b by omega)
  have hqr : 2 ≤ (q : ℝ) := by exact_mod_cast (show 2 ≤ q by omega)
  have hqp : 0 < (q : ℝ) := by linarith
  have hqsubp : 0 < (q : ℝ) - 1 := by linarith
  have haq : (a : ℝ) ≤ (q : ℝ) - 1 := by
    have h : a + 1 ≤ q := by omega
    have hcast : (a : ℝ) + 1 ≤ (q : ℝ) := by exact_mod_cast h
    linarith
  have hbqr : (b : ℝ) ≤ (q : ℝ) := by exact_mod_cast hbq
  have hdelta : 1 ≤ (b : ℝ) - (a : ℝ) := by
    have hcast : (a : ℝ) + 1 ≤ (b : ℝ) := by exact_mod_cast (show a + 1 ≤ b by omega)
    linarith
  have hproduct : (a : ℝ) * (b : ℝ) ≤ ((q : ℝ) - 1) * (q : ℝ) :=
    mul_le_mul haq hbqr (le_of_lt hbr) (by linarith)
  have hscaled : (q : ℝ) * ((q : ℝ) - 1) ≤
      ((b : ℝ) - (a : ℝ)) * ((q : ℝ) * ((q : ℝ) - 1)) := by
    simpa using mul_le_mul_of_nonneg_right hdelta (by positivity :
      0 ≤ (q : ℝ) * ((q : ℝ) - 1))
  have hid : 1 / (a : ℝ) - 1 / (b : ℝ) =
      ((b : ℝ) - (a : ℝ)) / ((a : ℝ) * (b : ℝ)) := by field_simp
  rw [hid]
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith

theorem transitionVariance_residual_bound (q k : ℕ) (c : Counts)
    (hq : 2 ≤ q) (hk : 0 < k) (hc1 : 1 ≤ c.n1) (hc2 : 1 ≤ c.n2)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hsum : c.n1 + c.n2 + k = 2 * q - 1) :
    1 / ((q : ℝ) * ((q : ℝ) - 1)) ≤ transitionVariance k c := by
  have hcaps := balancedRun_preterminal_invariant q k c (by omega) h1 h2 hsum.le
  have hmono := balancedRun_monotone k c
  have htotal := balancedRun_sum k c
  have hprogress : c.n1 < (balancedRun k c).n1 ∨ c.n2 < (balancedRun k c).n2 := by omega
  unfold transitionVariance
  rcases hprogress with hinc1 | hinc2
  · have hb := inverse_difference_preterminal_lower_bound c.n1 _ q hc1 hinc1 hcaps.1
    have hn := inverse_difference_nonnegative c.n2 _ (by omega) hmono.2
    linarith
  · have hb := inverse_difference_preterminal_lower_bound c.n2 _ q hc2 hinc2 hcaps.2.1
    have hn := inverse_difference_nonnegative c.n1 _ (by omega) hmono.1
    linarith

theorem to_anchor_variance_bound (q k : ℕ) (c : Counts)
    (hq : 2 ≤ q) (hk : 0 < k) (hc1 : 1 ≤ c.n1) (hc2 : 1 ≤ c.n2)
    (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hsum : c.n1 + c.n2 + k = 2 * q - 1) :
    (1 / (c.n1 : ℝ) - 1 / (q : ℝ)) +
      (1 / (c.n2 : ℝ) - 1 / (q : ℝ)) ≤ 2 * transitionVariance k c := by
  have hterminal := equation22_terminal_counts q k c (by omega) h1 h2 hsum
  have hresidual := transitionVariance_residual_bound q k c hq hk hc1 hc2 h1 h2 hsum
  have hqr : 2 ≤ (q : ℝ) := by exact_mod_cast hq
  have hqp : (q : ℝ) ≠ 0 := by linarith
  have hqsubp : (q : ℝ) - 1 ≠ 0 := by linarith
  have hqsub : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - 1 := by
    norm_num [Nat.cast_sub (show 1 ≤ q by omega)]
  have hresid : 1 / ((q : ℝ) - 1) - 1 / (q : ℝ) =
      1 / ((q : ℝ) * ((q : ℝ) - 1)) := by
    field_simp
    ring
  unfold transitionVariance at *
  rw [hterminal.1, hterminal.2] at *
  by_cases h2q : c.n2 = q
  · simp only [if_pos h2q, hqsub] at *
    linarith
  · simp only [if_neg h2q, hqsub] at *
    linarith

theorem log_horizon_ge_one (T : ℕ) (hT : 5 ≤ T) : 1 ≤ Real.log (T : ℝ) := by
  have hTr : 5 ≤ (T : ℝ) := by exact_mod_cast hT
  apply (Real.le_log_iff_exp_le (by positivity)).mpr
  have hb := Real.exp_one_lt_d9
  linarith

def eventX (T : ℕ) (CU : ℝ) (p : MeanPath) : Prop :=
  p 3 1 ≤ -CU * Real.sqrt (Real.log (T : ℝ))

def eventY (T q i : ℕ) (p : MeanPath) : Prop := |p i q| ≤ 1 / (T : ℝ) ^ 2

def eventZ (T q i : ℕ) (p : MeanPath) : Prop :=
  ∀ n : ℕ, 1 ≤ n → n ≤ q →
    |p i n - p i q| ≤ 4 * Real.sqrt (Real.log (T : ℝ)) *
      Real.sqrt (reciprocalSqSum n q)

/-- The exact deterministic event W on page 9, including both Y and Z events
with their printed coefficient 4 and all three count restrictions. -/
def paperW (T q : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts) : Prop :=
  eventX T CU p ∧ eventY T q 1 p ∧ eventY T q 2 p ∧
    eventZ T q 1 p ∧ eventZ T q 2 p ∧
    c.n1 ≤ q ∧ c.n2 ≤ q ∧ c.n3 ≤ 1

theorem transitionVariance_le_two (k : ℕ) (c : Counts)
    (hc1 : 1 ≤ c.n1) (hc2 : 1 ≤ c.n2) : transitionVariance k c ≤ 2 := by
  have hn1 : 1 ≤ (c.n1 : ℝ) := by exact_mod_cast hc1
  have hn2 : 1 ≤ (c.n2 : ℝ) := by exact_mod_cast hc2
  have h1 : 1 / (c.n1 : ℝ) ≤ 1 := by
    apply (div_le_iff₀ (by positivity)).mpr
    linarith
  have h2 : 1 / (c.n2 : ℝ) ≤ 1 := by
    apply (div_le_iff₀ (by positivity)).mpr
    linarith
  have hs1 : 0 ≤ 1 / ((balancedRun k c).n1 : ℝ) := by positivity
  have hs2 : 0 ≤ 1 / ((balancedRun k c).n2 : ℝ) := by positivity
  unfold transitionVariance
  linarith

/-- Corrected Equation (24), proved from the complete printed W event.
The residual between the q anchor and the q-1 terminal count is retained.
This yields coefficient 204 with the printed Z coefficient 4. -/
theorem lemma17_fullW_drift_bound (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q CU p c) :
    (terminalGap p c) ^ 2 ≤ 204 * Real.log (T : ℝ) * transitionVariance (T - t) c := by
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
    have h := (sq_le_sq₀ (abs_nonneg (terminalGap p c))
      (show 0 ≤ A + B + C by positivity)).mpr habs
    simpa only [sq_abs] using h
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
  nlinarith only [hsq, hthree, hSscaled, hinvScaled]

end Draft0.Drift
