import Junpei.Forecast

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal
open Draft0.Counts Draft0.Drift Draft0.PredictiveBound

namespace Junpei.Forecast

noncomputable section

/-- Every deterministic count-only tie resolution is allowed. -/
def SmallerCountRule (r : ℕ → ℕ → Bool) : Prop :=
  ∀ a b, if r a b then a ≤ b else b ≤ a

def ruleStep (r : ℕ → ℕ → Bool) (c : Counts) : Counts :=
  if r c.n1 c.n2 then ⟨c.n1 + 1, c.n2, c.n3⟩ else ⟨c.n1, c.n2 + 1, c.n3⟩

def ruleRun (r : ℕ → ℕ → Bool) : ℕ → Counts → Counts
  | 0, c => c
  | k + 1, c => ruleRun r k (ruleStep r c)

theorem ruleStep_sum (r : ℕ → ℕ → Bool) (c : Counts) :
    (ruleStep r c).n1 + (ruleStep r c).n2 = c.n1 + c.n2 + 1 := by
  unfold ruleStep
  split <;> simp <;> omega

theorem ruleStep_monotone (r : ℕ → ℕ → Bool) (c : Counts) :
    c.n1 ≤ (ruleStep r c).n1 ∧ c.n2 ≤ (ruleStep r c).n2 := by
  unfold ruleStep
  split <;> simp

theorem ruleRun_sum (r : ℕ → ℕ → Bool) (k : ℕ) (c : Counts) :
    (ruleRun r k c).n1 + (ruleRun r k c).n2 = c.n1 + c.n2 + k := by
  induction k generalizing c with
  | zero => simp [ruleRun]
  | succ k ih =>
    rw [ruleRun, ih, ruleStep_sum]
    omega

theorem ruleRun_monotone (r : ℕ → ℕ → Bool) (k : ℕ) (c : Counts) :
    c.n1 ≤ (ruleRun r k c).n1 ∧ c.n2 ≤ (ruleRun r k c).n2 := by
  induction k generalizing c with
  | zero => simp [ruleRun]
  | succ k ih =>
    have hs := ruleStep_monotone r c
    have hi := ih (ruleStep r c)
    exact ⟨hs.1.trans hi.1, hs.2.trans hi.2⟩

theorem ruleStep_caps (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (q : ℕ) (c : Counts) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hsum : c.n1 + c.n2 < 2 * q) :
    (ruleStep r c).n1 ≤ q ∧ (ruleStep r c).n2 ≤ q := by
  have hm := hr c.n1 c.n2
  cases hd : r c.n1 c.n2
  · simp [hd] at hm
    simp [ruleStep, hd]
    omega
  · simp [hd] at hm
    simp [ruleStep, hd]
    omega

theorem ruleRun_caps (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (q k : ℕ) (c : Counts) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q)
    (hsum : c.n1 + c.n2 + k ≤ 2 * q) :
    (ruleRun r k c).n1 ≤ q ∧ (ruleRun r k c).n2 ≤ q := by
  induction k generalizing c with
  | zero => exact ⟨h1, h2⟩
  | succ k ih =>
    have hs := ruleStep_caps r hr q c h1 h2 (by omega)
    have ht := ruleStep_sum r c
    exact ih (ruleStep r c) hs.1 hs.2 (by omega)

theorem ruleRun_terminal_pair (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (T q t : ℕ) (c : Counts) (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (h1 : c.n1 ≤ q) (h2 : c.n2 ≤ q) :
    ((ruleRun r (T - t) c).n1 = q ∧ (ruleRun r (T - t) c).n2 = q - 1) ∨
      ((ruleRun r (T - t) c).n1 = q - 1 ∧ (ruleRun r (T - t) c).n2 = q) := by
  rcases hc with ⟨_, _, hc3, htotal⟩
  have hcaps := ruleRun_caps r hr q (T - t) c h1 h2 (by omega)
  have hs := ruleRun_sum r (T - t) c
  omega

def ruleSampledCount (r : ℕ → ℕ → Bool) (c : Counts) : ℕ :=
  if r c.n1 c.n2 then c.n1 else c.n2

def ruleInnovation (r : ℕ → ℕ → Bool) (c : Counts) : ℝ≥0 :=
  Draft0.Gaussian.updateVariance (ruleSampledCount r c : ℝ≥0)

def ruleAccumulatedVariance (r : ℕ → ℕ → Bool) : ℕ → Counts → ℝ≥0
  | 0, _ => 0
  | k + 1, c => ruleInnovation r c + ruleAccumulatedVariance r k (ruleStep r c)

def ruleForecast (r : ℕ → ℕ → Bool) : ℕ → Counts → ℝ → Measure ℝ
  | 0, _, δ => gaussianReal δ 0
  | k + 1, c, δ => gaussianReal 0 (ruleInnovation r c) ∗ ruleForecast r k (ruleStep r c) δ

theorem ruleForecast_gaussian (r : ℕ → ℕ → Bool) (k : ℕ) (c : Counts) (δ : ℝ) :
    ruleForecast r k c δ = gaussianReal δ (ruleAccumulatedVariance r k c) := by
  induction k generalizing c with
  | zero => rfl
  | succ k ih =>
    rw [ruleForecast, ih, gaussianReal_conv_gaussianReal, zero_add]
    rfl

theorem ruleAccumulatedVariance_coe (r : ℕ → ℕ → Bool) (k : ℕ) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    (ruleAccumulatedVariance r k c : ℝ) =
      finalVariance c.n1 c.n2 (ruleRun r k c).n1 (ruleRun r k c).n2 := by
  induction k generalizing c with
  | zero => simp [ruleAccumulatedVariance, ruleRun, finalVariance]
  | succ k ih =>
    have hm := ruleStep_monotone r c
    rw [ruleAccumulatedVariance, NNReal.coe_add, ih _ (by omega) (by omega)]
    unfold ruleInnovation ruleSampledCount
    cases hd : r c.n1 c.n2
    · simp only [Bool.false_eq_true, ↓reduceIte]
      rw [Draft0.Gaussian.updateVariance, NNReal.coe_inv, NNReal.coe_mul,
        NNReal.coe_add, NNReal.coe_natCast, NNReal.coe_one]
      have hv := varianceDecrement_identity c.n2 h2
      simp only [ruleRun, ruleStep, hd, Bool.false_eq_true, ↓reduceIte, finalVariance,
        Nat.cast_add, Nat.cast_one]
      simp only [varianceDecrement, one_div] at hv
      rw [← hv]
      ring
    · simp only [↓reduceIte]
      rw [Draft0.Gaussian.updateVariance, NNReal.coe_inv, NNReal.coe_mul,
        NNReal.coe_add, NNReal.coe_natCast, NNReal.coe_one]
      have hv := varianceDecrement_identity c.n1 h1
      simp only [ruleRun, ruleStep, hd, ↓reduceIte, finalVariance, Nat.cast_add, Nat.cast_one]
      simp only [varianceDecrement, one_div] at hv
      rw [← hv]
      ring

theorem lemma18 (r : ℕ → ℕ → Bool) (hr : SmallerCountRule r)
    (T q t : ℕ) (CU : ℝ) (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : JunpeiComparison.newW T q CU p c) :
    lowerConstant * Draft0.Rates.invPower 170 T ≤
      (ruleForecast r (T - t) c (terminalGap p c)
        (Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T))).toReal := by
  have hc1 : c.n1 ≤ q := hW.2.2.2.2.2.1
  have hc2 : c.n2 ≤ q := hW.2.2.2.2.2.2.1
  have hf := ruleRun_terminal_pair r hr T q t c hT ht hpre hc hc1 hc2
  have hm := ruleRun_monotone r (T - t) c
  rw [ruleForecast_gaussian]
  exact general_forecast_probability170 T q t CU p c hT ht hpre hc hW
    _ _ hm.1 hm.2 hf _ (ruleAccumulatedVariance_coe r _ c (by have := hc.1; omega)
      (by have := hc.2.1; omega))

end

end Junpei.Forecast
