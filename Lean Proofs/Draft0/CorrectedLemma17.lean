import Draft0.Drift
import Draft0.PredictiveLaw
import Draft0.PredictiveBound

/-!
The repaired nonterminal Lemma 17 for the explicitly defined balanced
Gaussian-innovation forecast. The deterministic W event is the full printed
event, including its anchored Y and Z restrictions. The forecast is an
actual probability measure, not a symbolic probability or numerical model.
Connecting this law to the paper's conditional filtration remains a separate
model-identification obligation.
-/

open MeasureTheory ProbabilityTheory Set
open Draft0.Counts Draft0.Drift Draft0.PredictiveLaw Draft0.PredictiveBound

namespace Draft0.CorrectedLemma17

noncomputable section

/-- The corrected positive-interval forecast bound on every admissible W state. -/
theorem lemma17_forecast_probability (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q CU p c) :
    lowerConstant * Draft0.Rates.invPower 206 T ≤
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
      204 * Real.log (T : ℝ) * (accumulatedVariance (T - t) c : ℝ) := by
    rw [hVcoe]
    exact lemma17_fullW_drift_bound T q t CU p c hT ht hpre hc hW
  rw [balancedPredictiveLaw_gaussian]
  exact gaussian_interval_probability_drift204 T hT5 hv hupper hlower hδ

/-- The absolute-gap event used by the paper contains the positive interval. -/
theorem lemma17_forecast_absolute_gap_probability (T q t : ℕ) (CU : ℝ)
    (p : MeanPath) (c : Counts)
    (hT : T = 2 * q + 1) (ht : 4 ≤ t) (hpre : t < T)
    (hc : horizonConsistent t c) (hW : paperW T q CU p c) :
    lowerConstant * Draft0.Rates.invPower 206 T ≤
      (balancedPredictiveLaw (T - t) c (terminalGap p c)
        {x | gapBand T |x|}).toReal := by
  have hsub : Icc (Draft0.Rates.invPower 2 T) (2 * Draft0.Rates.invPower 2 T) ⊆
      {x | gapBand T |x|} := by
    intro x hx
    have hxnonneg : 0 ≤ x := (Draft0.Rates.invPower_nonneg 2 T).trans hx.1
    change gapBand T |x|
    unfold gapBand
    rw [abs_of_nonneg hxnonneg]
    simpa [Draft0.Rates.invPower, one_div, mul_div_assoc] using hx
  have hm := lemma17_forecast_probability T q t CU p c hT ht hpre hc hW
  rw [balancedPredictiveLaw_gaussian] at hm ⊢
  exact hm.trans (measureReal_mono hsub)

end

end Draft0.CorrectedLemma17
