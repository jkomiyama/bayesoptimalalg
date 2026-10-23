import Draft0.Rates
import Mathlib.Topology.Instances.EReal.Lemmas
import Mathlib.Topology.Order.LiminfLimsup
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

open Filter
open scoped Topology

namespace Junpei.RegretExponent

noncomputable section

/-- The paper's convention -log(0)=+infinity, normalized by the horizon. -/
def rate (r : ℕ → ℝ) (T : ℕ) : EReal :=
  if r T = 0 then ⊤ else (((-Real.log (r T)) / (T : ℝ) : ℝ) : EReal)

def oddBudget (n : ℕ) : ℕ := 2 * n + 1

theorem oddBudget_tendsto : Tendsto oddBudget atTop atTop := by
  apply tendsto_atTop.mpr
  intro b
  filter_upwards [eventually_ge_atTop b] with n hn
  unfold oddBudget
  omega

theorem rate_lower_bound (r : ℕ → ℝ) (T : ℕ) (Δ : ℝ)
    (hT : 0 < T) (_hΔ : 0 < Δ) (hr : 0 ≤ r T) (hu : r T ≤ Δ) :
    (((-Real.log Δ) / (T : ℝ) : ℝ) : EReal) ≤ rate r T := by
  by_cases hz : r T = 0
  · simp [rate, hz]
  · have hrp : 0 < r T := lt_of_le_of_ne hr (Ne.symm hz)
    have hlog := Real.log_le_log hrp hu
    rw [rate, if_neg hz]
    apply EReal.coe_le_coe
    exact div_le_div_of_nonneg_right (neg_le_neg hlog)
      (by exact_mod_cast hT.le)

theorem rate_upper_bound (r : ℕ → ℝ) (T C : ℕ) (c : ℝ)
    (hT : 0 < T) (hc : 0 < c)
    (hlower : c * Draft0.Rates.invPower C T ≤ r T) :
    rate r T ≤ ((((C : ℝ) * Real.log (T : ℝ) - Real.log c) / (T : ℝ) : ℝ) : EReal) := by
  have hTp : 0 < (T : ℝ) := by exact_mod_cast hT
  have hprod : 0 < c * Draft0.Rates.invPower C T :=
    mul_pos hc (Draft0.Rates.invPower_pos C T hT)
  have hrp : 0 < r T := hprod.trans_le hlower
  have hlog := Real.log_le_log hprod hlower
  unfold Draft0.Rates.invPower at hlog
  rw [Real.log_mul (ne_of_gt hc) (by positivity), Real.log_inv, Real.log_pow] at hlog
  rw [rate, if_neg hrp.ne']
  apply EReal.coe_le_coe
  apply div_le_div_of_nonneg_right _ hTp.le
  linarith only [hlog]

theorem upper_bound_tendsto_zero (C : ℕ) (c : ℝ) :
    Tendsto (fun T : ℕ =>
      ((((C : ℝ) * Real.log (T : ℝ) - Real.log c) / (T : ℝ) : ℝ) : EReal))
      atTop (𝓝 0) := by
  have hlog : Tendsto (fun T : ℕ => Real.log (T : ℝ) / (T : ℝ)) atTop (𝓝 0) := by
    simpa using (Real.tendsto_pow_log_div_mul_add_atTop 1 0 1 one_ne_zero).comp
      (tendsto_natCast_atTop_atTop : Tendsto (fun T : ℕ => (T : ℝ)) atTop atTop)
  have hconst := tendsto_const_div_atTop_nhds_zero_nat (Real.log c)
  have hu : Tendsto (fun T : ℕ =>
      ((C : ℝ) * Real.log (T : ℝ) - Real.log c) / (T : ℝ)) atTop (𝓝 0) := by
    simpa only [mul_zero, sub_zero, sub_div, mul_div_assoc] using
      (hlog.const_mul (C : ℝ)).sub hconst
  simpa only [EReal.coe_zero] using EReal.tendsto_coe.mpr hu

/-- The exact consequence of Theorem 2: an odd-horizon polynomial lower
bound forces the full-horizon extended-real limit inferior of the normalized
negative log regret to equal zero, including zero regrets at other horizons. -/
theorem liminf_rate_eq_zero (r : ℕ → ℝ) (Δ c : ℝ) (C : ℕ)
    (hΔ : 0 < Δ) (hc : 0 < c) (hbound : ∀ T, 0 ≤ r T ∧ r T ≤ Δ)
    (hpoly : ∀ᶠ T : ℕ in atTop, Odd T → c * Draft0.Rates.invPower C T ≤ r T) :
    liminf (rate r) atTop = 0 := by
  have hlow : Tendsto (fun T : ℕ => (((-Real.log Δ) / (T : ℝ) : ℝ) : EReal))
      atTop (𝓝 0) := by
    simpa only [EReal.coe_zero] using EReal.tendsto_coe.mpr
      (tendsto_const_div_atTop_nhds_zero_nat (-Real.log Δ))
  have hglobal : ∀ᶠ T : ℕ in atTop,
      (((-Real.log Δ) / (T : ℝ) : ℝ) : EReal) ≤ rate r T := by
    filter_upwards [eventually_ge_atTop (1 : ℕ)] with T hT
    exact rate_lower_bound r T Δ (by omega) hΔ (hbound T).1 (hbound T).2
  have hge := liminf_le_liminf hglobal
  rw [hlow.liminf_eq] at hge
  have hodd : ∀ᶠ n : ℕ in atTop,
      rate r (oddBudget n) ≤
        ((((C : ℝ) * Real.log (oddBudget n : ℝ) - Real.log c) /
          (oddBudget n : ℝ) : ℝ) : EReal) := by
    filter_upwards [oddBudget_tendsto.eventually hpoly] with n hn
    apply rate_upper_bound r (oddBudget n) C c (by unfold oddBudget; omega) hc
    apply hn
    exact ⟨n, by unfold oddBudget; omega⟩
  have hoddle := liminf_le_liminf hodd
  have hoddlimit := (upper_bound_tendsto_zero C c).comp oddBudget_tendsto
  simp only [Function.comp_def] at hoddlimit
  rw [hoddlimit.liminf_eq] at hoddle
  have hsub : liminf (rate r) atTop ≤ liminf (rate r) (map oddBudget atTop) :=
    liminf_le_liminf_of_le oddBudget_tendsto
  rw [← liminf_comp] at hsub
  exact le_antisymm (hsub.trans hoddle) hge

end

end Junpei.RegretExponent
