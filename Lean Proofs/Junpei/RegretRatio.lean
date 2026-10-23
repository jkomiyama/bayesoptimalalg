import Junpei.MainExactRate

open Filter
open scoped Topology

namespace Junpei.RegretRatio

open Draft0.Rates Junpei.RegretExponent

noncomputable section

/-- The paper assigns +infinity whenever the comparison regret is zero. -/
def ratio (rBayes rFreq : ℕ → ℝ) (T : ℕ) : EReal :=
  if rFreq T = 0 then ⊤ else ((rBayes T / rFreq T : ℝ) : EReal)

theorem ratio_lower (rBayes rFreq : ℕ → ℝ) (T C : ℕ) (c a : ℝ)
    (hT : 0 < T) (hc : 0 < c)
    (hpoly : c * invPower C T ≤ rBayes T)
    (hF : 0 ≤ rFreq T) (hupper : rFreq T ≤ Real.exp (-a * (T : ℝ))) :
    ((c * Real.exp (a * (T : ℝ)) / (T : ℝ) ^ C : ℝ) : EReal) ≤
      ratio rBayes rFreq T := by
  by_cases hz : rFreq T = 0
  · simp [ratio, hz]
  have hFp : 0 < rFreq T := lt_of_le_of_ne hF (Ne.symm hz)
  have hBp : 0 < rBayes T := (mul_pos hc (invPower_pos C T hT)).trans_le hpoly
  rw [ratio, if_neg hz]
  apply EReal.coe_le_coe
  calc
    _ = (c * invPower C T) / Real.exp (-a * (T : ℝ)) := by
      unfold invPower
      rw [neg_mul, Real.exp_neg, div_inv_eq_mul]
      ring
    _ ≤ rBayes T / Real.exp (-a * (T : ℝ)) :=
      div_le_div_of_nonneg_right hpoly (Real.exp_pos _).le
    _ ≤ rBayes T / rFreq T := div_le_div_of_nonneg_left hBp.le hFp hupper

theorem polynomial_exp_ratio_tendsto (C : ℕ) (c a : ℝ) (hc : 0 < c) (ha : 0 < a) :
    Tendsto (fun T : ℕ => c * Real.exp (a * (T : ℝ)) / (T : ℝ) ^ C) atTop atTop := by
  have h := ((tendsto_exp_mul_div_rpow_atTop (C : ℝ) a ha).const_mul_atTop hc).comp
    (tendsto_natCast_atTop_atTop : Tendsto (fun T : ℕ => (T : ℝ)) atTop atTop)
  simpa only [Function.comp_apply, Real.rpow_natCast, mul_div_assoc] using h

/-- Odd-horizon polynomial lower regret dominates any exponential regret
envelope, with the exact zero-denominator convention of Remark 2. -/
theorem limsup_ratio_eq_top_of_exponential_envelope
    (rBayes rFreq : ℕ → ℝ) (C : ℕ) (c a : ℝ) (hc : 0 < c) (ha : 0 < a)
    (hpoly : ∀ᶠ T : ℕ in atTop, Odd T → c * invPower C T ≤ rBayes T)
    (hF : ∀ T, 0 ≤ rFreq T)
    (hupper : ∀ᶠ T : ℕ in atTop, rFreq T ≤ Real.exp (-a * (T : ℝ))) :
    limsup (ratio rBayes rFreq) atTop = ⊤ := by
  have hlarge := (polynomial_exp_ratio_tendsto C c a hc ha).comp oddBudget_tendsto
  have hodd : Tendsto (fun n : ℕ => ratio rBayes rFreq (oddBudget n)) atTop (𝓝 ⊤) := by
    apply EReal.tendsto_nhds_top_iff_real.mpr
    intro x
    filter_upwards [hlarge.eventually_gt_atTop x,
      oddBudget_tendsto.eventually hpoly, oddBudget_tendsto.eventually hupper]
      with n hn hp hu
    have hh : Odd (oddBudget n) := ⟨n, by unfold oddBudget; omega⟩
    have hl := ratio_lower rBayes rFreq (oddBudget n) C c a
      (by unfold oddBudget; omega) hc (hp hh) (hF _) hu
    exact (EReal.coe_lt_coe_iff.mpr hn).trans_le hl
  have hsub : limsup (ratio rBayes rFreq) (map oddBudget atTop) ≤
      limsup (ratio rBayes rFreq) atTop := limsup_le_limsup_of_le oddBudget_tendsto
  rw [← limsup_comp] at hsub
  change limsup (fun n => ratio rBayes rFreq (oddBudget n)) atTop ≤ _ at hsub
  rw [hodd.limsup_eq] at hsub
  exact top_le_iff.mp hsub

/-- A positive extended-real frequentist rate supplies a positive exponential
envelope, even when some regrets are zero. -/
theorem exponential_envelope_of_positive_rate (r : ℕ → ℝ) (hr : ∀ T, 0 ≤ r T)
    (hpositive : 0 < liminf (rate r) atTop) :
    ∃ a : ℝ, 0 < a ∧ ∀ᶠ T : ℕ in atTop, r T ≤ Real.exp (-a * (T : ℝ)) := by
  obtain ⟨a, ha0, ha⟩ := EReal.exists_between_coe_real hpositive
  have hap : 0 < a := EReal.coe_lt_coe_iff.mp (by simpa only [EReal.coe_zero] using ha0)
  refine ⟨a, hap, ?_⟩
  filter_upwards [eventually_lt_of_lt_liminf ha, eventually_ge_atTop (1 : ℕ)]
    with T hrate hT
  by_cases hz : r T = 0
  · rw [hz]
    exact (Real.exp_pos _).le
  have hrp : 0 < r T := lt_of_le_of_ne (hr T) (Ne.symm hz)
  have hTp : 0 < (T : ℝ) := by exact_mod_cast (show 0 < T by omega)
  rw [rate, if_neg hz] at hrate
  have hreal := EReal.coe_lt_coe_iff.mp hrate
  apply (Real.log_le_iff_le_exp hrp).mp
  have hprod := (lt_div_iff₀ hTp).mp hreal
  linarith only [hprod]

theorem limsup_ratio_eq_top_of_positive_rate
    (rBayes rFreq : ℕ → ℝ) (C : ℕ) (c : ℝ) (hc : 0 < c)
    (hpoly : ∀ᶠ T : ℕ in atTop, Odd T → c * invPower C T ≤ rBayes T)
    (hF : ∀ T, 0 ≤ rFreq T) (hpositive : 0 < liminf (rate rFreq) atTop) :
    limsup (ratio rBayes rFreq) atTop = ⊤ := by
  obtain ⟨a, ha, hupper⟩ := exponential_envelope_of_positive_rate rFreq hF hpositive
  exact limsup_ratio_eq_top_of_exponential_envelope rBayes rFreq C c a hc ha hpoly hF hupper

/-- Revised Remark 2 for every posterior-optimal policy family and every
nonnegative comparison regret with a positive frequentist rate. -/
theorem remark2 (Δ : ℝ) (hΔ : 0 < Δ) (family : ℕ → Junpei.Policy.StatePolicy)
    (hfamily : ∀ T, 3 ≤ T → Junpei.Policy.AEOptimal (family T) T)
    (rFreq : ℕ → ℝ) (hF : ∀ T, 0 ≤ rFreq T)
    (hpositive : 0 < liminf (rate rFreq) atTop) :
    limsup (ratio (fun T => Junpei.Main.regret (family T) T Δ) rFreq) atTop = ⊤ := by
  obtain ⟨c, C, T0, hc, _, hmain⟩ := Junpei.Main.theorem2 Δ hΔ
  apply limsup_ratio_eq_top_of_positive_rate _ rFreq C c hc _ hF hpositive
  filter_upwards [eventually_ge_atTop (max T0 3)] with T hT
  intro hodd
  exact hmain T ((le_max_left T0 3).trans hT) hodd (family T)
    (hfamily T ((le_max_right T0 3).trans hT))

end

end Junpei.RegretRatio
