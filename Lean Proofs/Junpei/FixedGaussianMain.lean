import Junpei.FixedGaussianMainLaw
import Junpei.MainExactRate
import Junpei.RegretRatio

open MeasureTheory ProbabilityTheory Filter

namespace Junpei.FixedGaussian

open Draft0.BayesianModel Junpei.Policy Junpei.InitialOptimal

noncomputable section

/-- The full main regret bound with initialization-only actual-risk
optimality. Neither sequential nor local Bellman optimality is a hypothesis. -/
theorem theorem2_initialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (C T0 : ℕ), 0 < c ∧ 0 < C ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, InitialRiskOptimal π T →
        c * Draft0.Rates.invPower C T ≤ Junpei.Main.regret π T Δ := by
  obtain ⟨c, C, T0, hc, hC, hmain⟩ := Junpei.Main.theorem2 Δ hΔ
  refine ⟨c, C, max T0 3, hc, hC, ?_⟩
  intro T hT ho π hπ
  have h3 : 3 ≤ T := (le_max_right T0 3).trans hT
  rw [initialRiskOptimal_regret_repair_eq π T Δ h3 hπ]
  exact hmain T ((le_max_left T0 3).trans hT) ho (repair π T)
    (pointwiseOptimal_aeOptimal _ _ (repair_pointwiseOptimal π T))

theorem theorem2_paper_exponent_initialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (T0 : ℕ), 0 < c ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, InitialRiskOptimal π T →
        c * (T : ℝ)^(-((33+Δ)^2/2+4)) ≤ Junpei.Main.regret π T Δ := by
  obtain ⟨c, T0, hc, hmain⟩ := Junpei.Main.theorem2_paper_exponent Δ hΔ
  refine ⟨c, max T0 3, hc, ?_⟩
  intro T hT ho π hπ
  have h3 : 3 ≤ T := (le_max_right T0 3).trans hT
  rw [initialRiskOptimal_regret_repair_eq π T Δ h3 hπ]
  exact hmain T ((le_max_left T0 3).trans hT) ho (repair π T)
    (pointwiseOptimal_aeOptimal _ _ (repair_pointwiseOptimal π T))

theorem theorem2_real_exponent_initialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c C : ℝ) (T0 : ℕ), 0 < c ∧ 0 < C ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, InitialRiskOptimal π T →
        c * (T : ℝ)^(-C) ≤ Junpei.Main.regret π T Δ := by
  obtain ⟨c, T0, hc, hmain⟩ := theorem2_paper_exponent_initialRiskOptimal Δ hΔ
  exact ⟨c, (33+Δ)^2/2+4, T0, hc, by positivity, hmain⟩

theorem theorem2_exponential_rate_initialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ)
    (family : ℕ → StatePolicy) (hfamily : ∀ T, 3 ≤ T → InitialRiskOptimal (family T) T) :
    liminf (Junpei.RegretExponent.rate
      (fun T => Junpei.Main.regret (family T) T Δ)) atTop = 0 := by
  obtain ⟨c, C, T0, hc, _, hmain⟩ := theorem2_initialRiskOptimal Δ hΔ
  apply Junpei.RegretExponent.liminf_rate_eq_zero _ Δ c C hΔ hc
    (fun T => Junpei.Main.regret_bounds (family T) T Δ hΔ.le)
  filter_upwards [eventually_ge_atTop (max T0 3)] with T hT
  intro ho
  exact hmain T ((le_max_left T0 3).trans hT) ho (family T)
    (hfamily T ((le_max_right T0 3).trans hT))

theorem remark2_initialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ)
    (family : ℕ → StatePolicy) (hfamily : ∀ T, 3 ≤ T → InitialRiskOptimal (family T) T)
    (rFreq : ℕ → ℝ) (hF : ∀ T, 0 ≤ rFreq T)
    (hpositive : 0 < liminf (Junpei.RegretExponent.rate rFreq) atTop) :
    limsup (Junpei.RegretRatio.ratio
      (fun T => Junpei.Main.regret (family T) T Δ) rFreq) atTop = ⊤ := by
  obtain ⟨c, C, T0, hc, _, hmain⟩ := theorem2_initialRiskOptimal Δ hΔ
  apply Junpei.RegretRatio.limsup_ratio_eq_top_of_positive_rate _ rFreq C c hc _ hF hpositive
  filter_upwards [eventually_ge_atTop (max T0 3)] with T hT
  intro ho
  exact hmain T ((le_max_left T0 3).trans hT) ho (family T)
    (hfamily T ((le_max_right T0 3).trans hT))

/-- A concrete witness ensures that the initialization-only policy class is
nonempty; the main theorem is not a statement about an empty hypothesis. -/
theorem exists_initialRiskOptimal (T : ℕ) (hT : 3 ≤ T) :
    ∃ π : StatePolicy, InitialRiskOptimal π T := by
  refine ⟨deterministicBellman T, ?_⟩
  intro m
  exact Junpei.SequentialRisk.deterministicBellman_minimizes_actual_risk T 4 (T - 3)
    (fun _ => 1) m (by omega) (by intro a; norm_num)

end
end Junpei.FixedGaussian
