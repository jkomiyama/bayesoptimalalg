import Junpei.PosteriorStateHistory
import Junpei.FixedGaussianMain

open Filter

namespace Junpei.InitialHistoryMain

noncomputable section

open Junpei.Policy Junpei.PosteriorStateHistory

/-- The exact original frequentist risk is unchanged by the optimal repair
when the premise is minimization against concrete Borel history algorithms. -/
theorem regret_repair_eq (π : StatePolicy) (T : ℕ) (Δ : ℝ) (hT : 3 ≤ T)
    (hπ : InitialHistoryRiskOptimal π T) :
    Junpei.Main.regret π T Δ = Junpei.Main.regret (repair π T) T Δ :=
  Junpei.FixedGaussian.initialRiskOptimal_regret_repair_eq π T Δ hT
    ((initialHistoryRiskOptimal_iff π T hT).mp hπ)

theorem theorem2 (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (C T0 : ℕ), 0 < c ∧ 0 < C ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, InitialHistoryRiskOptimal π T →
        c * Draft0.Rates.invPower C T ≤ Junpei.Main.regret π T Δ := by
  obtain ⟨c, C, T0, hc, hC, hmain⟩ := Junpei.FixedGaussian.theorem2_initialRiskOptimal Δ hΔ
  refine ⟨c, C, max T0 3, hc, hC, ?_⟩
  intro T hT ho π hπ
  exact hmain T ((le_max_left T0 3).trans hT) ho π
    ((initialHistoryRiskOptimal_iff π T ((le_max_right T0 3).trans hT)).mp hπ)

theorem theorem2_paper_exponent (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (T0 : ℕ), 0 < c ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, InitialHistoryRiskOptimal π T →
        c * (T : ℝ)^(-((33+Δ)^2/2+4)) ≤ Junpei.Main.regret π T Δ := by
  obtain ⟨c, T0, hc, hmain⟩ := Junpei.FixedGaussian.theorem2_paper_exponent_initialRiskOptimal Δ hΔ
  refine ⟨c, max T0 3, hc, ?_⟩
  intro T hT ho π hπ
  exact hmain T ((le_max_left T0 3).trans hT) ho π
    ((initialHistoryRiskOptimal_iff π T ((le_max_right T0 3).trans hT)).mp hπ)

theorem theorem2_exponential_rate (Δ : ℝ) (hΔ : 0 < Δ) (family : ℕ → StatePolicy)
    (hfamily : ∀ T, 3 ≤ T → InitialHistoryRiskOptimal (family T) T) :
    liminf (Junpei.RegretExponent.rate
      (fun T => Junpei.Main.regret (family T) T Δ)) atTop = 0 := by
  apply Junpei.FixedGaussian.theorem2_exponential_rate_initialRiskOptimal Δ hΔ family
  intro T hT
  exact (initialHistoryRiskOptimal_iff _ _ hT).mp (hfamily T hT)

theorem remark2 (Δ : ℝ) (hΔ : 0 < Δ) (family : ℕ → StatePolicy)
    (hfamily : ∀ T, 3 ≤ T → InitialHistoryRiskOptimal (family T) T)
    (rFreq : ℕ → ℝ) (hF : ∀ T, 0 ≤ rFreq T)
    (hpositive : 0 < liminf (Junpei.RegretExponent.rate rFreq) atTop) :
    limsup (Junpei.RegretRatio.ratio
      (fun T => Junpei.Main.regret (family T) T Δ) rFreq) atTop = ⊤ := by
  apply Junpei.FixedGaussian.remark2_initialRiskOptimal Δ hΔ family _ rFreq hF hpositive
  intro T hT
  exact (initialHistoryRiskOptimal_iff _ _ hT).mp (hfamily T hT)

theorem exists_initialHistoryRiskOptimal (T : ℕ) (hT : 3 ≤ T) :
    ∃ π : StatePolicy, InitialHistoryRiskOptimal π T := by
  obtain ⟨π, hπ⟩ := Junpei.FixedGaussian.exists_initialRiskOptimal T hT
  exact ⟨π, (initialHistoryRiskOptimal_iff π T hT).mpr hπ⟩

end
end Junpei.InitialHistoryMain
