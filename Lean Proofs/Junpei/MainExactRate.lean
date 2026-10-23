import Junpei.Main
import Junpei.ProbabilityExactRate

open Filter

namespace Junpei.Main

open Draft0.BadEvent Junpei.Policy

noncomputable section

/-- The exponent explicitly proposed in the revised main proof, with CU=33.
The bound holds uniformly over all randomized posterior-optimal policies. -/
theorem theorem2_paper_exponent (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (T0 : ℕ), 0 < c ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, AEOptimal π T →
        c * (T : ℝ)^(-((33+Δ)^2/2+4)) ≤ regret π T Δ := by
  obtain ⟨N, hN⟩ := eventually_pointwise_event_lower
  obtain ⟨M, hM⟩ := Filter.eventually_atTop.mp
    (Junpei.Probability.rawBadEvent_exact_rate_eventually Δ 33 hΔ (by norm_num))
  refine ⟨Δ * Junpei.Probability.exactRateConstant Δ 33, max N M,
    mul_pos hΔ (Junpei.Probability.exactRateConstant_positive Δ 33 hΔ (by norm_num)), ?_⟩
  intro T hlarge hodd π hπ
  obtain ⟨q, hq⟩ := hodd
  have hTodd : T=2*q+1 := by omega
  have hp := hM T ((le_max_right N M).trans hlarge) q hTodd
  have he := hN T ((le_max_left N M).trans hlarge) q hTodd (repair π T)
    (repair_pointwiseOptimal π T) Δ hΔ.le
  rw [regret_repair_eq π T Δ hπ]
  calc
    (Δ * Junpei.Probability.exactRateConstant Δ 33) * (T : ℝ)^(-((33+Δ)^2/2+4)) =
        Δ * (Junpei.Probability.exactRateConstant Δ 33 * (T : ℝ)^(-((33+Δ)^2/2+4))) := mul_assoc _ _ _
    _ ≤ Δ * (rawArraysLaw T).real (Junpei.Probability.rawBadEvent T q Δ 33 (by omega) (by omega)) :=
      mul_le_mul_of_nonneg_left hp hΔ.le
    _ ≤ regret (repair π T) T Δ := he

/-- The main theorem in the paper's real-exponent existential form. -/
theorem theorem2_real_exponent (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c C : ℝ) (T0 : ℕ), 0 < c ∧ 0 < C ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, AEOptimal π T →
        c * (T : ℝ)^(-C) ≤ regret π T Δ := by
  obtain ⟨c, T0, hc, hmain⟩ := theorem2_paper_exponent Δ hΔ
  exact ⟨c, (33+Δ)^2/2+4, T0, hc, by positivity, hmain⟩

end
end Junpei.Main
