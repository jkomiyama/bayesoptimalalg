import Junpei.Empirical
import Junpei.PolicyAEExecution
import Junpei.ProbabilityPolynomial
import Junpei.ProbabilityExact
import Junpei.RegretExponent

open MeasureTheory ProbabilityTheory Filter Set

namespace Junpei.Main

open Draft0.BayesianModel Draft0.BadEvent Draft0.Rates
open Junpei.Policy Junpei.Empirical

noncomputable section

abbrev JointSample (T : ℕ) := RawArrays T × SeedArray (T+1)

def jointLaw (T : ℕ) : Measure (JointSample T) := (rawArraysLaw T).prod (randomizerLaw T)

instance jointLaw_probability (T : ℕ) : IsProbabilityMeasure (jointLaw T) := by
  unfold jointLaw
  infer_instance

def actualRecommendation (π : StatePolicy) (T : ℕ) (Δ : ℝ) (ω : JointSample T) : Arm :=
  recommendation π T (empiricalPath Δ ω.1) (randomPath T ω.2)

theorem actualRecommendation_measurable (π : StatePolicy) (T : ℕ) (Δ : ℝ) :
    Measurable (actualRecommendation π T Δ) :=
  recommendation_execution_measurable π T (empiricalPath Δ) (empiricalPath_measurable T Δ)

def wrongEvent (π : StatePolicy) (T : ℕ) (Δ : ℝ) : Set (JointSample T) :=
  {ω | actualRecommendation π T Δ ω ≠ 2}

theorem wrongEvent_measurable (π : StatePolicy) (T : ℕ) (Δ : ℝ) :
    MeasurableSet (wrongEvent π T Δ) :=
  ((actualRecommendation_measurable π T Δ) (measurableSet_singleton 2)).compl

/-- Actual frequentist simple regret under independent reward stacks with
true means (0,0,Delta) and independent fresh standard Gaussian randomizers. -/
def regret (π : StatePolicy) (T : ℕ) (Δ : ℝ) : ℝ :=
  ∫ ω : JointSample T, Δ - trueMean Δ (actualRecommendation π T Δ ω) ∂jointLaw T

theorem regret_eq (π : StatePolicy) (T : ℕ) (Δ : ℝ) :
    regret π T Δ = Δ * (jointLaw T).real (wrongEvent π T Δ) := by
  have hi : (fun ω : JointSample T => Δ - trueMean Δ (actualRecommendation π T Δ ω)) =
      (wrongEvent π T Δ).indicator (fun _ => Δ) := by
    funext ω
    rw [true_gap]
    by_cases h : actualRecommendation π T Δ ω=2 <;> simp [wrongEvent, h]
  rw [regret, hi, integral_indicator_const Δ (wrongEvent_measurable π T Δ), smul_eq_mul, mul_comm]

theorem regret_bounds (π : StatePolicy) (T : ℕ) (Δ : ℝ) (hΔ : 0 ≤ Δ) :
    0 ≤ regret π T Δ ∧ regret π T Δ ≤ Δ := by
  rw [regret_eq]
  exact ⟨mul_nonneg hΔ measureReal_nonneg,
    (mul_le_mul_of_nonneg_left measureReal_le_one hΔ).trans_eq (mul_one Δ)⟩

theorem regret_repair_eq (π : StatePolicy) (T : ℕ) (Δ : ℝ) (hπ : AEOptimal π T) :
    regret π T Δ = regret (repair π T) T Δ :=
  recommendation_integral_repair_eq (rawArraysLaw T) π T hπ
    (empiricalPath Δ) (empiricalPath_measurable T Δ) (fun i => Δ - trueMean Δ i)

def jointBadEvent (T q : ℕ) (Δ CU : ℝ) (hqT : q ≤ T) (hT : 0 < T) : Set (JointSample T) :=
  Junpei.Probability.rawBadEvent T q Δ CU hqT hT ×ˢ univ

theorem jointBadEvent_probability (T q : ℕ) (Δ CU : ℝ) (hqT : q ≤ T) (hT : 0 < T) :
    (jointLaw T).real (jointBadEvent T q Δ CU hqT hT) =
      (rawArraysLaw T).real (Junpei.Probability.rawBadEvent T q Δ CU hqT hT) := by
  unfold jointLaw jointBadEvent Measure.real
  rw [Measure.prod_prod, ENNReal.toReal_mul, measure_univ, ENNReal.toReal_one, mul_one]

theorem eventually_pointwise_event_lower :
    ∃ T0 : ℕ, ∀ T : ℕ, T0 ≤ T → ∀ (q : ℕ) (hTodd : T=2*q+1),
      ∀ (π : StatePolicy), PointwiseOptimal π T → ∀ (Δ : ℝ), 0 ≤ Δ →
      Δ * (rawArraysLaw T).real
        (Junpei.Probability.rawBadEvent T q Δ 33 (by omega) (by omega)) ≤ regret π T Δ := by
  obtain ⟨N, hN⟩ := Junpei.PolicyChoice.eventually_optimal_choices 33 (by norm_num)
  refine ⟨max N 16, ?_⟩
  intro T hlarge q hT π hπ Δ hΔ
  have hNlarge : N ≤ T := (le_max_left N 16).trans hlarge
  have hT16 : 16 ≤ T := (le_max_right N 16).trans hlarge
  have hchoices := hN T hNlarge
  have hsubset : jointBadEvent T q Δ 33 (by omega) (by omega) ⊆ wrongEvent π T Δ := by
    intro ω hω
    have hp := badEvent_goodPath T q Δ 33 (by omega) (by omega) ω.1 hω.1
    exact Junpei.PathInduction.optimal_bad_path_wrong π T q 33 hT (by omega) (by norm_num)
      hπ (empiricalPath Δ ω.1) (randomPath T ω.2) hp
      (fun t c a ht hpre hc hW ha => hchoices.1 q t (Draft0.PathInduction.paperPath
        (empiricalPath Δ ω.1)) c a hT ht hpre hc hW ha)
      (fun c a hc hW ha => hchoices.2 q (Draft0.PathInduction.paperPath
        (empiricalPath Δ ω.1)) c a hT hc hW ha)
  rw [regret_eq, ← jointBadEvent_probability T q Δ 33 (by omega) (by omega)]
  exact mul_le_mul_of_nonneg_left (measureReal_mono hsubset) hΔ

/-- The complete polynomial lower bound has arbitrary positive gap, every
odd sufficiently large budget, and every randomized posterior-optimal policy.
The constants are chosen before the policy, so are uniform over all policies. -/
theorem theorem2 (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (C T0 : ℕ), 0 < c ∧ 0 < C ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, AEOptimal π T →
        c * invPower C T ≤ regret π T Δ := by
  obtain ⟨B, hB⟩ := exists_nat_ge (33 + Δ + 1)
  obtain ⟨N, hN⟩ := eventually_pointwise_event_lower
  refine ⟨Δ * (underestimationConstant/64), B^2+4, max N 16,
    mul_pos hΔ (div_pos underestimationConstant_positive (by norm_num)), by omega, ?_⟩
  intro T hlarge hodd π hπ
  obtain ⟨q, hq⟩ := hodd
  have hT : T=2*q+1 := by omega
  have hT16 : 16 ≤ T := (le_max_right N 16).trans hlarge
  have hp := Junpei.Probability.rawBadEvent_polynomial_lower T q B Δ 33
    hT16 hT hΔ.le (by norm_num) hB
  have he := hN T ((le_max_left N 16).trans hlarge) q hT (repair π T)
    (repair_pointwiseOptimal π T) Δ hΔ.le
  rw [regret_repair_eq π T Δ hπ]
  calc
    (Δ * (underestimationConstant/64)) * invPower (B^2+4) T =
        Δ * ((underestimationConstant/64) * invPower (B^2+4) T) := mul_assoc _ _ _
    _ ≤ Δ * (rawArraysLaw T).real
        (Junpei.Probability.rawBadEvent T q Δ 33 (by omega) (by omega)) :=
      mul_le_mul_of_nonneg_left hp hΔ.le
    _ ≤ regret (repair π T) T Δ := he

theorem theorem2_exponential_rate (Δ : ℝ) (hΔ : 0 < Δ) (family : ℕ → StatePolicy)
    (hfamily : ∀ T, 3 ≤ T → AEOptimal (family T) T) :
    Filter.liminf (Junpei.RegretExponent.rate (fun T => regret (family T) T Δ)) atTop = 0 := by
  obtain ⟨c, C, T0, hc, _, hmain⟩ := theorem2 Δ hΔ
  apply Junpei.RegretExponent.liminf_rate_eq_zero _ Δ c C hΔ hc
    (fun T => regret_bounds (family T) T Δ hΔ.le)
  filter_upwards [eventually_ge_atTop (max T0 3)] with T hT
  intro hodd
  exact hmain T ((le_max_left T0 3).trans hT) hodd (family T)
    (hfamily T ((le_max_right T0 3).trans hT))

end
end Junpei.Main
