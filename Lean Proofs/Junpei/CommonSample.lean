import Junpei.TwoArmOptimal
import Junpei.Information
import Junpei.ForecastRules

/-! Explicit common observation streams and deterministic observation schedules.
Every coordinate is a Gaussian posterior-mean innovation indexed by arm and
current integer precision. The corresponding observation is obtained by the
normal conjugate update. Permuting an allocation preserves the common samples
pointwise, and preserves its actual posterior-predictive terminal regret. -/

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.CommonSample

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman
open Junpei.TwoArmOptimal

noncomputable section

def step (i : Bool) (c : Counts) : Counts := if i then leftStep c else rightStep c

def sampled (i : Bool) (c : Counts) : ℕ := if i then c.n1 else c.n2

def finalCounts : List Bool → Counts → Counts
  | [], c => c
  | i :: s, c => finalCounts s (step i c)

def allocationVariance : List Bool → Counts → ℝ≥0
  | [], _ => 0
  | i :: s, c => updateVariance (sampled i c) + allocationVariance s (step i c)

def utility : List Bool → Counts → ℝ → ℝ → ℝ
  | [], _, m1, m2 => max m1 m2
  | i :: s, c, m1, m2 => if i then
      ∫ z : ℝ, utility s (leftStep c) (m1 + z) m2
        ∂gaussianReal 0 (updateVariance c.n1)
    else ∫ z : ℝ, utility s (rightStep c) m1 (m2 + z)
        ∂gaussianReal 0 (updateVariance c.n2)

def potential (c : Counts) (m1 m2 : ℝ) : ℝ :=
  FinitePosterior.potential (pairPrecisions c.n1 c.n2) (pairMeans m1 m2)

def terminalRisk (c : Counts) (m1 m2 : ℝ) : ℝ := potential c m1 m2 - max m1 m2

def risk : List Bool → Counts → ℝ → ℝ → ℝ
  | [], c, m1, m2 => terminalRisk c m1 m2
  | i :: s, c, m1, m2 => if i then
      ∫ z : ℝ, risk s (leftStep c) (m1 + z) m2
        ∂gaussianReal 0 (updateVariance c.n1)
    else ∫ z : ℝ, risk s (rightStep c) m1 (m2 + z)
        ∂gaussianReal 0 (updateVariance c.n2)

theorem terminalRisk_is_actual_regret (c : Counts) (m1 m2 : ℝ) :
    terminalRisk c m1 m2 = FinitePosterior.terminalRegret
      (pairPrecisions c.n1 c.n2) (pairMeans m1 m2) (if m1 ≤ m2 then 1 else 0) := by
  rw [FinitePosterior.terminalRegret_eq]
  unfold terminalRisk potential
  congr 1
  by_cases h : m1 ≤ m2
  · simp [h, pairMeans]
  · simp [h, pairMeans, max_eq_left (le_of_not_ge h)]

theorem utility_eq (s : List Bool) (c : Counts) (m1 m2 : ℝ) :
    utility s c m1 m2 = m2 + hingeLoss (m2 - m1) (allocationVariance s c) := by
  induction s generalizing c m1 m2 with
  | nil => simpa [utility, allocationVariance] using (payoff_eq 0 m1 m2 c)
  | cons i s ih =>
    cases i <;> simp only [utility, allocationVariance, step, sampled, Bool.false_eq_true,
      ↓reduceIte, ih]
    · exact right_formula_integral _ _ _ _
    · exact left_formula_integral _ _ _ _

theorem allocationVariance_decomposition (s : List Bool) (c : Counts)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    allocationVariance s c + variance (finalCounts s c) = variance c := by
  induction s generalizing c with
  | nil => simp [allocationVariance, finalCounts]
  | cons i s ih =>
    cases i <;> simp only [allocationVariance, finalCounts, step, sampled,
      Bool.false_eq_true, ↓reduceIte]
    · rw [add_assoc, ih (rightStep c) (by simpa [rightStep] using h1)
        (by simp [rightStep])]
      exact (rightStep_variance c h2).symm
    · rw [add_assoc, ih (leftStep c) (by simp [leftStep])
        (by simpa [leftStep] using h2)]
      exact (leftStep_variance c h1).symm

theorem left_potential_integrable (c : Counts) (m1 m2 : ℝ) :
    Integrable (fun z : ℝ => potential (leftStep c) (m1 + z) m2)
      (gaussianReal 0 (updateVariance c.n1)) := by
  have h := FiniteGaussian.shifted_integrable
    (FinitePosterior.potential_lipschitz (pairPrecisions (c.n1 + 1) c.n2))
    (pairMeans m1 m2) 0 (updateVariance c.n1)
  simpa only [potential, leftStep, shift_pair_left] using h

theorem right_potential_integrable (c : Counts) (m1 m2 : ℝ) :
    Integrable (fun z : ℝ => potential (rightStep c) m1 (m2 + z))
      (gaussianReal 0 (updateVariance c.n2)) := by
  have h := FiniteGaussian.shifted_integrable
    (FinitePosterior.potential_lipschitz (pairPrecisions c.n1 (c.n2 + 1)))
    (pairMeans m1 m2) 1 (updateVariance c.n2)
  simpa only [potential, rightStep, shift_pair_right] using h

theorem left_potential_tower (c : Counts) (m1 m2 : ℝ) (h1 : 0 < c.n1) :
    (∫ z : ℝ, potential (leftStep c) (m1 + z) m2
      ∂gaussianReal 0 (updateVariance c.n1)) = potential c m1 m2 := by
  have h := FinitePosterior.potential_tower (pairPrecisions c.n1 c.n2)
    (pairMeans m1 m2) 0 (by change (0 : ℝ≥0) < c.n1; exact_mod_cast h1)
  rw [advance_pair_left] at h
  simpa [potential, leftStep, shift_pair_left,
    FiniteGaussian.innovation, pairPrecisions] using h

theorem right_potential_tower (c : Counts) (m1 m2 : ℝ) (h2 : 0 < c.n2) :
    (∫ z : ℝ, potential (rightStep c) m1 (m2 + z)
      ∂gaussianReal 0 (updateVariance c.n2)) = potential c m1 m2 := by
  have h := FinitePosterior.potential_tower (pairPrecisions c.n1 c.n2)
    (pairMeans m1 m2) 1 (by change (0 : ℝ≥0) < c.n2; exact_mod_cast h2)
  rw [advance_pair_right] at h
  simpa [potential, rightStep, shift_pair_right,
    FiniteGaussian.innovation, pairPrecisions] using h

/-- The risk is the iterated integral of actual posterior latent regret, not
an assumed potential-minus-payoff definition. -/
theorem risk_eq (s : List Bool) (c : Counts) (m1 m2 : ℝ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) :
    risk s c m1 m2 = potential c m1 m2 - utility s c m1 m2 := by
  induction s generalizing c m1 m2 with
  | nil => rfl
  | cons i s ih =>
    cases i <;> simp only [risk, utility, Bool.false_eq_true, ↓reduceIte]
    · simp_rw [ih (rightStep c) _ _ (by simpa [rightStep] using h1)
        (by simp [rightStep])]
      have hu : Integrable (fun z : ℝ => utility s (rightStep c) m1 (m2 + z))
          (gaussianReal 0 (updateVariance c.n2)) := by
        simp only [utility_eq]
        exact right_formula_integrable _ _ _ _
      rw [integral_sub (right_potential_integrable c m1 m2) hu,
        right_potential_tower c m1 m2 h2]
    · simp_rw [ih (leftStep c) _ _ (by simp [leftStep])
        (by simpa [leftStep] using h2)]
      have hu : Integrable (fun z : ℝ => utility s (leftStep c) (m1 + z) m2)
          (gaussianReal 0 (updateVariance c.n1)) := by
        simp only [utility_eq]
        exact left_formula_integrable _ _ _ _
      rw [integral_sub (left_potential_integrable c m1 m2) hu,
        left_potential_tower c m1 m2 h1]

/-- Exact allocation-only dependence of the posterior-predictive terminal risk. -/
theorem risk_depends_only_terminal_counts (s u : List Bool) (c : Counts) (m1 m2 : ℝ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hfinal : finalCounts s c = finalCounts u c) :
    risk s c m1 m2 = risk u c m1 m2 := by
  have hs := allocationVariance_decomposition s c h1 h2
  have hu := allocationVariance_decomposition u c h1 h2
  rw [hfinal] at hs
  have hv : allocationVariance s c = allocationVariance u c := add_right_cancel (hs.trans hu.symm)
  rw [risk_eq s c m1 m2 h1 h2, risk_eq u c m1 m2 h1 h2, utility_eq, utility_eq, hv]

abbrev Stream := Bool → ℕ → ℝ

def commonUpdate (i : Bool) (c : Counts) (m : ℝ × ℝ) (z : Stream) : ℝ × ℝ :=
  if i then (m.1 + z true c.n1, m.2) else (m.1, m.2 + z false c.n2)

def terminalMeans : List Bool → Counts → (ℝ × ℝ) → Stream → (ℝ × ℝ)
  | [], _, m, _ => m
  | i :: s, c, m, z => terminalMeans s (step i c) (commonUpdate i c m z) z

theorem step_commute (i j : Bool) (c : Counts) : step j (step i c) = step i (step j c) := by
  cases i <;> cases j <;> rfl

theorem commonUpdate_commute (i j : Bool) (c : Counts) (m : ℝ × ℝ) (z : Stream) :
    commonUpdate j (step i c) (commonUpdate i c m z) z =
      commonUpdate i (step j c) (commonUpdate j c m z) z := by
  cases i <;> cases j <;> rfl

theorem finalCounts_permutation (s u : List Bool) (h : s.Perm u) (c : Counts) :
    finalCounts s c = finalCounts u c := by
  induction h generalizing c with
  | nil => rfl
  | cons i h ih => exact ih (step i c)
  | swap i j s => simp only [finalCounts, step_commute]
  | trans h₁ h₂ ih₁ ih₂ => exact (ih₁ c).trans (ih₂ c)

/-- Coupling is pointwise on a single per-arm stream, stronger than equality in law. -/
theorem terminalMeans_permutation (s u : List Bool) (h : s.Perm u)
    (c : Counts) (m : ℝ × ℝ) (z : Stream) :
    terminalMeans s c m z = terminalMeans u c m z := by
  induction h generalizing c m with
  | nil => rfl
  | cons i h ih => exact ih (step i c) (commonUpdate i c m z)
  | swap i j s => simp only [terminalMeans, step_commute, commonUpdate_commute]
  | trans h₁ h₂ ih₁ ih₂ => exact (ih₁ c m).trans (ih₂ c m)

theorem risk_permutation (s u : List Bool) (h : s.Perm u) (c : Counts) (m1 m2 : ℝ)
    (h1 : 0 < c.n1) (h2 : 0 < c.n2) : risk s c m1 m2 = risk u c m1 m2 :=
  risk_depends_only_terminal_counts s u c m1 m2 h1 h2 (finalCounts_permutation s u h c)

/-- A single probability space supplies all common samples and either final sample. -/
def streamLaw (N : ℕ) : Measure (Bool × Fin N → ℝ) :=
  Measure.pi (fun j => gaussianReal 0 (updateVariance (j.2.val : ℝ≥0)))

instance streamLaw_probability (N : ℕ) : IsProbabilityMeasure (streamLaw N) := by
  unfold streamLaw
  infer_instance

theorem stream_coordinates_independent (N : ℕ) :
    iIndepFun (fun j : Bool × Fin N => fun x : Bool × Fin N → ℝ => x j) (streamLaw N) := by
  exact iIndepFun_pi (fun _ => measurable_id.aemeasurable)

def finiteStream (N : ℕ) (x : Bool × Fin N → ℝ) : Stream :=
  fun i n => if h : n < N then x (i, ⟨n, h⟩) else 0

theorem stream_coordinate_law (N n : ℕ) (i : Bool) (hn : n < N) :
    (streamLaw N).map (fun x => finiteStream N x i n) =
      gaussianReal 0 (updateVariance (n : ℝ≥0)) := by
  simpa only [finiteStream, dif_pos hn] using
    (measurePreserving_eval (fun j : Bool × Fin N =>
      gaussianReal 0 (updateVariance (j.2.val : ℝ≥0))) (i, ⟨n, hn⟩)).map_eq

/-- Raw observation corresponding to the independent conjugate innovation. -/
def observation (n : ℕ) (m z : ℝ) : ℝ := m + (n + 1 : ℝ) * z

theorem observation_conjugate_update (n : ℕ) (m z : ℝ) :
    ((n : ℝ) * m + observation n m z) / (n + 1 : ℝ) = m + z := by
  unfold observation
  field_simp
  ring

theorem observation_predictive_law (n : ℕ) (m : ℝ) (hn : 0 < n) :
    (gaussianReal 0 (updateVariance (n : ℝ≥0))).map (observation n m) =
      gaussianReal m (1 + (n : ℝ≥0)⁻¹) := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hn)
  have hnp0 : (n : ℝ) + 1 ≠ 0 := by positivity
  let sqn : ℝ≥0 := ⟨((n : ℝ) + 1) ^ 2, sq_nonneg ((n : ℝ) + 1)⟩
  have hv : sqn * updateVariance (n : ℝ≥0) =
      1 + (n : ℝ≥0)⁻¹ := by
    apply NNReal.coe_injective
    change ((n : ℝ) + 1) ^ 2 * ((n : ℝ) * ((n : ℝ) + 1))⁻¹ = 1 + (n : ℝ)⁻¹
    field_simp [hn0, hnp0]
  unfold observation
  change Measure.map ((fun x : ℝ => m + x) ∘ (fun x : ℝ => ((n : ℝ) + 1) * x))
    (gaussianReal 0 (updateVariance (n : ℝ≥0))) = _
  rw [← Measure.map_map (by fun_prop : Measurable (fun x : ℝ => m + x))
    (by fun_prop : Measurable (fun x : ℝ => (n + 1 : ℝ) * x)),
    gaussianReal_map_const_mul, mul_zero, hv, gaussianReal_map_const_add, zero_add]

end
end Junpei.CommonSample
