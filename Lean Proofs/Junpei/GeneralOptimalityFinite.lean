import Junpei.GeneralOptimalityProper

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSectionVars false
set_option maxHeartbeats 1500000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.GeneralOptimality.FiniteHorizon

open Junpei.FiniteGaussian Junpei.FinitePosterior
open Junpei.PolicyContinuation Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open ProperDomain (State)

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

/-- The manuscript's finite collection of Borel posterior-state rules.
There are exactly N sampling rules and one terminal rule, all defined only
on proper Gaussian posteriors. Index j is the number of draws remaining
after that sampling decision, so the calendar order is N-1,...,0. -/
structure Policy (A : Type*) [MeasurableSpace A] (N : ℕ) where
  draw : Fin N → State A → ℝ → A
  recommend : State A → ℝ → A
  draw_measurable : ∀ j, Measurable (fun p : State A × ℝ => draw j p.1 p.2)
  recommend_measurable : Measurable (fun p : State A × ℝ => recommend p.1 p.2)

variable {N : ℕ}

/-- Pad the finite collection by a constant arm at unused indices. Thus
the paper's policy has no extra rules or Borel requirements outside its budget. -/
def extend (π : Policy A N) : ProperDomain.Policy A where
  draw := fun H s u => if hH : H < N then π.draw ⟨H, hH⟩ s u
    else Classical.choice (inferInstance : Nonempty A)
  recommend := π.recommend
  draw_measurable := by
    intro H
    by_cases hH : H < N
    · simpa only [dif_pos hH] using π.draw_measurable ⟨H, hH⟩
    · simpa only [dif_neg hH] using
        (measurable_const : Measurable (fun _ : State A × ℝ =>
          Classical.choice (inferInstance : Nonempty A)))
  recommend_measurable := π.recommend_measurable

theorem extend_draw (π : Policy A N) (j : Fin N) (s : State A) (u : ℝ) :
    (extend π).draw j.val s u = π.draw j s u := by
  simp only [extend, dif_pos j.isLt]

theorem extend_recommend (π : Policy A N) (s : State A) (u : ℝ) :
    (extend π).recommend s u = π.recommend s u := rfl

/-- Direct raw-observation execution of the finite collection itself.
Every draw index carries its proof of membership in Fin N. -/
def rawTree (π : Policy A N) : (H : ℕ) → H ≤ N →
    State A → Seeds H → Strategy (A := A) H
  | 0, _, s, u => .stop (π.recommend s (u ⟨0, by omega⟩))
  | H + 1, hH, s, u =>
      let a := π.draw ⟨H, by omega⟩ s (u ⟨0, by omega⟩)
      .sample a (fun y => rawTree π H (by omega)
        (ProperDomain.update s a y) (seedTail u))

/-- Padding changes no raw strategy at any stage of the given budget. -/
theorem rawTree_extend (π : Policy A N) (H : ℕ) (hH : H ≤ N)
    (s : State A) (u : Seeds H) :
    ProperDomain.rawTree (extend π) H s u = rawTree π H hH s u := by
  induction H generalizing s with
  | zero => rfl
  | succ H ih =>
    have hlt : H < N := by omega
    simp only [ProperDomain.rawTree, rawTree, extend, dif_pos hlt]
    congr 1
    funext y
    exact ih (by omega) _ (seedTail u)

/-- Actual posterior expected simple regret of the finite policy's direct
latent-parameter and raw-observation execution. -/
def actualRisk (π : Policy A N) (H : ℕ) (hH : H ≤ N) (s : State A) : ℝ :=
  ∫ u, rawRisk (rawTree π H hH s u) s.1.1 s.1.2 ∂Junpei.Policy.seedLaw (H + 1)

theorem actualRisk_extend (π : Policy A N) (H : ℕ) (hH : H ≤ N) (s : State A) :
    ProperDomain.actualRisk (extend π) H s = actualRisk π H hH s := by
  simp only [ProperDomain.actualRisk, actualRisk, rawTree_extend π H hH]

/-- Definition 3's minimization at one proper posterior and one remaining
horizon, against all concrete Borel history continuation policies. -/
def MinimizesHistoryRisk (π : Policy A N) (H : ℕ) (hH : H ≤ N) (s : State A) : Prop :=
  ∀ ρ : HistoryPolicy.Policy A,
    actualRisk π H hH s ≤ HistoryPolicy.actualRisk ρ 0 H
      GeneralOptimality.emptyHistory s.1.1 s.1.2

theorem minimizing_extend_iff (π : Policy A N) (H : ℕ) (hH : H ≤ N) (s : State A) :
    ProperDomain.MinimizesHistoryRisk (extend π) H s ↔
      MinimizesHistoryRisk π H hH s := by
  simp only [ProperDomain.MinimizesHistoryRisk, MinimizesHistoryRisk, actualRisk_extend π H hH]

/-- Equation (2) for a fixed finite collection of posterior-state rules.
There is no premise or policy rule at a horizon larger than N. -/
def SequentiallyOptimal (π : Policy A N) : Prop :=
  ∀ (H : ℕ) (hH : H ≤ N) (s : State A), MinimizesHistoryRisk π H hH s

theorem sequentiallyOptimal_extend (π : Policy A N) (hπ : SequentiallyOptimal π) :
    ProperDomain.SequentiallyOptimalUpTo (extend π) N := by
  intro H hH s
  exact (minimizing_extend_iff π H hH s).mpr (hπ H hH s)

section Ordered
variable [LinearOrder A]

/-- Theorem 1 for the literal finite collection of Borel proper-state
rules: every actual-risk minimizing algorithm has the Bellman risk, chooses
an EBI-maximizing arm, and recommends a largest posterior mean almost surely.
All conclusions concern the direct finite-policy execution. -/
theorem theorem1 (π : Policy A N) (hπ : SequentiallyOptimal π) :
    (∀ (H : ℕ) (hH : H ≤ N) (s : State A),
      actualRisk π H hH s = loss H s.1.1 s.1.2) ∧
    (∀ j : Fin N, ∀ s : State A, ∀ᵐ u : ℝ ∂gaussianReal 0 1,
      ∀ a, BayesOptimal.EBI j.val s.1.1 s.1.2 a ≤
        BayesOptimal.EBI j.val s.1.1 s.1.2 (π.draw j s u)) ∧
    (∀ s : State A, ∀ᵐ u : ℝ ∂gaussianReal 0 1,
      s.1.2 (π.recommend s u) = maxAll s.1.2) := by
  obtain ⟨hr, hd, ht⟩ := ProperDomain.theorem1 (extend π) N
    (sequentiallyOptimal_extend π hπ)
  refine ⟨?_, ?_, ?_⟩
  · intro H hH s
    simpa only [actualRisk_extend π H hH] using hr H hH s
  · intro j s
    simpa only [extend_draw] using hd j.val j.isLt s
  · intro s
    simpa only [extend_recommend] using ht s

/-- The minimum-index Bellman selector is a member of the literal finite
paper-policy class, including the case N=0 with only a terminal decision. -/
def canonical : Policy A N where
  draw := fun j s _ => JointGaussianMeasurable.minimumOptimal j.val s.1.1 s.1.2
  recommend := fun s _ => MockTie.minimumMaximizer s.1.2
  draw_measurable := fun j =>
    (JointGaussianMeasurable.minimumOptimal_joint_measurable j.val).comp
      (measurable_subtype_coe.comp measurable_fst)
  recommend_measurable := MockTie.minimumMaximizer_measurable.comp
    (measurable_snd.comp (measurable_subtype_coe.comp measurable_fst))

theorem canonical_rawTree (H : ℕ) (hH : H ≤ N) (s : State A) (u : Seeds H) :
    rawTree (canonical (A := A) (N := N)) H hH s u =
      RealPrecisionPolicy.rawTree RealPrecisionPolicy.canonical H s.1.1 s.1.2 u := by
  induction H generalizing s with
  | zero => rfl
  | succ H ih =>
    simp only [rawTree, canonical, RealPrecisionPolicy.rawTree, RealPrecisionPolicy.canonical]
    congr 1
    funext y
    exact ih (by omega)
      (ProperDomain.update s (JointGaussianMeasurable.minimumOptimal H s.1.1 s.1.2) y)
      (seedTail u)

theorem canonical_actualRisk (H : ℕ) (hH : H ≤ N) (s : State A) :
    actualRisk (canonical (A := A) (N := N)) H hH s = loss H s.1.1 s.1.2 := by
  unfold actualRisk
  simp only [canonical_rawTree, RealPrecisionPolicy.canonical_rawRisk H s.1.1 s.1.2 _ s.2]
  simp

theorem canonical_sequentiallyOptimal :
    SequentiallyOptimal (canonical (A := A) (N := N)) := by
  intro H hH s ρ
  rw [canonical_actualRisk H hH s]
  exact HistoryPolicy.randomized_history_risk_ge_loss ρ 0 H
    GeneralOptimality.emptyHistory s.1.1 s.1.2 s.2

/-- The finite proper-state policy class used by the exact paper theorem
has a concrete Borel minimizing policy for every fixed budget. -/
theorem exists_sequentiallyOptimal (N : ℕ) :
    ∃ π : Policy A N, SequentiallyOptimal π :=
  ⟨canonical, canonical_sequentiallyOptimal⟩

end Ordered
end
end Junpei.GeneralOptimality.FiniteHorizon
