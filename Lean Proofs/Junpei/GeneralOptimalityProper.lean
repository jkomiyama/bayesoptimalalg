import Junpei.GeneralOptimality

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSectionVars false
set_option maxHeartbeats 1500000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.GeneralOptimality.ProperDomain

open Junpei.FiniteGaussian Junpei.FinitePosterior
open Junpei.PolicyContinuation Junpei.EmbeddingRaw Junpei.EmbeddingExecution

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

/-- Exactly the paper's proper product-Gaussian posterior domain: means
are arbitrary reals and every precision is a strictly positive real. -/
def State (A : Type*) := {p : Precisions A × Means A // ∀ a, 0 < p.1 a}

instance : MeasurableSpace (State A) := inferInstanceAs
  (MeasurableSpace {p : Precisions A × Means A // ∀ a, 0 < p.1 a})

/-- Borel policy rules defined only on proper Gaussian states. No behavior
or regularity is required at zero-precision states. -/
structure Policy (A : Type*) [MeasurableSpace A] where
  draw : ℕ → State A → ℝ → A
  recommend : State A → ℝ → A
  draw_measurable : ∀ k, Measurable (fun p : State A × ℝ => draw k p.1 p.2)
  recommend_measurable : Measurable (fun p : State A × ℝ => recommend p.1 p.2)

theorem proper_input_measurableSet :
    MeasurableSet {p : (Precisions A × Means A) × ℝ | ∀ a, 0 < p.1.1 a} := by
  simp only [Set.setOf_forall]
  exact MeasurableSet.iInter (fun a => measurableSet_lt measurable_const
    ((measurable_pi_apply a).comp (measurable_fst.comp measurable_fst)))

/-- Extend only for interoperability with the globally defined interpreter.
On the unused improper states the policy chooses one fixed arm. -/
def extend (π : Policy A) : RealPrecisionPolicy.Policy A where
  draw := fun k c m u => if hc : ∀ a, 0 < c a then π.draw k ⟨(c, m), hc⟩ u
    else Classical.choice (inferInstance : Nonempty A)
  recommend := fun c m u => if hc : ∀ a, 0 < c a then π.recommend ⟨(c, m), hc⟩ u
    else Classical.choice (inferInstance : Nonempty A)
  draw_measurable := by
    intro k
    let S : Set ((Precisions A × Means A) × ℝ) := {p | ∀ a, 0 < p.1.1 a}
    have hf : Measurable (fun p : S => π.draw k ⟨p.1.1, p.2⟩ p.1.2) := by
      exact (π.draw_measurable k).comp
        (((measurable_fst.comp measurable_subtype_coe).subtype_mk).prodMk
          (measurable_snd.comp measurable_subtype_coe))
    exact hf.dite measurable_const proper_input_measurableSet
  recommend_measurable := by
    let S : Set ((Precisions A × Means A) × ℝ) := {p | ∀ a, 0 < p.1.1 a}
    have hf : Measurable (fun p : S => π.recommend ⟨p.1.1, p.2⟩ p.1.2) := by
      exact π.recommend_measurable.comp
        (((measurable_fst.comp measurable_subtype_coe).subtype_mk).prodMk
          (measurable_snd.comp measurable_subtype_coe))
    exact hf.dite measurable_const proper_input_measurableSet

theorem extend_draw (π : Policy A) (k : ℕ) (s : State A) (u : ℝ) :
    (extend π).draw k s.1.1 s.1.2 u = π.draw k s u := by
  rcases s with ⟨⟨c, m⟩, hc⟩
  simp only [extend, dif_pos hc]

theorem extend_recommend (π : Policy A) (s : State A) (u : ℝ) :
    (extend π).recommend s.1.1 s.1.2 u = π.recommend s u := by
  rcases s with ⟨⟨c, m⟩, hc⟩
  simp only [extend, dif_pos hc]

/-- The actual observed-reward conjugate update remains inside the proper
domain, so an execution never invokes the arbitrary extension branch. -/
def update (s : State A) (a : A) (y : ℝ) : State A :=
  ⟨GeneralOptimality.updateState a (s.1, y), advance_positive s.1.1 a s.2⟩

/-- Direct execution of the policy on its original proper domain. This
definition does not refer to the extension. -/
def rawTree (π : Policy A) : (H : ℕ) → State A → Seeds H → Strategy (A := A) H
  | 0, s, u => .stop (π.recommend s (u ⟨0, by omega⟩))
  | H + 1, s, u =>
      let a := π.draw H s (u ⟨0, by omega⟩)
      .sample a (fun y => rawTree π H (update s a y) (seedTail u))

/-- Pointwise equality of the independently defined direct and extended
executions, for every reward history and every seed array. -/
theorem rawTree_extend (π : Policy A) (H : ℕ) (s : State A) (u : Seeds H) :
    RealPrecisionPolicy.rawTree (extend π) H s.1.1 s.1.2 u = rawTree π H s u := by
  induction H generalizing s with
  | zero => simp only [RealPrecisionPolicy.rawTree, rawTree, extend_recommend]
  | succ H ih =>
    simp only [RealPrecisionPolicy.rawTree, rawTree, extend_draw]
    congr 1
    funext y
    exact ih (update s (π.draw H s (u ⟨0, by omega⟩)) y) (seedTail u)

/-- Actual posterior expected regret of the direct proper-domain policy,
integrated over the latent Gaussian parameter, observations, and fresh seeds. -/
def actualRisk (π : Policy A) (H : ℕ) (s : State A) : ℝ :=
  ∫ u, rawRisk (rawTree π H s u) s.1.1 s.1.2 ∂Junpei.Policy.seedLaw (H + 1)

theorem actualRisk_extend (π : Policy A) (H : ℕ) (s : State A) :
    RealPrecisionPolicy.actualRisk (extend π) H s.1.1 s.1.2 = actualRisk π H s := by
  simp only [RealPrecisionPolicy.actualRisk, actualRisk, rawTree_extend]

/-- Single-state minimization against the concrete continuation policies of
Definition 3, using the directly defined proper-domain execution. -/
def MinimizesHistoryRisk (π : Policy A) (H : ℕ) (s : State A) : Prop :=
  ∀ ρ : HistoryPolicy.Policy A,
    actualRisk π H s ≤ HistoryPolicy.actualRisk ρ 0 H GeneralOptimality.emptyHistory s.1.1 s.1.2

theorem minimizing_extend_iff (π : Policy A) (H : ℕ) (s : State A) :
    GeneralOptimality.MinimizesHistoryRisk (extend π) H s.1.1 s.1.2 ↔
      MinimizesHistoryRisk π H s := by
  simp only [GeneralOptimality.MinimizesHistoryRisk, MinimizesHistoryRisk, actualRisk_extend]

/-- Fixed-budget sequential posterior optimality on exactly the paper's
proper-state domain, requiring only the remaining horizons 0 through N. -/
def SequentiallyOptimalUpTo (π : Policy A) (N : ℕ) : Prop :=
  ∀ H, H ≤ N → ∀ s : State A, MinimizesHistoryRisk π H s

theorem sequentiallyOptimalUpTo_extend (π : Policy A) (N : ℕ)
    (hπ : SequentiallyOptimalUpTo π N) :
    GeneralOptimality.SequentiallyOptimalUpTo (extend π) N := by
  intro H hH c m hc
  exact (minimizing_extend_iff π H ⟨(c, m), hc⟩).mpr (hπ H hH ⟨(c, m), hc⟩)

section Ordered
variable [LinearOrder A]

/-- Theorem 1 for ANY Borel posterior-state policy on the exact proper
Gaussian domain in the manuscript. All three conclusions use its direct
actual-observation execution; extending to improper states is an internal
proved construction, not an extra hypothesis on the paper's policy. -/
theorem theorem1 (π : Policy A) (N : ℕ) (hπ : SequentiallyOptimalUpTo π N) :
    (∀ H, H ≤ N → ∀ s : State A, actualRisk π H s = loss H s.1.1 s.1.2) ∧
    (∀ H, H < N → ∀ s : State A,
      ∀ᵐ u : ℝ ∂gaussianReal 0 1,
        ∀ a, BayesOptimal.EBI H s.1.1 s.1.2 a ≤
          BayesOptimal.EBI H s.1.1 s.1.2 (π.draw H s u)) ∧
    (∀ s : State A, ∀ᵐ u : ℝ ∂gaussianReal 0 1,
      s.1.2 (π.recommend s u) = maxAll s.1.2) := by
  obtain ⟨hr, hd, ht⟩ := GeneralOptimality.theorem1 (extend π) N
    (sequentiallyOptimalUpTo_extend π N hπ)
  refine ⟨?_, ?_, ?_⟩
  · intro H hH s
    simpa only [actualRisk_extend] using hr H hH s.1.1 s.1.2 s.2
  · intro H hH s
    simpa only [extend_draw] using hd H hH s.1.1 s.1.2 s.2
  · intro s
    simpa only [extend_recommend] using ht s.1.1 s.1.2 s.2

end Ordered
end
end Junpei.GeneralOptimality.ProperDomain
