import Junpei.FixedGaussianLaws

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FixedGaussian

open FiniteGaussian FinitePosterior EmbeddingRaw EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

/-- Independent randomization preserves finite-horizon equivalence.  The same
seed law is used under the predictive model and every fixed true mean. -/
theorem seeded_fixed_predictive_equivalent {S : Type*} [MeasurableSpace S]
    (σ : Measure S) (H : ℕ) (c : Precisions A) (m μ : Means A) :
    σ.prod (fixedLaw H μ) ≪ σ.prod (predictiveLaw H c m) ∧
      σ.prod (predictiveLaw H c m) ≪ σ.prod (fixedLaw H μ) := by
  constructor
  · exact Measure.AbsolutelyContinuous.rfl.prod (fixed_predictive_equivalent H c m μ).1
  · exact Measure.AbsolutelyContinuous.rfl.prod (fixed_predictive_equivalent H c m μ).2

/-- In particular, any measurable adaptive transcript or recommendation map
has the same null events under the two models.  The map is literally the same
function, so it cannot depend on the unknown true mean. -/
theorem adaptive_map_equivalent {S B : Type*} [MeasurableSpace S] [MeasurableSpace B]
    (σ : Measure S) (H : ℕ) (c : Precisions A) (m μ : Means A)
    (f : S × RawStream A H → B) (hf : Measurable f) :
    (σ.prod (fixedLaw H μ)).map f ≪ (σ.prod (predictiveLaw H c m)).map f ∧
      (σ.prod (predictiveLaw H c m)).map f ≪ (σ.prod (fixedLaw H μ)).map f := by
  exact ⟨((seeded_fixed_predictive_equivalent σ H c m μ).1).map hf,
    ((seeded_fixed_predictive_equivalent σ H c m μ).2).map hf⟩

theorem seeded_world_observations_preserving {S : Type*} [MeasurableSpace S]
    (σ : Measure S) [SFinite σ] (H : ℕ) (c : Precisions A) (m : Means A) :
    MeasurePreserving (fun p : S × (Means A × RawStream A H) =>
      (p.1, observations p.2)) (σ.prod (worldLaw H c m))
        (σ.prod (predictiveLaw H c m)) := by
  exact (MeasurePreserving.id σ).prod ⟨observations_measurable H, rfl⟩

/-- A predictive almost-everywhere statement about the actual execution
transfers to every fixed-parameter experiment. -/
theorem ae_execution_transfer {S : Type*} [MeasurableSpace S]
    (σ : Measure S) [SFinite σ] (H : ℕ) (c : Precisions A) (m μ : Means A)
    (P : S × RawStream A H → Prop) (hP : MeasurableSet {p | P p})
    (h : ∀ᵐ p ∂σ.prod (worldLaw H c m), P (p.1, observations p.2)) :
    ∀ᵐ p ∂σ.prod (fixedLaw H μ), P p := by
  have hp := seeded_world_observations_preserving σ H c m
  have he : ∀ᵐ p ∂σ.prod (predictiveLaw H c m), P p := by
    rw [← hp.map_eq]
    exact (ae_map_iff hp.measurable.aemeasurable hP).2 h
  exact (seeded_fixed_predictive_equivalent σ H c m μ).1.ae_le he

/-- Null-history modification therefore preserves the full recommendation
law for every fixed true mean, once predictive execution agreement is proved. -/
theorem recommendation_law_eq {S B : Type*} [MeasurableSpace S] [MeasurableSpace B]
    [Countable B] [MeasurableSingletonClass B]
    (σ : Measure S) [SFinite σ] (H : ℕ) (c : Precisions A) (m μ : Means A)
    (f g : S × RawStream A H → B) (hf : Measurable f) (hg : Measurable g)
    (h : ∀ᵐ p ∂σ.prod (worldLaw H c m),
      f (p.1, observations p.2) = g (p.1, observations p.2)) :
    (σ.prod (fixedLaw H μ)).map f = (σ.prod (fixedLaw H μ)).map g := by
  exact Measure.map_congr (ae_execution_transfer σ H c m μ
    (fun p => f p = g p) (measurableSet_eq_fun_of_countable hf hg) h)

/-- Equality is stronger than equality of regret: every integrable measurable
payoff of the recommendation agrees, including fixed-parameter simple regret. -/
theorem recommendation_payoff_integral_eq {S B : Type*} [MeasurableSpace S]
    [MeasurableSpace B] [Countable B] [MeasurableSingletonClass B]
    (σ : Measure S) [SFinite σ] (H : ℕ) (c : Precisions A) (m μ : Means A)
    (f g : S × RawStream A H → B) (hf : Measurable f) (hg : Measurable g)
    (h : ∀ᵐ p ∂σ.prod (worldLaw H c m),
      f (p.1, observations p.2) = g (p.1, observations p.2)) (L : B → ℝ) :
    (∫ p, L (f p) ∂σ.prod (fixedLaw H μ)) =
      ∫ p, L (g p) ∂σ.prod (fixedLaw H μ) := by
  apply integral_congr_ae
  exact (ae_execution_transfer σ H c m μ
    (fun p => f p = g p) (measurableSet_eq_fun_of_countable hf hg) h).mono
      (fun _ hp => congrArg L hp)

end
end Junpei.FixedGaussian
