import Junpei.FixedGaussianAbsoluteContinuous

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FixedGaussian

open FiniteGaussian FinitePosterior EmbeddingRaw EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

/-- All calendar-indexed observations under a fixed true mean vector.  Reading
only the coordinates selected by an adaptive policy is a measurable map. -/
def fixedLaw (H : ℕ) (μ : Means A) : Measure (RawStream A H) :=
  Measure.pi (fun a : A => Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1))

instance fixedLaw_probability (H : ℕ) (μ : Means A) :
    IsProbabilityMeasure (fixedLaw H μ) := by unfold fixedLaw; infer_instance

/-- The actual predictive reward array, including its within-arm correlation
through the shared Gaussian latent mean. -/
def predictiveLaw (H : ℕ) (c : Precisions A) (m : Means A) : Measure (RawStream A H) :=
  (worldLaw H c m).map observations

instance predictiveLaw_probability (H : ℕ) (c : Precisions A) (m : Means A) :
    IsProbabilityMeasure (predictiveLaw H c m) :=
  Measure.isProbabilityMeasure_map (observations_measurable H).aemeasurable

theorem shift_noise_preserving (H : ℕ) (μ : Means A) :
    MeasurePreserving (fun e : RawStream A H => fun a t => μ a + e a t)
      (arrayLaw H) (fixedLaw H μ) := by
  unfold arrayLaw fixedLaw
  apply measurePreserving_pi (fun _ : A => MockStream.noiseLaw H)
    (fun a : A => Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1))
    (f := fun a e t => μ a + e t)
  intro a
  unfold MockStream.noiseLaw
  apply measurePreserving_pi (fun _ : Fin H => gaussianReal 0 1)
    (fun _ : Fin H => gaussianReal (μ a) 1) (f := fun _ e => μ a + e)
  intro t
  refine ⟨by fun_prop, ?_⟩
  simpa only [zero_add] using gaussianReal_map_const_add (μ := 0) (v := 1) (μ a)

theorem fixed_equivalent (H : ℕ) (μ ν : Means A) :
    fixedLaw H μ ≪ fixedLaw H ν ∧ fixedLaw H ν ≪ fixedLaw H μ := by
  constructor
  · apply finite_pi_absolutelyContinuous
    intro a
    apply finite_pi_absolutelyContinuous
    intro t
    exact (gaussian_unit_equivalent (μ a) (ν a)).1
  · apply finite_pi_absolutelyContinuous
    intro a
    apply finite_pi_absolutelyContinuous
    intro t
    exact (gaussian_unit_equivalent (μ a) (ν a)).2

theorem predictive_absolutelyContinuous_fixed (H : ℕ)
    (c : Precisions A) (m μ : Means A) : predictiveLaw H c m ≪ fixedLaw H μ := by
  apply Measure.AbsolutelyContinuous.mk
  intro s hs hzero
  rw [predictiveLaw, Measure.map_apply (observations_measurable H) hs]
  unfold worldLaw
  apply Measure.measure_prod_null_of_ae_null ((observations_measurable H) hs)
  apply Filter.Eventually.of_forall
  intro θ
  have hz := (fixed_equivalent H θ μ).1 hzero
  rw [← (shift_noise_preserving H θ).map_eq,
    Measure.map_apply (shift_noise_preserving H θ).measurable hs] at hz
  exact hz

theorem fixed_absolutelyContinuous_predictive (H : ℕ)
    (c : Precisions A) (m μ : Means A) : fixedLaw H μ ≪ predictiveLaw H c m := by
  apply Measure.AbsolutelyContinuous.mk
  intro s hs hzero
  rw [predictiveLaw, Measure.map_apply (observations_measurable H) hs] at hzero
  change (posteriorLaw c m).prod (arrayLaw H) (observations ⁻¹' s) = 0 at hzero
  have hsections := Measure.measure_ae_null_of_prod_null hzero
  obtain ⟨θ, hθ⟩ := hsections.exists
  have hz : fixedLaw H θ s = 0 := by
    rw [← (shift_noise_preserving H θ).map_eq,
      Measure.map_apply (shift_noise_preserving H θ).measurable hs]
    exact hθ
  exact (fixed_equivalent H μ θ).1 hz

theorem fixed_predictive_equivalent (H : ℕ) (c : Precisions A) (m μ : Means A) :
    fixedLaw H μ ≪ predictiveLaw H c m ∧ predictiveLaw H c m ≪ fixedLaw H μ :=
  ⟨fixed_absolutelyContinuous_predictive H c m μ,
    predictive_absolutelyContinuous_fixed H c m μ⟩

end
end Junpei.FixedGaussian
