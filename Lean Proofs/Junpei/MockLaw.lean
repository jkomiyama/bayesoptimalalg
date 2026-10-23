import Junpei.MockMeasure
import Junpei.EmbeddingExecution

set_option linter.unusedSectionVars false
set_option synthInstance.maxHeartbeats 400000
set_option maxHeartbeats 1000000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockLaw

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.MockStream
open Junpei.EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

def realNoise (H : ℕ) (i : A) (w : World A H i) : MockStream.RawStream A H :=
  MockMeasure.join i (w.1.1.2, w.2.2)

theorem realData_preserving (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    MeasurePreserving (fun p : Block H × OtherData A H i =>
      (m + FinitePosterior.join i (p.1.1, p.2.1), MockMeasure.join i (p.1.2, p.2.2)))
      ((blockLaw H (c i)⁻¹).prod (otherDataLaw H c i)) (worldLaw H c m) := by
  have hc := MockMeasure.cross_preserving (gaussianReal 0 (c i)⁻¹) (noiseLaw H)
    (otherLaw c i) (Measure.pi (fun _ : {a : A // a ≠ i} => noiseLaw H))
  have hj : MeasurePreserving (FinitePosterior.join i)
      ((gaussianReal 0 (c i)⁻¹).prod (otherLaw c i)) (centeredLaw c) :=
    ⟨(join_continuous i).measurable, join_law c i⟩
  have ht : MeasurePreserving (fun x : Means A => m + x) (centeredLaw c)
      (posteriorLaw c m) := ⟨by fun_prop, rfl⟩
  have hn := MockMeasure.join_preserving (fun _ : A => noiseLaw H) i
  exact ((ht.comp hj).prod hn).comp hc

/-- Forgetting the independent copy gives exactly the usual latent Gaussian
posterior times independent observation noise, rather than an assumed law. -/
theorem realWorld_preserving (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    MeasurePreserving (fun w : World A H i => (theta H m i w, realNoise H i w))
      (law H c i) (worldLaw H c m) := by
  have hp := (measurePreserving_fst
    (μ := (blockLaw H (c i)⁻¹).prod (otherDataLaw H c i)) (ν := blockLaw H (c i)⁻¹)).comp
    (isolate_copy_preserving H c i)
  exact (realData_preserving H c m i).comp hp

theorem real_observations (H : ℕ) (m : Means A) (i : A) (w : World A H i) :
    observations (theta H m i w, realNoise H i w) = realStream H m i w := by
  funext a t
  by_cases h : a = i <;>
    simp [observations, theta, realNoise, MockMeasure.join, FinitePosterior.join, realStream, h]

theorem tailNoise_preserving (H : ℕ) :
    MeasurePreserving (EmbeddingRaw.tail (A := A) (H := H)) (arrayLaw (A := A) (H + 1))
      (arrayLaw (A := A) H) := by
  exact (measurePreserving_snd
    (μ := Measure.pi (fun _ : A => gaussianReal 0 1)) (ν := arrayLaw (A := A) H)).comp
    (splitNoise_preserving H)

/-- Auxiliary posterior-predictive data are independent of the real latent
parameters and all H actual future observation rounds. -/
theorem seed_and_real_preserving (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    MeasurePreserving (fun w : World A (H + 1) i =>
      (w.1.2, (theta (H + 1) m i w, EmbeddingRaw.tail (realNoise (H + 1) i w))))
      (law (H + 1) c i) ((blockLaw (H + 1) (c i)⁻¹).prod (worldLaw H c m)) := by
  letI : IsProbabilityMeasure (blockLaw (H + 1) (c i)⁻¹) := blockLaw_probability _ _
  letI : IsProbabilityMeasure (otherDataLaw (H + 1) c i) := otherDataLaw_probability _ _ _
  have hs : MeasurePreserving (Prod.swap :
      (Block (H + 1) × OtherData A (H + 1) i) × Block (H + 1) →
        Block (H + 1) × (Block (H + 1) × OtherData A (H + 1) i))
      (((blockLaw (H + 1) (c i)⁻¹).prod (otherDataLaw (H + 1) c i)).prod
        (blockLaw (H + 1) (c i)⁻¹))
      ((blockLaw (H + 1) (c i)⁻¹).prod
        ((blockLaw (H + 1) (c i)⁻¹).prod (otherDataLaw (H + 1) c i))) :=
    Measure.measurePreserving_swap
    (μ := (blockLaw (H + 1) (c i)⁻¹).prod (otherDataLaw (H + 1) c i))
    (ν := blockLaw (H + 1) (c i)⁻¹)
  have hp := hs.comp (isolate_copy_preserving (H + 1) c i)
  have ht := (MeasurePreserving.id (posteriorLaw c m)).prod (tailNoise_preserving (A := A) H)
  exact ((MeasurePreserving.id (blockLaw (H + 1) (c i)⁻¹)).prod
    (ht.comp (realData_preserving (H + 1) c m i))).comp hp

end
end Junpei.MockLaw
