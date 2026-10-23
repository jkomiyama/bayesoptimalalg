import Junpei.FixedGaussianInitialReduction
import Junpei.FixedGaussianStackExecution

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory

namespace Junpei.FixedGaussian

open Draft0.BayesianModel Junpei.Policy Junpei.PolicyContinuation Junpei.SequentialRisk
open Junpei.CountModelBridge Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open Junpei.InitialOptimal

noncomputable section

def initializationLaw (μ : Means) : Measure Means :=
  Measure.pi (fun a : Arm => gaussianReal (μ a) 1)

instance initializationLaw_probability (μ : Means) :
    IsProbabilityMeasure (initializationLaw μ) := by unfold initializationLaw; infer_instance

def initializedLaw (H : ℕ) (μ : Means) : Measure (Input H) :=
  (initializationLaw μ).prod ((seedLaw (H + 1)).prod (fixedLaw H μ))

instance initializedLaw_probability (H : ℕ) (μ : Means) :
    IsProbabilityMeasure (initializedLaw H μ) := by unfold initializedLaw; infer_instance

def initializedRun (π : StatePolicy) (H : ℕ) (p : Input H) : Arm :=
  run (rawTree π 4 H (fun _ => 1) p.1 p.2.1) p.2.2

def initializedStackRun (π : StatePolicy) (H : ℕ) (p : Input H) : Arm :=
  stackRun (rawTree π 4 H (fun _ => 1) p.1 p.2.1) p.2.2

theorem initializedRun_measurable (π : StatePolicy) (H : ℕ) :
    Measurable (initializedRun π H) := rawTree_run_measurable π 4 H _

theorem initializedStackRun_measurable (π : StatePolicy) (H : ℕ) :
    Measurable (initializedStackRun π H) := rawTree_stackRun_measurable π 4 H _

theorem seeded_stack_law_eq (π : StatePolicy) (t H : ℕ) (c : Counts) (m μ : Means) :
    ((seedLaw (H + 1)).prod (fixedLaw H μ)).map
        (fun p => stackRun (rawTree π t H c m p.1) p.2) =
      ((seedLaw (H + 1)).prod (fixedLaw H μ)).map
        (fun p => run (rawTree π t H c m p.1) p.2) := by
  have hs : Measurable (fun p : Seeds H × RawStream Arm H =>
      stackRun (rawTree π t H c m p.1) p.2) :=
    (rawTree_stackRun_measurable π t H c).comp (measurable_const.prodMk measurable_id)
  have hc : Measurable (fun p : Seeds H × RawStream Arm H =>
      run (rawTree π t H c m p.1) p.2) :=
    (rawTree_run_measurable π t H c).comp (measurable_const.prodMk measurable_id)
  apply Measure.ext
  intro B hB
  rw [Measure.map_apply hs hB, Measure.map_apply hc hB,
    Measure.prod_apply (hs hB), Measure.prod_apply (hc hB)]
  apply lintegral_congr
  intro u
  have hu := congrArg (fun ν : Measure Arm => ν B) (rawTree_stack_law_eq π t H c m μ u)
  simpa only [Measure.map_apply (stackRegular_run_measurable _
    (rawTree_stackRegular π t H c m u)) hB,
    Measure.map_apply (regular_run_measurable _ (rawTree_regular π t H c m u)) hB] using hu

theorem initialized_stack_law_eq (π : StatePolicy) (H : ℕ) (μ : Means) :
    (initializedLaw H μ).map (initializedStackRun π H) =
      (initializedLaw H μ).map (initializedRun π H) := by
  apply Measure.ext
  intro B hB
  rw [Measure.map_apply (initializedStackRun_measurable π H) hB,
    Measure.map_apply (initializedRun_measurable π H) hB]
  unfold initializedLaw
  rw [Measure.prod_apply ((initializedStackRun_measurable π H) hB),
    Measure.prod_apply ((initializedRun_measurable π H) hB)]
  apply lintegral_congr
  intro m
  have hm := congrArg (fun ν : Measure Arm => ν B)
    (seeded_stack_law_eq π 4 H (fun _ => 1) m μ)
  have hs : Measurable (fun p : Seeds H × RawStream Arm H =>
      stackRun (rawTree π 4 H (fun _ => 1) m p.1) p.2) :=
    (rawTree_stackRun_measurable π 4 H (fun _ => 1)).comp
    (measurable_const.prodMk measurable_id)
  have hc : Measurable (fun p : Seeds H × RawStream Arm H =>
      run (rawTree π 4 H (fun _ => 1) m p.1) p.2) :=
    (rawTree_run_measurable π 4 H (fun _ => 1)).comp
    (measurable_const.prodMk measurable_id)
  simpa only [Measure.map_apply hs hB, Measure.map_apply hc hB] using hm

theorem initialRiskOptimal_initialized_repair_ae (π : StatePolicy) (T : ℕ)
    (hT : 3 ≤ T) (hπ : InitialRiskOptimal π T) (μ : Means) :
    initializedRun π (T - 3) =ᵐ[initializedLaw (T - 3) μ]
      initializedRun (repair π T) (T - 3) := by
  apply (Measure.ae_prod_iff_ae_ae (measurableSet_eq_fun_of_countable
    (initializedRun_measurable π (T - 3))
    (initializedRun_measurable (repair π T) (T - 3)))).mpr
  exact Filter.Eventually.of_forall (fun m => initialRiskOptimal_repair_fixed_ae π T hT hπ m μ)

/-- The complete initialized experiment, with fresh Gaussian initialization,
fresh Gaussian policy seeds, and independent per-pull reward stacks. -/
theorem initialRiskOptimal_initialized_stack_repair_law (π : StatePolicy) (T : ℕ)
    (hT : 3 ≤ T) (hπ : InitialRiskOptimal π T) (μ : Means) :
    (initializedLaw (T - 3) μ).map (initializedStackRun π (T - 3)) =
      (initializedLaw (T - 3) μ).map (initializedStackRun (repair π T) (T - 3)) := by
  rw [initialized_stack_law_eq, initialized_stack_law_eq]
  exact Measure.map_congr (initialRiskOptimal_initialized_repair_ae π T hT hπ μ)

end
end Junpei.FixedGaussian
