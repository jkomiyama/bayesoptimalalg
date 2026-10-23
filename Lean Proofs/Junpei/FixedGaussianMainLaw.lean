import Junpei.FixedGaussianInitialized
import Junpei.FixedGaussianWindows
import Junpei.FixedGaussianStackPath

set_option linter.unusedSectionVars false
set_option maxHeartbeats 800000

open MeasureTheory ProbabilityTheory

namespace Junpei.FixedGaussian

open Draft0.BayesianModel Junpei.Policy Junpei.PolicyContinuation
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution Junpei.Empirical
open Junpei.InitialOptimal

noncomputable section

theorem initialized_rewards_preserving (T : ℕ) (hT : 3 ≤ T) (μ : Means) :
    MeasurePreserving (fun r : RawStream Arm T =>
      (fun a => r a ⟨0, by omega⟩,
        fun a (j : Fin (T - 3)) => r a ⟨j.val + 1, by have := j.isLt; omega⟩))
      (fixedLaw T μ) ((initializationLaw μ).prod (fixedLaw (T - 3) μ)) := by
  have hp := fixed_prefix_preserving T (T - 3 + 1) (by omega) μ
  have hs := all_heads_split_preserving (T - 3) μ
  convert hs.comp hp using 1

theorem raw_initialized_rewards_preserving (T : ℕ) (Δ : ℝ) (hT : 3 ≤ T) :
    MeasurePreserving (fun r : Draft0.BadEvent.RawArrays T =>
      (StackPath.initialMeans T Δ hT r, StackPath.remainingStack T Δ hT r))
      (Draft0.BadEvent.rawArraysLaw T)
      ((initializationLaw (trueMean Δ)).prod (fixedLaw (T - 3) (trueMean Δ))) := by
  have hp := (initialized_rewards_preserving T hT (trueMean Δ)).comp
    (shift_noise_preserving T (trueMean Δ))
  convert hp using 1
  funext r
  apply Prod.ext
  · funext a
    simp [StackPath.initialMeans, observation, add_comm]
  · funext a j
    simp [StackPath.remainingStack, observation, add_comm]

theorem seedWindow_preserving (T : ℕ) (hT : 3 ≤ T) :
    MeasurePreserving (StackPath.seedWindow T hT) (randomizerLaw T)
      (seedLaw (T - 3 + 1)) := by
  apply pi_restrict_preserving (fun _ : Fin (T + 1) => gaussianReal 0 1)
    (fun j : Fin (T - 3 + 1) => (⟨j.val + 3, by have := j.isLt; omega⟩ : Fin (T + 1)))
  intro i j h
  apply Fin.ext
  have hi := congrArg (fun z : Fin (T + 1) => z.val) h
  change i.val + 3 = j.val + 3 at hi
  exact Nat.add_right_cancel hi

/-- The exact finite reward and seed arrays used by Main, projected to the
actual initialization and unused reward/seed windows, have the initialized
execution law proved above. -/
theorem initializedInput_preserving (T : ℕ) (Δ : ℝ) (hT : 3 ≤ T) :
    MeasurePreserving (StackPath.initializedInput T Δ hT) (Junpei.Main.jointLaw T)
      (initializedLaw (T - 3) (trueMean Δ)) := by
  have hp := (raw_initialized_rewards_preserving T Δ hT).prod (seedWindow_preserving T hT)
  have ha := measurePreserving_prodAssoc (initializationLaw (trueMean Δ))
    (fixedLaw (T - 3) (trueMean Δ)) (seedLaw (T - 3 + 1))
  have hs := (MeasurePreserving.id (initializationLaw (trueMean Δ))).prod
    (Measure.measurePreserving_swap (μ := fixedLaw (T - 3) (trueMean Δ))
      (ν := seedLaw (T - 3 + 1)))
  convert hs.comp (ha.comp hp) using 1

/-- The legacy per-pull-stack main theorem and the actual initialized raw
strategy have precisely the same recommendation distribution. -/
theorem actualRecommendation_law (π : StatePolicy) (T : ℕ) (Δ : ℝ) (hT : 3 ≤ T) :
    (Junpei.Main.jointLaw T).map (Junpei.Main.actualRecommendation π T Δ) =
      (initializedLaw (T - 3) (trueMean Δ)).map (initializedStackRun π (T - 3)) := by
  have hp := initializedInput_preserving T Δ hT
  calc
    _ = (Junpei.Main.jointLaw T).map
        (initializedStackRun π (T - 3) ∘ StackPath.initializedInput T Δ hT) := by
      congr 1
      funext r
      exact StackPath.actualRecommendation_eq_stackRun π T hT Δ r
    _ = ((Junpei.Main.jointLaw T).map (StackPath.initializedInput T Δ hT)).map
        (initializedStackRun π (T - 3)) :=
      (Measure.map_map (initializedStackRun_measurable π (T - 3)) hp.measurable).symm
    _ = _ := by rw [hp.map_eq]

theorem initialRiskOptimal_actualRecommendation_repair_law (π : StatePolicy) (T : ℕ)
    (Δ : ℝ) (hT : 3 ≤ T) (hπ : InitialRiskOptimal π T) :
    (Junpei.Main.jointLaw T).map (Junpei.Main.actualRecommendation π T Δ) =
      (Junpei.Main.jointLaw T).map (Junpei.Main.actualRecommendation (repair π T) T Δ) := by
  rw [actualRecommendation_law π T Δ hT, actualRecommendation_law (repair π T) T Δ hT]
  exact initialRiskOptimal_initialized_stack_repair_law π T hT hπ (trueMean Δ)

/-- The previously missing initialization-only reduction, for exactly the
frequentist regret definition occurring in the existing Main theorem. -/
theorem initialRiskOptimal_regret_repair_eq (π : StatePolicy) (T : ℕ) (Δ : ℝ)
    (hT : 3 ≤ T) (hπ : InitialRiskOptimal π T) :
    Junpei.Main.regret π T Δ = Junpei.Main.regret (repair π T) T Δ := by
  let L : Arm → ℝ := fun a => Δ - trueMean Δ a
  have hL : Measurable L := measurable_of_countable L
  unfold Junpei.Main.regret
  rw [← integral_map (Junpei.Main.actualRecommendation_measurable π T Δ).aemeasurable
    hL.aestronglyMeasurable,
    initialRiskOptimal_actualRecommendation_repair_law π T Δ hT hπ,
    integral_map (Junpei.Main.actualRecommendation_measurable (repair π T) T Δ).aemeasurable
      hL.aestronglyMeasurable]

end
end Junpei.FixedGaussian
