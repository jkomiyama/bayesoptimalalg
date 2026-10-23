import Junpei.FixedGaussianMain
import Junpei.FlatPriorProduct

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FixedGaussian.FlatInitialization

open Junpei.FiniteGaussian Junpei.FinitePosterior
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution Junpei.EmbeddingRandomized

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

def dataMeans (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) : Means A :=
  fun a => Junpei.FlatPriorInitialization.empiricalMean (y a)

/-- Normalize the likelihood against the improper flat Lebesgue prior after
the mandatory observations. This is a genuine measure, not a stipulated
Gaussian posterior. Positive sample counts make its normalizer finite and
strictly positive, as proved by FlatPriorProduct. -/
def normalizedFlatLaw (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) : Measure (Means A) :=
  volume.withDensity (fun θ => ENNReal.ofReal
    (Junpei.FlatPriorProduct.productLikelihood n y θ /
      (∫ x, Junpei.FlatPriorProduct.productLikelihood n y x)))

theorem normalizedFlatLaw_eq_posterior (n : A → ℕ)
    (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a) :
    normalizedFlatLaw n y = posteriorLaw (fun a => (n a : ℝ≥0)) (dataMeans n y) :=
  Junpei.FlatPriorProduct.normalized_product_measure n y hn

theorem normalizedFlatLaw_probability (n : A → ℕ)
    (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a) :
    IsProbabilityMeasure (normalizedFlatLaw n y) := by
  rw [normalizedFlatLaw_eq_posterior n y hn]
  infer_instance

/-- The actual next raw Gaussian observation from the normalized flat-prior
posterior has the joint innovation/updated-posterior law used by the proof.
Thus the one-step bridge starts from the likelihood normalization itself. -/
theorem normalizedFlat_observation_joint_law (n : A → ℕ)
    (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a) (a : A) :
    ((normalizedFlatLaw n y).prod (gaussianReal 0 1)).map
        (Junpei.ObservationBridge.rawObservation a) =
      ((gaussianReal 0 (innovation (fun b => (n b : ℝ≥0)) a)).prod
        (centeredLaw (advance (fun b => (n b : ℝ≥0)) a))).map
          (Junpei.ObservationBridge.reconstructedObservation
            (fun b => (n b : ℝ≥0)) (dataMeans n y) a) := by
  rw [normalizedFlatLaw_eq_posterior n y hn]
  exact Junpei.ObservationBridge.observation_joint_law _ _ a (by exact_mod_cast hn a)

/-- The latent mean is sampled from the normalized flat likelihood, and all
future observation noise coordinates are independent unit Gaussians. -/
def flatWorldLaw (H : ℕ) (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) :
    Measure (Means A × RawStream A H) := (normalizedFlatLaw n y).prod (arrayLaw H)

theorem flatWorldLaw_eq_worldLaw (H : ℕ) (n : A → ℕ)
    (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a) :
    flatWorldLaw H n y = worldLaw H (fun a => (n a : ℝ≥0)) (dataMeans n y) := by
  unfold flatWorldLaw worldLaw
  rw [normalizedFlatLaw_eq_posterior n y hn]

section Measurable
variable [MeasurableSpace A] [MeasurableSingletonClass A]

theorem flat_rawPayoff_integrable {H : ℕ} (s : Strategy (A := A) H)
    (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a)
    (hs : Measurable (run s)) :
    Integrable (fun w : Means A × RawStream A H =>
      maxAll w.1 - w.1 (run s (observations w))) (flatWorldLaw H n y) := by
  rw [flatWorldLaw_eq_worldLaw H n y hn]
  exact rawRisk_integrable s _ _ hs

theorem flat_rawRisk_eq {H : ℕ} (s : Strategy (A := A) H)
    (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a) :
    (∫ w : Means A × RawStream A H,
      maxAll w.1 - w.1 (run s (observations w)) ∂flatWorldLaw H n y) =
        rawRisk s (fun a => (n a : ℝ≥0)) (dataMeans n y) := by
  rw [flatWorldLaw_eq_worldLaw H n y hn]
  rfl

variable {S : Type*} [MeasurableSpace S]

theorem flat_randomizedPayoff_integrable {H : ℕ}
    (σ : Measure S) [IsProbabilityMeasure σ] (s : S → Strategy (A := A) H)
    (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a)
    (hs : Measurable (fun p : S × RawStream A H => run (s p.1) p.2)) :
    Integrable (randomizedPayoff s) (σ.prod (flatWorldLaw H n y)) := by
  rw [flatWorldLaw_eq_worldLaw H n y hn]
  exact randomizedPayoff_integrable σ s _ _ hs

/-- Actual expected simple regret, including arbitrary independent policy
randomization, equals the risk used in the Gaussian Bellman development. -/
theorem flat_randomizedRisk_eq {H : ℕ}
    (σ : Measure S) [IsProbabilityMeasure σ] (s : S → Strategy (A := A) H)
    (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a)
    (hs : Measurable (fun p : S × RawStream A H => run (s p.1) p.2)) :
    (∫ p, randomizedPayoff s p ∂σ.prod (flatWorldLaw H n y)) =
      ∫ u, rawRisk (s u) (fun a => (n a : ℝ≥0)) (dataMeans n y) ∂σ := by
  rw [flatWorldLaw_eq_worldLaw H n y hn]
  exact integral_prod _ (randomizedPayoff_integrable σ s _ _ hs)

theorem flat_randomizedRisk_ge_loss {H : ℕ}
    (σ : Measure S) [IsProbabilityMeasure σ] (s : S → Strategy (A := A) H)
    (n : A → ℕ) (y : (a : A) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a)
    (hregular : ∀ u, Regular (s u))
    (hs : Measurable (fun p : S × RawStream A H => run (s p.1) p.2)) :
    loss H (fun a => (n a : ℝ≥0)) (dataMeans n y) ≤
      ∫ p, randomizedPayoff s p ∂σ.prod (flatWorldLaw H n y) := by
  rw [flat_randomizedRisk_eq σ s n y hn hs]
  exact randomized_rawRisk_ge_loss σ s _ _ (fun a => by exact_mod_cast hn a) hregular hs

end Measurable

open Draft0.BayesianModel Junpei.Policy Junpei.PolicyContinuation Junpei.SequentialRisk

theorem flat_familyRisk_eq {H : ℕ} (s : Family H) (n : Counts)
    (y : (a : Arm) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a) (hs : Feasible s) :
    (∫ p, randomizedPayoff s p ∂(seedLaw (H + 1)).prod (flatWorldLaw H n y)) =
      familyRisk s n (dataMeans n y) :=
  flat_randomizedRisk_eq (seedLaw (H + 1)) s n y hn hs.2

/-- In particular the paper's posterior-state policy starts from the actual
normalized flat-prior likelihood, with no separate assumed posterior law. -/
theorem flat_continuationRisk_eq (π : StatePolicy) (t H : ℕ) (n : Counts)
    (y : (a : Arm) → Fin (n a) → ℝ) (hn : ∀ a, 0 < n a) :
    (∫ p, randomizedPayoff (rawTree π t H n (dataMeans n y)) p
      ∂(seedLaw (H + 1)).prod (flatWorldLaw H n y)) =
        continuationRisk π t H n (dataMeans n y) :=
  flat_familyRisk_eq (rawTree π t H n (dataMeans n y)) n y hn
    (rawTree_feasible π t H n (dataMeans n y))

end
end Junpei.FixedGaussian.FlatInitialization
