import Junpei.MockMeasurable
import Junpei.MockLaw
import Junpei.EmbeddingRandomized

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockRisk

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.FiniteModelExclusion
open Junpei.FiniteModelMeasurable Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open Junpei.EmbeddingRandomized Junpei.MockPolicy Junpei.MockMeasurable Junpei.MockLaw

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

/-- The literal mock loss in the paper: actual latent parameters are kept,
the complete auxiliary copy includes the initial free mock sample, and the
remaining H actual samples are executed by the discard/replay policy. -/
def mockLoss (H : ℕ) (c : Precisions A) (m : Means A) : ℝ :=
  ∫ w : MockStream.World A (H + 1) i,
    maxAll (MockStream.theta (H + 1) m i w) -
      MockStream.theta (H + 1) m i w
        (run (mockPolicy i H c m (auxiliaryOfBlock i H m w.1.2))
          (tail (MockStream.realStream (H + 1) m i w))) ∂MockStream.law (H + 1) c i

theorem substitute_real_mock (H : ℕ) (m : Means A)
    (w : MockStream.World A (H + 1) i) :
    substitute i (MockStream.realStream (H + 1) m i w) (auxiliaryOfBlock i H m w.1.2) =
      MockStream.mockStream (H + 1) m i w := by
  funext a t
  by_cases h : a = i <;>
    simp [substitute, MockStream.mockStream, MockStream.realStream, MockStream.swap,
      auxiliaryOfBlock, h]

theorem mockLoss_eq_modified_mock_risk (H : ℕ) (c : Precisions A) (m : Means A) :
    mockLoss i H c m = Junpei.MockSample.risk (H + 1) c m i
      (MockStream.mockStream (H + 1) m i)
      (run (terminalModified i (forcedPolicy i H c m) c m)) := by
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro w
  have he := mockPolicy_run_eq_full i H c m (auxiliaryOfBlock i H m w.1.2)
    (MockStream.realStream (H + 1) m i w)
  rw [substitute_real_mock] at he
  dsimp only
  rw [he]

theorem mockLoss_eq_modified_real_risk (H : ℕ) (c : Precisions A) (m : Means A) :
    mockLoss i H c m = Junpei.MockSample.risk (H + 1) c m i
      (MockStream.realStream (H + 1) m i)
      (run (terminalModified i (forcedPolicy i H c m) c m)) := by
  rw [mockLoss_eq_modified_mock_risk]
  exact Junpei.MockSample.risk_coupling (H + 1) c m i _
    (forcedPolicy_modified_run_measurable i H c m)
    (terminalModified_excludes i (forcedPolicy i H c m) c m)

theorem real_risk_eq_rawRisk (H : ℕ) (c : Precisions A) (m : Means A)
    (s : Strategy (A := A) H) (hs : Measurable (run s)) :
    Junpei.MockSample.risk H c m i (MockStream.realStream H m i) (run s) = rawRisk s c m := by
  have hm := realWorld_preserving H c m i
  have hi := integral_map (μ := MockStream.law H c i)
    hm.measurable.aemeasurable (rawPayoff_measurable s hs).aestronglyMeasurable
  rw [hm.map_eq] at hi
  symm
  unfold rawRisk Junpei.MockSample.risk
  rw [hi]
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro w
  simp only [real_observations]

theorem mockLoss_le_forced_rawRisk (H : ℕ) (c : Precisions A) (m : Means A) :
    mockLoss i H c m ≤ rawRisk (forcedPolicy i H c m) c m + exclusion i c m := by
  rw [mockLoss_eq_modified_real_risk]
  have hmod := forcedPolicy_modified_run_measurable i H c m
  have horig := forcedPolicy_run_measurable i H c m
  have h := integral_mono
    (Junpei.MockSample.risk_integrable (H + 1) c m i _ _
      (MockStream.realStream_measurable (H + 1) m i) hmod)
    ((Junpei.MockSample.risk_integrable (H + 1) c m i _ _
      (MockStream.realStream_measurable (H + 1) m i) horig).add
      (Junpei.MockSample.exclusion_world_integrable i (H + 1) c m)) (fun w => by
        have hp := terminalModified_regret_bound i (forcedPolicy i H c m) c m
          (MockStream.realStream (H + 1) m i w) (MockStream.theta (H + 1) m i w)
        simp only [Pi.add_apply]
        linarith)
  dsimp only [Pi.add_apply] at h
  rw [integral_add
    (Junpei.MockSample.risk_integrable (H + 1) c m i _ _
      (MockStream.realStream_measurable (H + 1) m i) horig)
    (Junpei.MockSample.exclusion_world_integrable i (H + 1) c m),
    Junpei.MockSample.exclusion_world_integral i (H + 1) c m] at h
  change Junpei.MockSample.risk (H + 1) c m i
    (MockStream.realStream (H + 1) m i) (run (terminalModified i (forcedPolicy i H c m) c m)) ≤
    Junpei.MockSample.risk (H + 1) c m i
      (MockStream.realStream (H + 1) m i) (run (forcedPolicy i H c m)) + exclusion i c m at h
  rw [real_risk_eq_rawRisk i (H + 1) c m _ horig] at h
  exact h

theorem forced_rawRisk_eq_actionLoss (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) : rawRisk (forcedPolicy i H c m) c m = actionLoss H c m i := by
  have hreg := forcedPolicy_regular i H c m
  change rawRisk (.sample i (continuation i H c m)) c m = _
  rw [rawRisk_succ i (continuation i H c m) c m (hc i) hreg.2]
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro z
  simp only [continuation, recovered_innovation]
  exact optimalFirst_rawRisk H (advance c i) (shift m i z) (advance_positive c i hc)

theorem mockLoss_eq_randomized_rawRisk (H : ℕ) (c : Precisions A) (m : Means A) :
    mockLoss i H c m =
      ∫ b : MockStream.Block (H + 1),
        rawRisk (mockPolicy i H c m (auxiliaryOfBlock i H m b)) c m
          ∂MockStream.blockLaw (H + 1) (c i)⁻¹ := by
  let s : MockStream.Block (H + 1) → Strategy (A := A) H :=
    fun b => mockPolicy i H c m (auxiliaryOfBlock i H m b)
  have hs := mockPolicy_block_run_measurable i H c m
  have hmeas := randomizedPayoff_measurable s hs
  have hint := randomizedPayoff_integrable (MockStream.blockLaw (H + 1) (c i)⁻¹) s c m hs
  have hm := seed_and_real_preserving H c m i
  have hi := integral_map (μ := MockStream.law (H + 1) c i)
    hm.measurable.aemeasurable hmeas.aestronglyMeasurable
  rw [hm.map_eq, integral_prod _ hint] at hi
  change mockLoss i H c m = ∫ b, ∫ w, randomizedPayoff s (b, w) ∂worldLaw H c m
    ∂MockStream.blockLaw (H + 1) (c i)⁻¹
  rw [hi]
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro w
  have ho : observations (MockStream.theta (H + 1) m i w,
      tail (realNoise (H + 1) i w)) = tail (MockStream.realStream (H + 1) m i w) := by
    change tail (observations (MockStream.theta (H + 1) m i w, realNoise (H + 1) i w)) = _
    rw [real_observations]
  simp only [randomizedPayoff, s, ho]

/-- Both displayed inequalities of Definition 5, for arbitrary finite arm
sets and proper positive real precisions. Every policy, raw law, coupling,
measurability, discarded sample, and budget is constructed above. -/
theorem mock_loss_chain (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) :
    loss H c m ≤ mockLoss i H c m ∧ mockLoss i H c m ≤ actionLoss H c m i + exclusion i c m := by
  constructor
  · rw [mockLoss_eq_randomized_rawRisk]
    exact randomized_rawRisk_ge_loss (MockStream.blockLaw (H + 1) (c i)⁻¹)
      (fun b => mockPolicy i H c m (auxiliaryOfBlock i H m b)) c m hc
      (fun b => mockPolicy_regular i H c m _)
      (mockPolicy_block_run_measurable i H c m)
  · rw [← forced_rawRisk_eq_actionLoss i H c m hc]
    exact mockLoss_le_forced_rawRisk i H c m

/-- Lemma 12 obtained by the paper's actual mock construction and displayed
risk sandwich, independently of the previous reduced-Bellman proof. -/
theorem lemma12_by_mock_samples (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) : loss H c m - actionLoss H c m i ≤ exclusion i c m := by
  have h := mock_loss_chain i H c m hc
  linarith

end Removed
end
end Junpei.MockRisk
