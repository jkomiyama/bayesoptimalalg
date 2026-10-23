import Junpei.EmbeddingRandomized
import Junpei.MockLaw
import Junpei.MockSample

set_option linter.unusedSectionVars false
set_option synthInstance.maxHeartbeats 400000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.EmbeddingCoupling

open FiniteGaussian FinitePosterior FiniteModelExclusion EmbeddingDirect
open EmbeddingRaw EmbeddingExecution EmbeddingRandomized

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

def retainedWorld (H : ℕ) (m : Means A) (w : MockStream.World A H i) :
    Means (Retained i) × RawStream (Retained i) H :=
  (fun a => MockStream.theta H m i w a, w.2.2)

def copiedObservations {H : ℕ} (m : Means A) (b : MockStream.Block H) : Fin H → ℝ :=
  fun t => m i + b.1 + b.2 t

theorem retainedData_preserving (H : ℕ) (c : Precisions A) (m : Means A) :
    MeasurePreserving (fun p : MockStream.OtherData A H i =>
      (restrictMeans i m + p.1, p.2)) (MockStream.otherDataLaw H c i)
      (worldLaw H (restrictPrecisions i c) (restrictMeans i m)) := by
  have ht : MeasurePreserving (fun x : Means (Retained i) => restrictMeans i m + x)
      (centeredLaw (restrictPrecisions i c))
      (posteriorLaw (restrictPrecisions i c) (restrictMeans i m)) := ⟨by fun_prop, rfl⟩
  exact ht.prod (MeasurePreserving.id (arrayLaw (A := Retained i) H))

/-- The imaginary removed-arm block is independent of the true retained
latent parameters and all their real reward noise. -/
theorem copy_retained_preserving (H : ℕ) (c : Precisions A) (m : Means A) :
    MeasurePreserving (fun w : MockStream.World A H i => (w.1.2, retainedWorld i H m w))
      (MockStream.law H c i)
      ((MockStream.blockLaw H (c i)⁻¹).prod
        (worldLaw H (restrictPrecisions i c) (restrictMeans i m))) := by
  have hp := (measurePreserving_snd (μ := MockStream.blockLaw H (c i)⁻¹)
    (ν := MockStream.blockLaw H (c i)⁻¹)).prod
      (MeasurePreserving.id (MockStream.otherDataLaw H c i))
  have h := ((MeasurePreserving.id (MockStream.blockLaw H (c i)⁻¹)).prod
    (retainedData_preserving i H c m)).comp hp
  convert h using 1
  funext w
  apply Prod.ext
  · rfl
  · apply Prod.ext
    · ext a
      simp [retainedWorld, MockStream.theta, FinitePosterior.join, a.property, restrictMeans]
    · rfl

theorem inserted_copy_stream (H : ℕ) (m : Means A) (w : MockStream.World A H i) :
    insert i (observations (retainedWorld i H m w)) (copiedObservations i m w.1.2) =
      MockStream.mockStream H m i w := by
  ext a t
  by_cases ha : a = i
  · subst a
    simp [EmbeddingRaw.insert, copiedObservations, MockStream.mockStream,
      MockStream.realStream, MockStream.swap]
  · simp [EmbeddingRaw.insert, observations, retainedWorld, MockStream.mockStream,
      MockStream.realStream, MockStream.swap, MockStream.theta, FinitePosterior.join, ha]

def simulated {H : ℕ} (dummy : Retained i) (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (b : MockStream.Block H) : Strategy (A := Retained i) H :=
  simulate i dummy s (restrictPrecisions i c) (restrictMeans i m) (copiedObservations i m b)

section Measurable
variable [MeasurableSpace A] [MeasurableSingletonClass A]

theorem mockRisk_eq_rawRisk {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (h : Measurable (run s)) :
    MockSample.risk H c m i (MockStream.realStream H m i) (run s) = rawRisk s c m := by
  unfold rawRisk
  rw [← (MockLaw.realWorld_preserving H c m i).map_eq,
    integral_map (MockLaw.realWorld_preserving H c m i).measurable.aemeasurable
      (rawPayoff_measurable s h).aestronglyMeasurable]
  simp only [MockLaw.real_observations]
  rfl

theorem simulatedRisk_eq_world {H : ℕ} (dummy : Retained i) (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A)
    (hmeas : Measurable (fun p : MockStream.Block H × RawStream (Retained i) H =>
      run (A := Retained i) (simulated i dummy s c m p.1) p.2)) :
    (∫ b, rawRisk (simulated i dummy s c m b) (restrictPrecisions i c) (restrictMeans i m)
      ∂MockStream.blockLaw H (c i)⁻¹) =
    ∫ w, randomizedPayoff (simulated i dummy s c m) (w.1.2, retainedWorld i H m w)
      ∂MockStream.law H c i := by
  have hi := randomizedPayoff_integrable (MockStream.blockLaw H (c i)⁻¹)
    (simulated i dummy s c m) (restrictPrecisions i c) (restrictMeans i m) hmeas
  change (∫ b, ∫ v, randomizedPayoff (simulated i dummy s c m) (b, v)
    ∂worldLaw H (restrictPrecisions i c) (restrictMeans i m)
    ∂MockStream.blockLaw H (c i)⁻¹) = _
  rw [← integral_prod _ hi, ← (copy_retained_preserving i H c m).map_eq,
    integral_map (copy_retained_preserving i H c m).measurable.aemeasurable
      (randomizedPayoff_measurable (simulated i dummy s c m) hmeas).aestronglyMeasurable]

/-- Risk comparison for the literal budget-preserving simulator, before
any optimality argument. Both risks use the actual latent Gaussian laws. -/
theorem simulatedRisk_le_fullRisk {H : ℕ} (dummy : Retained i) (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hfull : Measurable (run s))
    (hmeas : Measurable (fun p : MockStream.Block H × RawStream (Retained i) H =>
      run (A := Retained i) (simulated i dummy s c m p.1) p.2)) :
    (∫ b, rawRisk (simulated i dummy s c m b) (restrictPrecisions i c) (restrictMeans i m)
      ∂MockStream.blockLaw H (c i)⁻¹) ≤ rawRisk s c m + exclusion i c m := by
  rw [simulatedRisk_eq_world i dummy s c m hmeas]
  have hcand := (copy_retained_preserving i H c m).integrable_comp_of_integrable
    (randomizedPayoff_integrable (MockStream.blockLaw H (c i)⁻¹)
      (simulated i dummy s c m) (restrictPrecisions i c) (restrictMeans i m) hmeas)
  have hfullInt := (MockStream.swap_preserving H c i).integrable_comp_of_integrable
    (MockSample.risk_integrable H c m i (MockStream.realStream H m i) (run s)
      (MockStream.realStream_measurable H m i) hfull)
  have hexInt := (MockStream.swap_preserving H c i).integrable_comp_of_integrable
    (MockSample.exclusion_world_integrable i H c m)
  have hb := integral_mono hcand (hfullInt.add hexInt) (fun w => by
    have hp := simulate_regret_bound i dummy s (restrictPrecisions i c) (restrictMeans i m)
      (copiedObservations i m w.1.2) (observations (retainedWorld i H m w))
      (MockStream.theta H m i (MockStream.swap H i w))
    rw [inserted_copy_stream i H m w] at hp
    rw [MockSample.otherMaximum_swap i H m w] at hp
    have hj := MockStream.theta_other_unchanged H m i
      (run (A := Retained i) (simulated i dummy s c m w.1.2)
        (observations (retainedWorld i H m w)))
      (run (A := Retained i) (simulated i dummy s c m w.1.2)
        (observations (retainedWorld i H m w))).property w
    change otherMaximum i (MockStream.theta H m i w) -
      MockStream.theta H m i (MockStream.swap H i w)
        (run (A := Retained i) (simulated i dummy s c m w.1.2)
          (observations (retainedWorld i H m w))) ≤ _ at hp
    rw [hj] at hp
    exact hp)
  have he : (∫ w, max (MockStream.theta H m i (MockStream.swap H i w) i -
      otherMinimum i (MockStream.theta H m i (MockStream.swap H i w))) 0
      ∂MockStream.law H c i) = exclusion i c m := by
    exact (MockSample.integral_swap H c i (fun w => max
      (MockStream.theta H m i w i - otherMinimum i (MockStream.theta H m i w)) 0) (by
      have htheta := MockStream.theta_measurable H m i
      exact (((measurable_pi_apply i).comp htheta).sub
        ((otherMinimum_lipschitz i).continuous.measurable.comp htheta)).max measurable_const)).trans
      (MockSample.exclusion_world_integral i H c m)
  dsimp only [Pi.add_apply, Function.comp_def] at hb hfullInt hexInt
  rw [integral_add hfullInt hexInt] at hb
  simp_rw [MockStream.stream_swap] at hb
  rw [MockSample.full_risk_coupling H c m i (run s) hfull,
    he, mockRisk_eq_rawRisk i s c m hfull] at hb
  exact hb

/-- The second embedding's feasibility inequality. Regularity and joint
measurability are properties of its explicit raw strategy, never a supplied
risk-coupling equality or a supplied Bellman endpoint. -/
theorem lemma13_second_embedding {H : ℕ} (dummy : Retained i) (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a) (hfull : Measurable (run s))
    (hregular : ∀ b, Regular (simulated i dummy s c m b))
    (hmeas : Measurable (fun p : MockStream.Block H × RawStream (Retained i) H =>
      run (A := Retained i) (simulated i dummy s c m p.1) p.2)) :
    loss H (restrictPrecisions i c) (restrictMeans i m) ≤ rawRisk s c m + exclusion i c m := by
  exact (randomized_rawRisk_ge_loss (MockStream.blockLaw H (c i)⁻¹)
    (simulated i dummy s c m) (restrictPrecisions i c) (restrictMeans i m)
    (fun a => hc a) hregular hmeas).trans
      (simulatedRisk_le_fullRisk i dummy s c m hfull hmeas)

theorem lemma13_second_embedding_forced {H : ℕ} (dummy a : Retained i)
    (next : ℝ → Strategy (A := A) H) (c : Precisions A) (m : Means A)
    (hc : ∀ b, 0 < c b) (hfull : Measurable (run (.sample (a : A) next)))
    (hregular : ∀ b, Regular (simulated i dummy (.sample a next) c m b))
    (hmeas : Measurable (fun p : MockStream.Block (H + 1) × RawStream (Retained i) (H + 1) =>
      run (A := Retained i) (simulated i dummy (.sample a next) c m p.1) p.2)) :
    actionLoss H (restrictPrecisions i c) (restrictMeans i m) a ≤
      rawRisk (.sample (a : A) next) c m + exclusion i c m := by
  have hpoint (b : MockStream.Block (H + 1)) :
      actionLoss H (restrictPrecisions i c) (restrictMeans i m) a ≤
      rawRisk (simulated i dummy (.sample a next) c m b)
        (restrictPrecisions i c) (restrictMeans i m) := by
    have he := simulate_forced_retained i dummy a next (restrictPrecisions i c)
      (restrictMeans i m) (copiedObservations i m b)
    have hr := hregular b
    change Regular (simulate i dummy (.sample a next) (restrictPrecisions i c)
      (restrictMeans i m) (copiedObservations i m b)) at hr
    rw [he] at hr
    change actionLoss H (restrictPrecisions i c) (restrictMeans i m) a ≤
      rawRisk (simulate i dummy (.sample a next) (restrictPrecisions i c)
        (restrictMeans i m) (copiedObservations i m b)) _ _
    rw [he]
    exact rawRisk_ge_actionLoss a _ _ _ (fun a => hc a) hr
  have hl := integral_mono
    (integrable_const (actionLoss H (restrictPrecisions i c) (restrictMeans i m) a))
    (randomized_rawRisk_integrable (MockStream.blockLaw (H + 1) (c i)⁻¹)
      (simulated i dummy (.sample a next) c m) (restrictPrecisions i c) (restrictMeans i m) hmeas)
    hpoint
  simp only [integral_const, measureReal_univ_eq_one, one_smul] at hl
  exact hl.trans (simulatedRisk_le_fullRisk i dummy (.sample a next) c m hfull hmeas)

end Measurable
end Removed
end
end Junpei.EmbeddingCoupling
