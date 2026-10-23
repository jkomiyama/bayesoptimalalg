import Junpei.SequentialRisk

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.InitialOptimal

open Junpei.FiniteGaussian Junpei.FinitePosterior
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution

noncomputable section

/-- An almost-sure property of a raw observation and all remaining calendar
noise can be checked in the posterior-innovation representation. -/
theorem observation_ae_transfer {H : ℕ} (c : Precisions Draft0.BayesianModel.Arm)
    (m : Means Draft0.BayesianModel.Arm) (a : Draft0.BayesianModel.Arm)
    (hc : 0 < c a)
    (P : (Means Draft0.BayesianModel.Arm × ℝ) ×
      RawStream Draft0.BayesianModel.Arm H → Prop)
    (hP : MeasurableSet {p | P p})
    (hpost : ∀ᵐ p ∂((gaussianReal 0 (innovation c a)).prod
      (centeredLaw (advance c a))).prod (arrayLaw H),
      P (ObservationBridge.reconstructedObservation c m a p.1, p.2)) :
    ∀ᵐ p ∂((posteriorLaw c m).prod (gaussianReal 0 1)).prod (arrayLaw H),
      P (ObservationBridge.rawObservation a p.1, p.2) := by
  let f := ObservationBridge.rawObservation a
  let g := ObservationBridge.reconstructedObservation c m a
  have hf : MeasurePreserving f ((posteriorLaw c m).prod (gaussianReal 0 1))
      (((posteriorLaw c m).prod (gaussianReal 0 1)).map f) :=
    ⟨by fun_prop, rfl⟩
  have hg : MeasurePreserving g
      ((gaussianReal 0 (innovation c a)).prod (centeredLaw (advance c a)))
      (((gaussianReal 0 (innovation c a)).prod (centeredLaw (advance c a))).map g) :=
    ⟨by fun_prop, rfl⟩
  have hmap :
      ((((posteriorLaw c m).prod (gaussianReal 0 1)).map f).prod
        (arrayLaw (A := Draft0.BayesianModel.Arm) H)) =
      ((((gaussianReal 0 (innovation c a)).prod
        (centeredLaw (advance c a))).map g).prod
          (arrayLaw (A := Draft0.BayesianModel.Arm) H)) := by
    rw [ObservationBridge.observation_joint_law c m a hc]
  have hgprod := hg.prod (MeasurePreserving.id (arrayLaw (A := Draft0.BayesianModel.Arm) H))
  have hp : ∀ᵐ p ∂((((gaussianReal 0 (innovation c a)).prod
      (centeredLaw (advance c a))).map g).prod (arrayLaw H)), P p := by
    rw [← hgprod.map_eq]
    exact (ae_map_iff hgprod.measurable.aemeasurable hP).mpr hpost
  rw [← hmap] at hp
  exact (hf.prod (MeasurePreserving.id
    (arrayLaw (A := Draft0.BayesianModel.Arm) H))).quasiMeasurePreserving.ae hp

/-- Equal conditional recommendations after one actual Gaussian observation
give equal parent recommendations. All remaining observations use the same
latent parameter and the same calendar-noise array. -/
theorem sample_run_ae_of_conditional {H : ℕ}
    (a : Draft0.BayesianModel.Arm)
    (s r : ℝ → Strategy (A := Draft0.BayesianModel.Arm) H)
    (c : Precisions Draft0.BayesianModel.Arm) (m : Means Draft0.BayesianModel.Arm)
    (hc : 0 < c a)
    (hs : Measurable (fun p : ℝ × RawStream Draft0.BayesianModel.Arm H =>
      run (s p.1) p.2))
    (hr : Measurable (fun p : ℝ × RawStream Draft0.BayesianModel.Arm H =>
      run (r p.1) p.2))
    (hcond : ∀ᵐ z ∂gaussianReal 0 (innovation c a),
      ∀ᵐ w ∂worldLaw H (advance c a) (shift m a z),
        run (s (m a + ((c a : ℝ) + 1) * z)) (observations w) =
          run (r (m a + ((c a : ℝ) + 1) * z)) (observations w)) :
    ∀ᵐ w ∂worldLaw (H + 1) c m,
      run (.sample a s) (observations w) = run (.sample a r) (observations w) := by
  let P : (Means Draft0.BayesianModel.Arm × ℝ) ×
      RawStream Draft0.BayesianModel.Arm H → Prop := fun p =>
    run (s p.1.2) (observations (p.1.1, p.2)) =
      run (r p.1.2) (observations (p.1.1, p.2))
  have hP : MeasurableSet {p | P p} := by
    have hin : Measurable (fun p : (Means Draft0.BayesianModel.Arm × ℝ) ×
        RawStream Draft0.BayesianModel.Arm H => (p.1.2, observations (p.1.1, p.2))) := by
      unfold observations
      fun_prop
    apply measurableSet_eq_fun_of_countable
    · exact hs.comp hin
    · exact hr.comp hin
  have hpost' : ∀ᵐ p : ℝ ×
      (Means Draft0.BayesianModel.Arm × RawStream Draft0.BayesianModel.Arm H)
      ∂(gaussianReal 0 (innovation c a)).prod ((centeredLaw (advance c a)).prod (arrayLaw H)),
      P (ObservationBridge.reconstructedObservation c m a (p.1, p.2.1), p.2.2) := by
    apply (Measure.ae_prod_iff_ae_ae (by
      exact hP.preimage (by fun_prop))).mpr
    filter_upwards [hcond] with z hz
    have ht : MeasurePreserving (fun w : Means Draft0.BayesianModel.Arm ×
        RawStream Draft0.BayesianModel.Arm H => (shift m a z + w.1, w.2))
        ((centeredLaw (advance c a)).prod (arrayLaw H))
        (worldLaw H (advance c a) (shift m a z)) := by
      have htrans : MeasurePreserving (fun v : Means Draft0.BayesianModel.Arm =>
          shift m a z + v) (centeredLaw (advance c a))
          (posteriorLaw (advance c a) (shift m a z)) := ⟨by fun_prop, rfl⟩
      exact htrans.prod (MeasurePreserving.id (arrayLaw H))
    exact ht.quasiMeasurePreserving.ae hz
  have hpost : ∀ᵐ p ∂((gaussianReal 0 (innovation c a)).prod
      (centeredLaw (advance c a))).prod (arrayLaw H),
      P (ObservationBridge.reconstructedObservation c m a p.1, p.2) := by
    exact (measurePreserving_prodAssoc (gaussianReal 0 (innovation c a))
      (centeredLaw (advance c a)) (arrayLaw H)).quasiMeasurePreserving.ae hpost'
  have hraw := observation_ae_transfer c m a hc P hP hpost
  have hsplit := (MeasurePreserving.symm MeasurableEquiv.prodAssoc
    (measurePreserving_prodAssoc (posteriorLaw c m) (gaussianReal 0 1) (arrayLaw H))).comp
      ((MeasurePreserving.id (posteriorLaw c m)).prod (selected_noise_split_preserving H a))
  exact hsplit.quasiMeasurePreserving.ae hraw

end
end Junpei.InitialOptimal
