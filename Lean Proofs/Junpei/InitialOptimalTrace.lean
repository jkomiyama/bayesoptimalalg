import Junpei.InitialOptimalPaths

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.InitialOptimal

open Draft0.BayesianModel
open Junpei.Policy Junpei.PolicyContinuation Junpei.CountModelBridge
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution Junpei.SequentialRisk

noncomputable section

instance : MeasurableSpace (List Arm) := ⊤

instance : MeasurableSingletonClass (List Arm) where
  measurableSet_singleton _ := trivial

/-- Conditional a.e. properties of an observed reward and remaining raw
observations hold in the parent actual latent/noise execution. -/
theorem conditional_property_ae {H : ℕ} (a : Arm)
    (c : Junpei.FiniteGaussian.Precisions Arm) (m : Means) (hc : 0 < c a)
    (Q : ℝ × RawStream Arm H → Prop) (hQ : MeasurableSet {p | Q p})
    (hcond : ∀ᵐ z ∂gaussianReal 0 (Junpei.FiniteGaussian.innovation c a),
      ∀ᵐ w ∂worldLaw H (Junpei.FiniteGaussian.advance c a)
        (Junpei.FiniteGaussian.shift m a z),
        Q (m a + ((c a : ℝ) + 1) * z, observations w)) :
    ∀ᵐ w ∂worldLaw (H + 1) c m,
      Q (observations w a ⟨0, by omega⟩, tail (observations w)) := by
  let P : (Means × ℝ) × RawStream Arm H → Prop := fun p =>
    Q (p.1.2, observations (p.1.1, p.2))
  have hP : MeasurableSet {p | P p} := hQ.preimage (by
    unfold observations
    fun_prop)
  have hpost' : ∀ᵐ p : ℝ × (Means × RawStream Arm H)
      ∂(gaussianReal 0 (Junpei.FiniteGaussian.innovation c a)).prod
        ((Junpei.FinitePosterior.centeredLaw (Junpei.FiniteGaussian.advance c a)).prod
          (arrayLaw H)),
      P (ObservationBridge.reconstructedObservation c m a (p.1, p.2.1), p.2.2) := by
    apply (Measure.ae_prod_iff_ae_ae (hP.preimage (by fun_prop))).mpr
    filter_upwards [hcond] with z hz
    have htrans : MeasurePreserving (fun v : Means =>
        Junpei.FiniteGaussian.shift m a z + v)
        (Junpei.FinitePosterior.centeredLaw (Junpei.FiniteGaussian.advance c a))
        (Junpei.FinitePosterior.posteriorLaw (Junpei.FiniteGaussian.advance c a)
          (Junpei.FiniteGaussian.shift m a z)) := ⟨by fun_prop, rfl⟩
    exact (htrans.prod (MeasurePreserving.id (arrayLaw H))).quasiMeasurePreserving.ae hz
  have hpost := (measurePreserving_prodAssoc
    (gaussianReal 0 (Junpei.FiniteGaussian.innovation c a))
    (Junpei.FinitePosterior.centeredLaw (Junpei.FiniteGaussian.advance c a))
    (arrayLaw H)).quasiMeasurePreserving.ae hpost'
  have hraw := observation_ae_transfer c m a hc P hP hpost
  have hsplit := (MeasurePreserving.symm MeasurableEquiv.prodAssoc
    (measurePreserving_prodAssoc (Junpei.FinitePosterior.posteriorLaw c m)
      (gaussianReal 0 1) (arrayLaw H))).comp
      ((MeasurePreserving.id (Junpei.FinitePosterior.posteriorLaw c m)).prod
        (selected_noise_split_preserving H a))
  exact hsplit.quasiMeasurePreserving.ae hraw

/-- Exact initial risk equality forces both optimality of the first arm and
exact conditional continuation-risk equality almost everywhere. -/
theorem exact_seed_successor_data (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means)
    (u : Seeds (H + 1)) (hc : ∀ a, 0 < c a)
    (heq : rawRisk (rawTree π t (H + 1) c m u) (natPrecisions c) m =
      Junpei.FinitePosterior.loss (H + 1) (natPrecisions c) m) :
    let a := π.draw t c m (u ⟨0, by omega⟩)
    action H c m a = value (H + 1) c m ∧
      ∀ᵐ z ∂gaussianReal 0 (innovation c a),
        rawRisk (rawTree π (t + 1) H (advance c a) (shift m a z) (seedTail u))
          (natPrecisions (advance c a)) (shift m a z) =
          Junpei.FinitePosterior.loss H (natPrecisions (advance c a)) (shift m a z) := by
  let a := π.draw t c m (u ⟨0, by omega⟩)
  let next : ℝ → Junpei.FiniteGaussian.Strategy (A := Arm) H := fun y =>
    rawTree π (t + 1) H (advance c a)
      (shift m a ((y - m a) / ((c a : ℝ) + 1))) (seedTail u)
  have htree : rawTree π t (H + 1) c m u = .sample a next := rfl
  have hreg : Regular (.sample a next) := by
    rw [← htree]
    exact rawTree_regular π t (H + 1) c m u
  have hact : Junpei.FinitePosterior.actionLoss H (natPrecisions c) m a =
      Junpei.FinitePosterior.loss (H + 1) (natPrecisions c) m := by
    apply le_antisymm
    · calc
        _ ≤ rawRisk (.sample a next) (natPrecisions c) m :=
          rawRisk_ge_actionLoss a next _ _ (natPrecisions_positive c hc) hreg
        _ = _ := by rw [← htree, heq]
    · exact Junpei.BayesOptimal.actionLoss_minimal H _ _ (natPrecisions_positive c hc) a
  constructor
  · exact (actionLoss_eq_loss_iff H c m a (hc a)).mp hact
  · have hdata := rawRisk_succ_data a next (natPrecisions c) m
      (natPrecisions_positive c hc a) hreg.2
    have hlowint : Integrable (fun z => Junpei.FinitePosterior.loss H
        (Junpei.FiniteGaussian.advance (natPrecisions c) a)
        (Junpei.FiniteGaussian.shift m a z))
        (gaussianReal 0 (Junpei.FiniteGaussian.innovation (natPrecisions c) a)) :=
      (Junpei.FiniteGaussian.shifted_integrable (Junpei.FinitePosterior.potential_lipschitz _)
        m a _).sub (Junpei.FiniteGaussian.action_integrable H (natPrecisions c) m a)
    have hlow (z : ℝ) : Junpei.FinitePosterior.loss H
        (Junpei.FiniteGaussian.advance (natPrecisions c) a)
        (Junpei.FiniteGaussian.shift m a z) ≤
        rawRisk (next (m a + (((natPrecisions c a : ℝ≥0) : ℝ) + 1) * z))
          (Junpei.FiniteGaussian.advance (natPrecisions c) a)
          (Junpei.FiniteGaussian.shift m a z) :=
      rawRisk_ge_loss _ _ _
        (Junpei.FinitePosterior.advance_positive _ a (natPrecisions_positive c hc)) (hreg.1 _)
    have he : ∀ᵐ z ∂gaussianReal 0
        (Junpei.FiniteGaussian.innovation (natPrecisions c) a),
        rawRisk (next (m a + (((natPrecisions c a : ℝ≥0) : ℝ) + 1) * z))
          (Junpei.FiniteGaussian.advance (natPrecisions c) a)
          (Junpei.FiniteGaussian.shift m a z) =
        Junpei.FinitePosterior.loss H
          (Junpei.FiniteGaussian.advance (natPrecisions c) a)
          (Junpei.FiniteGaussian.shift m a z) := by
      apply ((integral_eq_iff_of_ae_le hlowint hdata.2
        (Filter.Eventually.of_forall hlow)).mp ?_).symm
      change Junpei.FinitePosterior.actionLoss H (natPrecisions c) m a = _
      rw [← hdata.1, ← htree, heq, hact]
    have hrecover (z : ℝ) :
        (m a + (((natPrecisions c a : ℝ≥0) : ℝ) + 1) * z - m a) /
          ((c a : ℝ) + 1) = z := by
      simpa only [natPrecisions, NNReal.coe_natCast] using
        recovered_innovation (natPrecisions c a) (m a) z
    simpa only [next, hrecover, advance_natPrecisions, shift_eq, innovation_eq] using he

theorem rawTree_trace_measurable (π : StatePolicy) (t H : ℕ) (c : Counts) :
    Measurable (fun p : Input H => trace (rawTree π t H c p.1 p.2.1) p.2.2) := by
  induction H generalizing t c with
  | zero => exact measurable_const
  | succ H ih =>
    let g : Input (H + 1) × Arm → List Arm := fun p =>
      p.2 :: trace (rawTree π (t + 1) H (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2.2 p.2 ⟨0, by omega⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1))) (seedTail p.1.2.1)) (tail p.1.2.2)
    have hg : Measurable g := by
      apply measurable_from_prod_countable_left
      intro a
      have hin : Measurable (fun p : Input (H + 1) =>
          (shift p.1 a ((p.2.2 a ⟨0, by omega⟩ - p.1 a) / ((c a : ℝ) + 1)),
            (seedTail p.2.1, tail p.2.2))) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro b
          simp only [shift]
          split_ifs <;> fun_prop
        · unfold seedTail tail
          fun_prop
      exact (measurable_of_countable (List.cons a)).comp ((ih (t + 1) (advance c a)).comp hin)
    have hdraw : Measurable (fun p : Input (H + 1) =>
        π.draw t c p.1 (p.2.1 ⟨0, by omega⟩)) := by
      exact (π.draw_measurable t).comp (measurable_const.prodMk
        (measurable_fst.prodMk ((measurable_pi_apply (⟨0, by omega⟩ : Fin (H + 2))).comp
          (measurable_fst.comp measurable_snd))))
    exact hg.comp (measurable_id.prodMk hdraw)

/-- The same null-history repair preserves the full ordered list of sampled
arms, rather than only the final recommendation. -/
theorem exact_seed_trace_repair_ae (π : StatePolicy) (T t H : ℕ) (c : Counts) (m : Means)
    (u : Seeds H) (horizon : t + H = T + 1) (ht : 4 ≤ t)
    (hc : ∀ a, 0 < c a)
    (heq : rawRisk (rawTree π t H c m u) (natPrecisions c) m =
      Junpei.FinitePosterior.loss H (natPrecisions c) m) :
    ∀ᵐ w ∂worldLaw H (natPrecisions c) m,
      trace (rawTree π t H c m u) (observations w) =
        trace (rawTree (repair π T) t H c m u) (observations w) := by
  induction H generalizing t c m with
  | zero => exact Filter.Eventually.of_forall (fun _ => rfl)
  | succ H ih =>
      have hremaining : T - t = H := by omega
      let a := π.draw t c m (u ⟨0, by omega⟩)
      have hdata := exact_seed_successor_data π t H c m u hc heq
      have hrepair : (repair π T).draw t c m (u ⟨0, by omega⟩) = a := by
        have hgood : action (T - t) c m (π.draw t c m (u ⟨0, by omega⟩)) =
            value (T - t + 1) c m := by
          rw [hremaining]
          exact hdata.1
        simp only [repair, repairedDraw, if_pos hgood]
        rfl
      have hca : ∀ b, 0 < advance c a b := by
        intro b
        have hb := hc b
        simp only [advance]
        omega
      let Q : ℝ × RawStream Arm H → Prop := fun p =>
        trace (rawTree π (t + 1) H (advance c a)
          (shift m a ((p.1 - m a) / ((c a : ℝ) + 1))) (seedTail u)) p.2 =
        trace (rawTree (repair π T) (t + 1) H (advance c a)
          (shift m a ((p.1 - m a) / ((c a : ℝ) + 1))) (seedTail u)) p.2
      have hQ : MeasurableSet {p | Q p} := by
        have hin : Measurable (fun p : ℝ × RawStream Arm H =>
            (shift m a ((p.1 - m a) / ((c a : ℝ) + 1)), (seedTail u, p.2))) := by
          apply Measurable.prodMk
          · apply measurable_pi_lambda
            intro b
            simp only [shift]
            split_ifs <;> fun_prop
          · exact measurable_const.prodMk measurable_snd
        exact measurableSet_eq_fun_of_countable
          ((rawTree_trace_measurable π (t + 1) H (advance c a)).comp hin)
          ((rawTree_trace_measurable (repair π T) (t + 1) H (advance c a)).comp hin)
      have hcond : ∀ᵐ z ∂gaussianReal 0
          (Junpei.FiniteGaussian.innovation (natPrecisions c) a),
          ∀ᵐ w ∂worldLaw H (Junpei.FiniteGaussian.advance (natPrecisions c) a)
            (Junpei.FiniteGaussian.shift m a z),
            Q (m a + (((natPrecisions c a : ℝ≥0) : ℝ) + 1) * z, observations w) := by
        filter_upwards [hdata.2] with z hz
        have hrecover :
            (m a + (((natPrecisions c a : ℝ≥0) : ℝ) + 1) * z - m a) /
              ((c a : ℝ) + 1) = z := by
          simpa only [natPrecisions, NNReal.coe_natCast] using
            recovered_innovation (natPrecisions c a) (m a) z
        simpa only [Q, hrecover, advance_natPrecisions, shift_eq] using
          ih (t + 1) (advance c a) (shift m a z) (seedTail u) (by omega) (by omega) hca hz
      have hp := conditional_property_ae a (natPrecisions c) m
        (natPrecisions_positive c hc a) Q hQ hcond
      filter_upwards [hp] with w hw
      simp only [rawTree, hrepair, trace]
      exact congrArg (List.cons a) hw

theorem actual_trace_joint_measurable (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means) :
    Measurable (fun p : Seeds H × (Means × RawStream Arm H) =>
      trace (rawTree π t H c m p.1) (observations p.2)) := by
  exact (rawTree_trace_measurable π t H c).comp
    (measurable_const.prodMk (measurable_fst.prodMk
      ((observations_measurable H).comp measurable_snd)))

/-- Actual-risk minimization at one state preserves the entire sampling
trace under null-history repair, a.e. over both seeds and actual observations. -/
theorem minimizing_trace_repair_ae (π : StatePolicy) (T t H : ℕ)
    (c : Counts) (m : Means) (horizon : t + H = T + 1) (ht : 4 ≤ t)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t H c m) :
    ∀ᵐ p ∂(seedLaw (H + 1)).prod (worldLaw H (natPrecisions c) m),
      trace (rawTree π t H c m p.1) (observations p.2) =
        trace (rawTree (repair π T) t H c m p.1) (observations p.2) := by
  apply (Measure.ae_prod_iff_ae_ae (measurableSet_eq_fun_of_countable
    (actual_trace_joint_measurable π t H c m)
    (actual_trace_joint_measurable (repair π T) t H c m))).mpr
  filter_upwards [minimizing_seed_risk_eq π t H c m hc hπ] with u hu
  exact exact_seed_trace_repair_ae π T t H c m u horizon ht hc hu

theorem deterministicBellman_initialRiskOptimal (T : ℕ) (hT : 3 ≤ T) :
    InitialRiskOptimal (deterministicBellman T) T := by
  intro m
  exact deterministicBellman_minimizes_actual_risk T 4 (T - 3) (fun _ => 1) m
    (by omega) (by intro a; norm_num)

theorem exists_initialRiskOptimal (T : ℕ) (hT : 3 ≤ T) :
    ∃ π : StatePolicy, InitialRiskOptimal π T :=
  ⟨deterministicBellman T, deterministicBellman_initialRiskOptimal T hT⟩

end
end Junpei.InitialOptimal
