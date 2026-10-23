import Junpei.InitialOptimalDisintegration

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.InitialOptimal

open Draft0.BayesianModel
open Junpei.Policy Junpei.PolicyContinuation Junpei.CountModelBridge
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution Junpei.SequentialRisk

noncomputable section

/-- Initial actual-risk minimization forces the risk of almost every fixed
seed tree to equal the Bellman optimum. No local optimality is assumed. -/
theorem minimizing_seed_risk_eq (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t H c m) :
    ∀ᵐ u ∂seedLaw (H + 1),
      rawRisk (rawTree π t H c m u) (natPrecisions c) m =
        Junpei.FinitePosterior.loss H (natPrecisions c) m := by
  exact ae_eq_const_of_lower_integral _ _ _
    (familyRisk_integrable _ c m (rawTree_feasible π t H c m))
    (fun u => rawRisk_ge_loss _ _ _ (natPrecisions_positive c hc)
      (rawTree_regular π t H c m u))
    (minimizing_risk_le_loss π t H c m hc hπ)

/-- A fixed seed tree attaining the actual Bayes optimum agrees almost
surely with the measurable pointwise-optimal repair, on its actual Gaussian
execution. The induction derives optimality of every reached decision from
initial risk equality. -/
theorem exact_seed_repair_ae (π : StatePolicy) (T t H : ℕ) (c : Counts) (m : Means)
    (u : Seeds H) (horizon : t + H = T + 1) (ht : 4 ≤ t)
    (hc : ∀ a, 0 < c a)
    (heq : rawRisk (rawTree π t H c m u) (natPrecisions c) m =
      Junpei.FinitePosterior.loss H (natPrecisions c) m) :
    ∀ᵐ w ∂worldLaw H (natPrecisions c) m,
      run (rawTree π t H c m u) (observations w) =
        run (rawTree (repair π T) t H c m u) (observations w) := by
  induction H generalizing t c m with
  | zero =>
      have hgood : m (π.recommend c m (u ⟨0, by omega⟩)) = terminal m := by
        rw [rawTree_zero, rawRisk_zero, Junpei.FinitePosterior.terminalRegret_eq] at heq
        unfold Junpei.FinitePosterior.loss at heq
        rw [value_eq, Draft0.BayesianModel.value] at heq
        linarith
      exact Filter.Eventually.of_forall (fun _ => by
        simp only [rawTree, run, repair, repairedRecommend, if_pos hgood])
  | succ H ih =>
      have hremaining : T - t = H := by omega
      let a := π.draw t c m (u ⟨0, by omega⟩)
      let next : ℝ → Junpei.FiniteGaussian.Strategy (A := Arm) H := fun y =>
        rawTree π (t + 1) H (advance c a)
          (shift m a ((y - m a) / ((c a : ℝ) + 1))) (seedTail u)
      have htree : rawTree π t (H + 1) c m u = .sample a next := rfl
      have hreg : Regular (.sample a next) := by
        rw [← htree]
        exact rawTree_regular π t (H + 1) c m u
      have hca : ∀ b, 0 < advance c a b := by
        intro b
        have hb := hc b
        simp only [advance]
        omega
      have hact : Junpei.FinitePosterior.actionLoss H (natPrecisions c) m a =
          Junpei.FinitePosterior.loss (H + 1) (natPrecisions c) m := by
        apply le_antisymm
        · calc
            _ ≤ rawRisk (.sample a next) (natPrecisions c) m :=
              rawRisk_ge_actionLoss a next _ _ (natPrecisions_positive c hc) hreg
            _ = _ := by rw [← htree, heq]
        · exact Junpei.BayesOptimal.actionLoss_minimal H _ _
            (natPrecisions_positive c hc) a
      have hgood : action (T - t) c m (π.draw t c m (u ⟨0, by omega⟩)) =
          value (T - t + 1) c m := by
        rw [hremaining]
        exact (actionLoss_eq_loss_iff H c m a (hc a)).mp hact
      have hrepair : (repair π T).draw t c m (u ⟨0, by omega⟩) = a := by
        simp only [repair, repairedDraw, if_pos hgood]
        rfl
      let nextRepair : ℝ → Junpei.FiniteGaussian.Strategy (A := Arm) H := fun y =>
        rawTree (repair π T) (t + 1) H (advance c a)
          (shift m a ((y - m a) / ((c a : ℝ) + 1))) (seedTail u)
      have htreeRepair : rawTree (repair π T) t (H + 1) c m u =
          .sample a nextRepair := by
        simp only [rawTree, hrepair]
        rfl
      have hregRepair : Regular (.sample a nextRepair) := by
        rw [← htreeRepair]
        exact rawTree_regular (repair π T) t (H + 1) c m u
      have hdata := rawRisk_succ_data a next (natPrecisions c) m
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
          (Junpei.FinitePosterior.advance_positive _ a (natPrecisions_positive c hc))
          (hreg.1 _)
      have hchild : ∀ᵐ z ∂gaussianReal 0
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
      rw [htree, htreeRepair]
      apply sample_run_ae_of_conditional a next nextRepair (natPrecisions c) m
        (natPrecisions_positive c hc a) hreg.2 hregRepair.2
      filter_upwards [hchild] with z hz
      have hrecover :
          (m a + (((natPrecisions c a : ℝ≥0) : ℝ) + 1) * z - m a) /
            ((c a : ℝ) + 1) = z := by
        simpa only [natPrecisions, NNReal.coe_natCast] using
          recovered_innovation (natPrecisions c a) (m a) z
      simp only [next, hrecover, advance_natPrecisions, shift_eq] at hz
      simpa only [next, nextRepair, hrecover, advance_natPrecisions, shift_eq] using
        ih (t + 1) (advance c a) (shift m a z) (seedTail u) (by omega) (by omega) hca hz

theorem actual_run_joint_measurable (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means) :
    Measurable (fun p : Seeds H × (Means × RawStream Arm H) =>
      run (rawTree π t H c m p.1) (observations p.2)) := by
  exact (rawTree_run_measurable π t H c).comp
    (measurable_const.prodMk (measurable_fst.prodMk
      ((observations_measurable H).comp measurable_snd)))

/-- A policy minimizing actual risk at this one starting state has the same
recommendation almost surely as its pointwise Bellman-optimal repair.
Neither sequential optimality nor local a.e. optimality is a premise. -/
theorem minimizing_rawTree_repair_ae (π : StatePolicy) (T t H : ℕ)
    (c : Counts) (m : Means) (horizon : t + H = T + 1) (ht : 4 ≤ t)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t H c m) :
    ∀ᵐ p ∂(seedLaw (H + 1)).prod (worldLaw H (natPrecisions c) m),
      run (rawTree π t H c m p.1) (observations p.2) =
        run (rawTree (repair π T) t H c m p.1) (observations p.2) := by
  apply (Measure.ae_prod_iff_ae_ae (measurableSet_eq_fun_of_countable
    (actual_run_joint_measurable π t H c m)
    (actual_run_joint_measurable (repair π T) t H c m))).mpr
  filter_upwards [minimizing_seed_risk_eq π t H c m hc hπ] with u hu
  exact exact_seed_repair_ae π T t H c m u horizon ht hc hu

/-- The initialization-only optimality predicate: actual continuation risk
is minimized after the mandatory one observation per arm, for each possible
initial posterior mean. It contains no sequential optimality requirement. -/
def InitialRiskOptimal (π : StatePolicy) (T : ℕ) : Prop :=
  ∀ m : Means, MinimizesActualRisk π 4 (T - 3) (fun _ => 1) m

theorem initialRiskOptimal_repair_pointwiseOptimal (π : StatePolicy) (T : ℕ) :
    PointwiseOptimal (repair π T) T := repair_pointwiseOptimal π T

/-- The posterior-predictive recommendation equivalence needed by the
initialization-only reduction, proved directly from its actual-risk premise. -/
theorem initialRiskOptimal_repair_ae (π : StatePolicy) (T : ℕ) (hT : 3 ≤ T)
    (hπ : InitialRiskOptimal π T) (m : Means) :
    ∀ᵐ p ∂(seedLaw (T - 3 + 1)).prod
      (worldLaw (T - 3) (natPrecisions (fun _ => 1)) m),
      run (rawTree π 4 (T - 3) (fun _ => 1) m p.1) (observations p.2) =
        run (rawTree (repair π T) 4 (T - 3) (fun _ => 1) m p.1) (observations p.2) := by
  exact minimizing_rawTree_repair_ae π T 4 (T - 3) (fun _ => 1) m
    (by omega) (by omega) (by intro a; norm_num) (hπ m)

end
end Junpei.InitialOptimal
