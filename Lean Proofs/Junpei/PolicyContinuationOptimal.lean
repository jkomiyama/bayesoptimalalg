import Junpei.PolicyContinuation
import Junpei.CountModelBridge
import Junpei.BayesOptimal

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.PolicyContinuationOptimal

open Draft0.BayesianModel
open Junpei.Policy Junpei.PolicyContinuation Junpei.CountModelBridge
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution

noncomputable section

/-- The concrete Bellman posterior-state policy realizes the actual Gaussian
Bayes risk, for every fixed external seed array. The statement concerns its
raw-observation execution, not a stipulated Bellman utility. -/
theorem deterministicBellman_rawRisk (T t H : ℕ) (c : Counts) (m : Means)
    (u : Seeds H) (horizon : t + H = T + 1) (hc : ∀ a, 0 < c a) :
    rawRisk (rawTree (deterministicBellman T) t H c m u) (natPrecisions c) m =
      Junpei.FinitePosterior.loss H (natPrecisions c) m := by
  induction H generalizing t c m with
  | zero =>
      rw [rawTree_zero, rawRisk_zero, Junpei.FinitePosterior.terminalRegret_eq]
      change Junpei.FinitePosterior.potential (natPrecisions c) m -
          m (Draft0.BayesianModel.recommend m) =
        Junpei.FinitePosterior.potential (natPrecisions c) m -
          Junpei.FiniteGaussian.maxAll m
      rw [Draft0.BayesianModel.recommend_max m, maxAll_eq_terminal m]
  | succ H ih =>
      have hremaining : T - t = H := by omega
      let a := Draft0.BayesianModel.choose H c m
      let next : ℝ → Junpei.FiniteGaussian.Strategy (A := Arm) H := fun y =>
        rawTree (deterministicBellman T) (t + 1) H
          (Draft0.BayesianModel.advance c a)
          (Draft0.BayesianModel.shift m a ((y - m a) / ((c a : ℝ) + 1))) (seedTail u)
      have htree : rawTree (deterministicBellman T) t (H + 1) c m u =
          .sample a next := by
        simp only [rawTree, deterministicBellman, hremaining]
        rfl
      have hreg : Regular (.sample a next) := by
        rw [← htree]
        exact rawTree_regular (deterministicBellman T) t (H + 1) c m u
      have hca : ∀ b, 0 < Draft0.BayesianModel.advance c a b := by
        intro b
        have hb := hc b
        simp only [Draft0.BayesianModel.advance]
        omega
      have hrecover (z : ℝ) :
          (m a + (((natPrecisions c a : ℝ≥0) : ℝ) + 1) * z - m a) /
            ((c a : ℝ) + 1) = z := by
        simpa only [natPrecisions, NNReal.coe_natCast] using
          recovered_innovation (natPrecisions c a) (m a) z
      rw [htree, rawRisk_succ a next (natPrecisions c) m
        (natPrecisions_positive c hc a) hreg.2]
      calc
        _ = ∫ z : ℝ,
            Junpei.FinitePosterior.loss H
              (Junpei.FiniteGaussian.advance (natPrecisions c) a)
              (Junpei.FiniteGaussian.shift m a z)
              ∂gaussianReal 0 (Junpei.FiniteGaussian.innovation (natPrecisions c) a) := by
          apply integral_congr_ae
          apply Filter.Eventually.of_forall
          intro z
          simp only [next, hrecover, advance_natPrecisions, shift_eq]
          exact ih (t + 1) (Draft0.BayesianModel.advance c a)
            (Draft0.BayesianModel.shift m a z) (seedTail u) (by omega) hca
        _ = Junpei.FinitePosterior.loss (H + 1) (natPrecisions c) m := by
          change Junpei.FinitePosterior.actionLoss H (natPrecisions c) m a = _
          exact (CountModelBridge.actionLoss_eq_loss_iff H c m a (hc a)).mpr
            (Draft0.BayesianModel.choose_optimal H c m)

end
end Junpei.PolicyContinuationOptimal
