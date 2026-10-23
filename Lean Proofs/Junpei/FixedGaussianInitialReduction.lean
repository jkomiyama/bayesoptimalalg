import Junpei.FixedGaussianTransfer
import Junpei.InitialOptimalPaths

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory

namespace Junpei.FixedGaussian

open Draft0.BayesianModel Junpei.Policy Junpei.PolicyContinuation Junpei.SequentialRisk
open Junpei.CountModelBridge Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open Junpei.InitialOptimal

noncomputable section

/-- Actual-risk minimization at one initial state suffices for agreement with
the sequentially optimal repair under every fixed true mean, not merely under
the Gaussian posterior mixture used in the optimization criterion. -/
theorem minimizing_rawTree_repair_fixed_ae (π : StatePolicy) (T t H : ℕ)
    (c : Counts) (m μ : Means) (horizon : t + H = T + 1) (ht : 4 ≤ t)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t H c m) :
    ∀ᵐ p ∂(seedLaw (H + 1)).prod (fixedLaw H μ),
      run (rawTree π t H c m p.1) p.2 =
        run (rawTree (repair π T) t H c m p.1) p.2 := by
  apply ae_execution_transfer (seedLaw (H + 1)) H (natPrecisions c) m μ
    (fun p => run (rawTree π t H c m p.1) p.2 =
      run (rawTree (repair π T) t H c m p.1) p.2)
  · apply measurableSet_eq_fun_of_countable
    · exact (rawTree_feasible π t H c m).2
    · exact (rawTree_feasible (repair π T) t H c m).2
  · exact minimizing_rawTree_repair_ae π T t H c m horizon ht hc hπ

theorem initialRiskOptimal_repair_fixed_ae (π : StatePolicy) (T : ℕ) (hT : 3 ≤ T)
    (hπ : InitialRiskOptimal π T) (m μ : Means) :
    ∀ᵐ p ∂(seedLaw (T - 3 + 1)).prod (fixedLaw (T - 3) μ),
      run (rawTree π 4 (T - 3) (fun _ => 1) m p.1) p.2 =
        run (rawTree (repair π T) 4 (T - 3) (fun _ => 1) m p.1) p.2 := by
  exact minimizing_rawTree_repair_fixed_ae π T 4 (T - 3) (fun _ => 1) m μ
    (by omega) (by omega) (by intro a; norm_num) (hπ m)

theorem initialRiskOptimal_fixed_payoff_repair_eq (π : StatePolicy) (T : ℕ)
    (hT : 3 ≤ T) (hπ : InitialRiskOptimal π T) (m μ : Means) (L : Arm → ℝ) :
    (∫ p, L (run (rawTree π 4 (T - 3) (fun _ => 1) m p.1) p.2)
      ∂(seedLaw (T - 3 + 1)).prod (fixedLaw (T - 3) μ)) =
    ∫ p, L (run (rawTree (repair π T) 4 (T - 3) (fun _ => 1) m p.1) p.2)
      ∂(seedLaw (T - 3 + 1)).prod (fixedLaw (T - 3) μ) := by
  apply integral_congr_ae
  exact (initialRiskOptimal_repair_fixed_ae π T hT hπ m μ).mono
    (fun _ hp => congrArg L hp)

end
end Junpei.FixedGaussian
