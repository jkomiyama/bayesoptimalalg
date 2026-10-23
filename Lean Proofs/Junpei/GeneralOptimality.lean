import Junpei.GeneralOptimalityHistory

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSectionVars false
set_option maxHeartbeats 1500000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.GeneralOptimality

open Junpei.FiniteGaussian Junpei.FinitePosterior
open Junpei.PolicyContinuation Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open Junpei.RealPrecisionPolicy

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A] [LinearOrder A]

/-- Actual posterior risk minimization at this single starting state,
against every concrete Borel history policy. This contains no Bellman-action
condition and no optimality assumption at other states or horizons. -/
def MinimizesHistoryRisk (π : Policy A) (H : ℕ) (c : Precisions A) (m : Means A) : Prop :=
  ∀ ρ : HistoryPolicy.Policy A,
    actualRisk π H c m ≤ HistoryPolicy.actualRisk ρ 0 H emptyHistory c m

theorem actualRisk_ge_loss (π : Policy A) (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) : loss H c m ≤ actualRisk π H c m :=
  EmbeddingRandomized.randomized_rawRisk_ge_loss (Junpei.Policy.seedLaw (H + 1))
    (rawTree π H c m) c m hc (rawTree_regular π H c m)
      (rawTree_family_measurable π H c m)

/-- The optimum is the Gaussian Bellman loss for every finite arm set and
every positive real precision vector. The upper bound uses an explicitly
compiled concrete history competitor, not an abstract selector premise. -/
theorem minimizing_actualRisk_eq_loss (π : Policy A) (H : ℕ)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a)
    (hπ : MinimizesHistoryRisk π H c m) : actualRisk π H c m = loss H c m := by
  apply le_antisymm _ (actualRisk_ge_loss π H c m hc)
  have h := hπ (compile (canonical (A := A)) H c m)
  rwa [actualRisk_compile, canonical_actualRisk H c m hc] at h

theorem minimizesHistoryRisk_iff_actualRisk_eq_loss (π : Policy A) (H : ℕ)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a) :
    MinimizesHistoryRisk π H c m ↔ actualRisk π H c m = loss H c m := by
  constructor
  · exact minimizing_actualRisk_eq_loss π H c m hc
  · intro he ρ
    rw [he]
    exact HistoryPolicy.randomized_history_risk_ge_loss ρ 0 H emptyHistory c m hc

/-- Minimum actual risk forces almost every fixed-seed raw execution to
attain that same loss. This is derived from integrability and the raw-strategy
lower bound, without assuming any action is optimal. -/
theorem minimizing_rawRisk_eq_loss_ae (π : Policy A) (H : ℕ)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a)
    (hπ : MinimizesHistoryRisk π H c m) :
    ∀ᵐ u ∂Junpei.Policy.seedLaw (H + 1),
      rawRisk (rawTree π H c m u) c m = loss H c m := by
  apply SequentialRisk.ae_eq_const_of_lower_integral _ _ _
    (EmbeddingRandomized.randomized_rawRisk_integrable (Junpei.Policy.seedLaw (H + 1))
      (rawTree π H c m) c m (rawTree_family_measurable π H c m))
    (fun u => rawRisk_ge_loss _ c m hc (rawTree_regular π H c m u))
  exact (minimizing_actualRisk_eq_loss π H c m hc hπ).le

/-- Necessity of the Bellman sampling rule for ANY policy minimizing
actual risk at this state: only a null set of its fresh scalar seeds can
choose a nonoptimal action. There is no three-arm or integer-count restriction. -/
theorem minimizing_draw_ae (π : Policy A) (H : ℕ)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a)
    (hπ : MinimizesHistoryRisk π (H + 1) c m) :
    ∀ᵐ u : ℝ ∂gaussianReal 0 1,
      action H c m (π.draw H c m u) = value (H + 1) c m := by
  have hh : ∀ᵐ u : Seeds (H + 1) ∂Junpei.Policy.seedLaw (H + 2),
      action H c m (π.draw H c m (u ⟨0, by omega⟩)) = value (H + 1) c m := by
    filter_upwards [minimizing_rawRisk_eq_loss_ae π (H + 1) c m hc hπ] with u hu
    apply (BayesOptimal.actionLoss_eq_loss_iff H c m _ (hc _)).mp
    apply le_antisymm _ (BayesOptimal.actionLoss_minimal H c m hc _)
    calc
      _ ≤ rawRisk (rawTree π (H + 1) c m u) c m :=
        rawRisk_ge_actionLoss (π.draw H c m (u ⟨0, by omega⟩)) _ c m hc
          (rawTree_regular π (H + 1) c m u)
      _ = _ := hu
  have hp : MeasurableSet {u : ℝ |
      action H c m (π.draw H c m u) = value (H + 1) c m} := by
    have hd : Measurable (π.draw H c m) := (π.draw_measurable H).comp
      ((measurable_const : Measurable (fun _ : ℝ => (c, m))).prodMk measurable_id)
    exact measurableSet_eq_fun ((measurable_of_countable (action H c m)).comp hd)
      measurable_const
  exact SequentialRisk.seed_head_ae (H + 1) _ hp hh

/-- Theorem 1's sampling rule, with EBI as in Eq. (9), at a single state
and for arbitrary K: any posterior-risk
minimizer samples an arm maximizing expected Bellman improvement almost surely. -/
theorem minimizing_draw_EBI_ae (π : Policy A) (H : ℕ)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a)
    (hπ : MinimizesHistoryRisk π (H + 1) c m) :
    ∀ᵐ u : ℝ ∂gaussianReal 0 1,
      ∀ a, BayesOptimal.EBI H c m a ≤ BayesOptimal.EBI H c m (π.draw H c m u) := by
  filter_upwards [minimizing_draw_ae π H c m hc hπ] with u hu
  exact (BayesOptimal.optimal_action_iff_EBI_max H c m _ hc).mp hu

/-- Theorem 1's terminal rule, using the loss of Eq. (6), for arbitrary K: every actual-risk
minimizer recommends a largest posterior mean almost surely. -/
theorem minimizing_recommend_ae (π : Policy A)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a)
    (hπ : MinimizesHistoryRisk π 0 c m) :
    ∀ᵐ u : ℝ ∂gaussianReal 0 1, m (π.recommend c m u) = maxAll m := by
  have hh : ∀ᵐ u : Seeds 0 ∂Junpei.Policy.seedLaw 1,
      m (π.recommend c m (u ⟨0, by omega⟩)) = maxAll m := by
    filter_upwards [minimizing_rawRisk_eq_loss_ae π 0 c m hc hπ] with u hu
    simp only [RealPrecisionPolicy.rawTree, rawRisk_zero, terminalRegret_eq, loss, value] at hu
    linarith
  have hp : MeasurableSet {u : ℝ | m (π.recommend c m u) = maxAll m} := by
    have hd : Measurable (π.recommend c m) := π.recommend_measurable.comp
      ((measurable_const : Measurable (fun _ : ℝ => (c, m))).prodMk measurable_id)
    exact measurableSet_eq_fun ((measurable_of_countable m).comp hd) measurable_const
  exact SequentialRisk.seed_head_ae 0 _ hp hh

/-- Sequential risk optimality restricted to one finite horizon N. Only
the remaining horizons 0 through N are quantified, matching a fixed-budget
algorithm; no optimality at larger horizons is assumed. -/
def SequentiallyOptimalUpTo (π : Policy A) (N : ℕ) : Prop :=
  ∀ (H : ℕ), H ≤ N → ∀ (c : Precisions A) (m : Means A), (∀ a, 0 < c a) →
    MinimizesHistoryRisk π H c m

/-- Theorem 1's necessity and risk identity, for arbitrary finite arm sets,
arbitrary positive REAL precisions, and any fixed finite budget. The sampling
and recommendation conclusions are derived from actual-risk optimality. -/
theorem theorem1 (π : Policy A) (N : ℕ) (hπ : SequentiallyOptimalUpTo π N) :
    (∀ H, H ≤ N → ∀ c m, (∀ a, 0 < c a) → actualRisk π H c m = loss H c m) ∧
    (∀ H, H < N → ∀ c m, (∀ a, 0 < c a) →
      ∀ᵐ u : ℝ ∂gaussianReal 0 1,
        ∀ a, BayesOptimal.EBI H c m a ≤ BayesOptimal.EBI H c m (π.draw H c m u)) ∧
    (∀ c m, (∀ a, 0 < c a) →
      ∀ᵐ u : ℝ ∂gaussianReal 0 1, m (π.recommend c m u) = maxAll m) := by
  refine ⟨?_, ?_, ?_⟩
  · intro H hH c m hc
    exact minimizing_actualRisk_eq_loss π H c m hc (hπ H hH c m hc)
  · intro H hH c m hc
    exact minimizing_draw_EBI_ae π H c m hc (hπ (H + 1) (by omega) c m hc)
  · intro c m hc
    exact minimizing_recommend_ae π c m hc (hπ 0 (Nat.zero_le N) c m hc)

theorem sequentiallyOptimal_implies_upTo (π : Policy A)
    (hπ : RealPrecisionPolicy.SequentiallyOptimal π) (N : ℕ) :
    SequentiallyOptimalUpTo π N := by
  intro H _ c m hc ρ
  exact hπ H c m hc ρ 0 emptyHistory

/-- The finite-budget class is nonempty: one globally Borel real-precision
policy simultaneously supplies a witness for every finite budget. -/
theorem exists_sequentiallyOptimalUpTo (N : ℕ) :
    ∃ π : Policy A, SequentiallyOptimalUpTo π N :=
  ⟨canonical, sequentiallyOptimal_implies_upTo canonical canonical_sequentiallyOptimal N⟩

end
end Junpei.GeneralOptimality
