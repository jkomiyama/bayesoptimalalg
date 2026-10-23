import Junpei.JointGaussianMeasurable
import Junpei.InitialOptimalHistory

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedSectionVars false
set_option maxHeartbeats 1000000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.RealPrecisionPolicy

open Junpei.FiniteGaussian Junpei.FinitePosterior
open Junpei.PolicyContinuation Junpei.EmbeddingRaw Junpei.EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

/-- Posterior-state rules on the full real-precision Gaussian state space.
The draw index is the number of samples remaining after that draw. A finite
horizon converts this index to the paper's calendar stage. -/
structure Policy (A : Type*) [MeasurableSpace A] where
  draw : ℕ → Precisions A → Means A → ℝ → A
  recommend : Precisions A → Means A → ℝ → A
  draw_measurable : ∀ k, Measurable (fun p : (Precisions A × Means A) × ℝ =>
    draw k p.1.1 p.1.2 p.2)
  recommend_measurable : Measurable (fun p : (Precisions A × Means A) × ℝ =>
    recommend p.1.1 p.1.2 p.2)

/-- Actual raw-observation execution consumes only a fresh seed and the
chosen arm's next reward. Precisions and means undergo the conjugate update. -/
def rawTree (π : Policy A) : (H : ℕ) → Precisions A → Means A → Seeds H → Strategy (A := A) H
  | 0, c, m, u => .stop (π.recommend c m (u ⟨0, by omega⟩))
  | H + 1, c, m, u =>
      let a := π.draw H c m (u ⟨0, by omega⟩)
      .sample a (fun y => rawTree π H (advance c a)
        (shift m a ((y - m a) / ((c a : ℝ) + 1))) (seedTail u))

abbrev Input (A : Type*) (H : ℕ) :=
  (Precisions A × Means A) × (Seeds H × RawStream A H)

theorem updatedState_measurable (H : ℕ) (a : A) :
    Measurable (fun p : Input A (H + 1) =>
      (advance p.1.1 a, shift p.1.2 a
        ((p.2.2 a ⟨0, by omega⟩ - p.1.2 a) / ((p.1.1 a : ℝ) + 1)))) := by
  apply Measurable.prodMk
  · exact (JointGaussianMeasurable.advance_measurable a).comp
      (measurable_fst.comp measurable_fst)
  · have hz : Measurable (fun p : Input A (H + 1) =>
        (p.2.2 a ⟨0, by omega⟩ - p.1.2 a) / ((p.1.1 a : ℝ) + 1)) := by fun_prop
    exact (JointGaussianMeasurable.shift_joint_measurable a).comp
      ((measurable_snd.comp measurable_fst).prodMk hz)

/-- The entire interpreter is jointly Borel in all real precisions, means,
fresh seeds, and raw observations. -/
theorem rawTree_run_measurable (π : Policy A) (H : ℕ) :
    Measurable (fun p : Input A H => run (rawTree π H p.1.1 p.1.2 p.2.1) p.2.2) := by
  induction H with
  | zero =>
    exact π.recommend_measurable.comp (measurable_fst.prodMk
      ((measurable_pi_apply (⟨0, by omega⟩ : Fin 1)).comp
        (measurable_fst.comp measurable_snd)))
  | succ H ih =>
    let g : Input A (H + 1) × A → A := fun p =>
      run (rawTree π H (advance p.1.1.1 p.2)
        (shift p.1.1.2 p.2 ((p.1.2.2 p.2 ⟨0, by omega⟩ - p.1.1.2 p.2) /
          ((p.1.1.1 p.2 : ℝ) + 1))) (seedTail p.1.2.1)) (tail p.1.2.2)
    have hg : Measurable g := by
      apply measurable_from_prod_countable_left
      intro a
      have hin : Measurable (fun p : Input A (H + 1) =>
          ((advance p.1.1 a, shift p.1.2 a
            ((p.2.2 a ⟨0, by omega⟩ - p.1.2 a) / ((p.1.1 a : ℝ) + 1))),
              (seedTail p.2.1, tail p.2.2))) := by
        apply (updatedState_measurable H a).prodMk
        unfold seedTail tail
        fun_prop
      exact ih.comp hin
    have ha : Measurable (fun p : Input A (H + 1) =>
        π.draw H p.1.1 p.1.2 (p.2.1 ⟨0, by omega⟩)) :=
      (π.draw_measurable H).comp (measurable_fst.prodMk
        ((measurable_pi_apply (⟨0, by omega⟩ : Fin (H + 2))).comp
          (measurable_fst.comp measurable_snd)))
    exact hg.comp (measurable_id.prodMk ha)

theorem rawTree_regular (π : Policy A) (H : ℕ) (c : Precisions A) (m : Means A)
    (u : Seeds H) : Regular (rawTree π H c m u) := by
  induction H generalizing c m with
  | zero => trivial
  | succ H ih =>
    constructor
    · intro y
      exact ih _ _ (seedTail u)
    · let a := π.draw H c m (u ⟨0, by omega⟩)
      have hin : Measurable (fun p : ℝ × RawStream A H =>
          ((advance c a, shift m a ((p.1 - m a) / ((c a : ℝ) + 1))), (seedTail u, p.2))) := by
        apply Measurable.prodMk
        · apply (measurable_const : Measurable (fun _ : ℝ × RawStream A H => advance c a)).prodMk
          have hz : Measurable (fun p : ℝ × RawStream A H =>
              ((p.1 - m a) / ((c a : ℝ) + 1))) := by fun_prop
          exact (JointGaussianMeasurable.shift_joint_measurable a).comp
            ((measurable_const : Measurable (fun _ : ℝ × RawStream A H => m)).prodMk hz)
        · exact measurable_const.prodMk measurable_snd
      exact (rawTree_run_measurable π H).comp hin

theorem rawTree_family_measurable (π : Policy A) (H : ℕ) (c : Precisions A) (m : Means A) :
    Measurable (fun p : Seeds H × RawStream A H => run (rawTree π H c m p.1) p.2) := by
  have hin : Measurable (fun p : Seeds H × RawStream A H => ((c, m), p)) :=
    measurable_const.prodMk measurable_id
  exact (rawTree_run_measurable π H).comp hin

theorem rawTree_budget (π : Policy A) (H : ℕ) (c : Precisions A) (m : Means A)
    (u : Seeds H) (r : RawStream A H) : (trace (rawTree π H c m u) r).length = H :=
  trace_budget _ _

/-- Actual posterior expected simple regret, including independent scalar
Gaussian policy seeds and the actual latent-parameter/noise execution. -/
def actualRisk (π : Policy A) (H : ℕ) (c : Precisions A) (m : Means A) : ℝ :=
  ∫ u, rawRisk (rawTree π H c m u) c m ∂Junpei.Policy.seedLaw (H + 1)

theorem actualRisk_eq_joint (π : Policy A) (H : ℕ) (c : Precisions A) (m : Means A) :
    actualRisk π H c m =
      ∫ p : Seeds H × (Means A × RawStream A H),
        maxAll p.2.1 - p.2.1 (run (rawTree π H c m p.1) (observations p.2))
        ∂(Junpei.Policy.seedLaw (H + 1)).prod (worldLaw H c m) := by
  exact (integral_prod _ (EmbeddingRandomized.randomizedPayoff_integrable
    (Junpei.Policy.seedLaw (H + 1)) (fun u => rawTree π H c m u) c m
      (rawTree_family_measurable π H c m))).symm

section Ordered
variable [LinearOrder A]

/-- One globally Borel policy, with the exact minimum-index optimal action
and recommendation at every Gaussian state and every remaining horizon. -/
def canonical : Policy A where
  draw := fun k c m _ => JointGaussianMeasurable.minimumOptimal k c m
  recommend := fun _ m _ => MockTie.minimumMaximizer m
  draw_measurable := fun k =>
    (JointGaussianMeasurable.minimumOptimal_joint_measurable k).comp measurable_fst
  recommend_measurable := MockTie.minimumMaximizer_measurable.comp
    (measurable_snd.comp measurable_fst)

/-- Every fixed-seed execution of the same global real-precision policy
attains the actual posterior loss. The Gaussian observation disintegration
is invoked through rawRisk_succ, rather than assumed. -/
theorem canonical_rawRisk (H : ℕ) (c : Precisions A) (m : Means A)
    (u : Seeds H) (hc : ∀ a, 0 < c a) :
    rawRisk (rawTree (canonical (A := A)) H c m u) c m = loss H c m := by
  induction H generalizing c m with
  | zero =>
    simp only [rawTree, canonical, rawRisk_zero, terminalRegret_eq,
      MockTie.minimumMaximizer_attains, loss, value]
  | succ H ih =>
    let a := JointGaussianMeasurable.minimumOptimal H c m
    let next : ℝ → Strategy (A := A) H := fun y =>
      rawTree canonical H (advance c a)
        (shift m a ((y - m a) / ((c a : ℝ) + 1))) (seedTail u)
    have htree : rawTree (canonical (A := A)) (H + 1) c m u = .sample a next := rfl
    have hreg : Regular (.sample a next) := by
      rw [← htree]
      exact rawTree_regular canonical (H + 1) c m u
    rw [htree, rawRisk_succ a next c m (hc a) hreg.2]
    calc
      _ = ∫ z, loss H (advance c a) (shift m a z) ∂gaussianReal 0 (innovation c a) := by
        apply integral_congr_ae
        apply Filter.Eventually.of_forall
        intro z
        simp only [next, recovered_innovation]
        exact ih _ _ (seedTail u) (advance_positive c a hc)
      _ = loss (H + 1) c m :=
        JointGaussianMeasurable.minimumOptimal_actionLoss H c m hc

theorem canonical_actualRisk (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) : actualRisk (canonical (A := A)) H c m = loss H c m := by
  unfold actualRisk
  simp only [canonical_rawRisk H c m _ hc]
  simp

/-- Sequential posterior optimality against all concrete Borel history
algorithms, including arbitrary dependence on observed histories and past
randomizers. States here have arbitrary positive REAL precisions. -/
def SequentiallyOptimal (π : Policy A) : Prop :=
  ∀ (H : ℕ) (c : Precisions A) (m : Means A), (∀ a, 0 < c a) →
    ∀ (ρ : HistoryPolicy.Policy A) (n : ℕ) (h : HistoryPolicy.History A n),
      actualRisk π H c m ≤ HistoryPolicy.actualRisk ρ n H h c m

theorem canonical_sequentiallyOptimal : SequentiallyOptimal (canonical (A := A)) := by
  intro H c m hc ρ n h
  rw [canonical_actualRisk H c m hc]
  exact HistoryPolicy.randomized_history_risk_ge_loss ρ n H h c m hc

/-- Existence of a globally Borel sequentially optimal Gaussian-state
policy, not a family of selectors chosen separately at fixed precisions. -/
theorem exists_globally_borel_sequentiallyOptimal :
    ∃ π : Policy A, SequentiallyOptimal π :=
  ⟨canonical, canonical_sequentiallyOptimal⟩

end Ordered
end
end Junpei.RealPrecisionPolicy
