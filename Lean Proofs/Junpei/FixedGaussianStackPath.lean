import Junpei.FixedGaussianStackExecution
import Junpei.Empirical
import Junpei.Main

set_option autoImplicit false
set_option warningAsError true
set_option maxHeartbeats 1000000

open MeasureTheory ProbabilityTheory

namespace Junpei.FixedGaussian.StackPath

open Draft0.BayesianModel Junpei.Policy Junpei.PolicyContinuation
open Junpei.EmbeddingRaw Junpei.Empirical

noncomputable section

def pathSeeds (u : RandomPath) (r H : ℕ) : Seeds H :=
  fun j => u (r + 3 + j.val)

def remainingRewards {T : ℕ} (Δ : ℝ) (ω : Draft0.BadEvent.RawArrays T)
    (c : Counts) (H : ℕ) (hb : ∀ a, c a + H ≤ T) : RawStream Arm H :=
  fun a j => observation Δ ω a ⟨c a + j.val, by have := hb a; omega⟩

theorem afterInitial_count_le (π : StatePolicy) (p : Path) (u : RandomPath)
    (r : ℕ) (a : Arm) : afterInitial π p u r a ≤ r + 1 := by
  induction r with
  | zero => exact le_rfl
  | succ r ih =>
    simp only [afterInitial, advance]
    split_ifs <;> omega

theorem afterInitial_count_pos (π : StatePolicy) (p : Path) (u : RandomPath)
    (r : ℕ) (a : Arm) : 0 < afterInitial π p u r a := by
  induction r with
  | zero => exact Nat.zero_lt_one
  | succ r ih =>
    simp only [afterInitial, advance]
    split_ifs <;> omega

theorem remainingRewards_tail {T : ℕ} (Δ : ℝ) (ω : Draft0.BadEvent.RawArrays T)
    (c : Counts) (a : Arm) (H : ℕ) (hb : ∀ b, c b + (H + 1) ≤ T)
    (hn : ∀ b, advance c a b + H ≤ T) :
    stackTail a (remainingRewards Δ ω c (H + 1) hb) =
      remainingRewards Δ ω (advance c a) H hn := by
  funext b j
  unfold stackTail remainingRewards advance
  by_cases hba : b = a
  · simp only [if_pos hba]
    congr 2
    simp only [Fin.val_succ]
    omega
  · simp only [if_neg hba]
    rfl

theorem meansAt_empirical_advance {T : ℕ} (Δ : ℝ)
    (ω : Draft0.BadEvent.RawArrays T) (c : Counts) (a : Arm)
    (hpos : 0 < c a) (hbound : c a < T) :
    Draft0.PolicyPaths.meansAt (empiricalPath Δ ω) (advance c a) =
      shift (Draft0.PolicyPaths.meansAt (empiricalPath Δ ω) c) a
        ((observation Δ ω a ⟨c a, hbound⟩ -
          Draft0.PolicyPaths.meansAt (empiricalPath Δ ω) c a) / ((c a : ℝ) + 1)) := by
  funext b
  by_cases hba : b = a
  · subst b
    simp only [Draft0.PolicyPaths.meansAt, advance, shift, if_true]
    rw [empiricalPath_update Δ ω a (c a) hpos hbound]
    unfold Draft0.Gaussian.posteriorMeanUpdate
    field_simp
    ring
  · simp only [Draft0.PolicyPaths.meansAt, advance, shift, if_neg hba, add_zero]

/-- Deterministic path identity, including every adaptive per-arm pull index,
posterior mean update, and policy-randomizer index. No probability assertion
or agreement-of-execution assumption occurs in this statement. -/
theorem rawTree_stack_recommendation (π : StatePolicy) (T : ℕ) (Δ : ℝ)
    (ω : Draft0.BadEvent.RawArrays T) (u : RandomPath) (r H : ℕ)
    (hbudget : r + H + 3 ≤ T) :
    stackRun (rawTree π (r + 4) H
      (afterInitial π (empiricalPath Δ ω) u r)
      (Draft0.PolicyPaths.meansAt (empiricalPath Δ ω)
        (afterInitial π (empiricalPath Δ ω) u r)) (pathSeeds u r H))
      (remainingRewards Δ ω (afterInitial π (empiricalPath Δ ω) u r) H
        (fun a => by have := afterInitial_count_le π (empiricalPath Δ ω) u r a; omega)) =
      π.recommend (afterInitial π (empiricalPath Δ ω) u (r + H))
        (Draft0.PolicyPaths.meansAt (empiricalPath Δ ω)
          (afterInitial π (empiricalPath Δ ω) u (r + H))) (u (r + H + 3)) := by
  induction H generalizing r with
  | zero => simp only [rawTree, stackRun, pathSeeds, Nat.add_zero]
  | succ H ih =>
    let p := empiricalPath Δ ω
    let c := afterInitial π p u r
    let m := Draft0.PolicyPaths.meansAt p c
    let a := π.draw (r + 4) c m (u (r + 3))
    have hb : ∀ b, c b + (H + 1) ≤ T := by
      intro b
      have := afterInitial_count_le π p u r b
      change c b ≤ r + 1 at this
      omega
    have hn : ∀ b, advance c a b + H ≤ T := by
      intro b
      have := hb b
      unfold advance
      split_ifs <;> omega
    have hnext : afterInitial π p u (r + 1) = advance c a := rfl
    have hm : shift m a
        ((observation Δ ω a ⟨c a, by have := hb a; omega⟩ - m a) / ((c a : ℝ) + 1)) =
        Draft0.PolicyPaths.meansAt p (afterInitial π p u (r + 1)) := by
      rw [hnext]
      exact (meansAt_empirical_advance Δ ω c a
        (afterInitial_count_pos π p u r a) (by have := hb a; omega)).symm
    have hseeds : seedTail (pathSeeds u r (H + 1)) = pathSeeds u (r + 1) H := by
      funext j
      unfold seedTail pathSeeds
      congr 1
      simp only [Fin.val_succ]
      omega
    have htail := remainingRewards_tail Δ ω c a H hb hn
    change stackRun (rawTree π ((r + 4) + 1) H (advance c a)
      (shift m a ((observation Δ ω a ⟨c a, by have := hb a; omega⟩ - m a) /
        ((c a : ℝ) + 1))) (seedTail (pathSeeds u r (H + 1))))
      (stackTail a (remainingRewards Δ ω c (H + 1) hb)) = _
    rw [hm, hseeds, htail]
    have hi := ih (r + 1) (by omega)
    have ht : r + 4 + 1 = r + 1 + 4 := by omega
    have hfin : r + (H + 1) = r + 1 + H := by omega
    simpa only [ht, hfin] using hi

def initialMeans (T : ℕ) (Δ : ℝ) (hT : 3 ≤ T)
    (ω : Draft0.BadEvent.RawArrays T) : Means :=
  fun a => observation Δ ω a ⟨0, by omega⟩

def seedWindow (T : ℕ) (hT : 3 ≤ T) (u : SeedArray (T + 1)) : Seeds (T - 3) :=
  fun j => u ⟨j.val + 3, by have := j.isLt; omega⟩

def remainingStack (T : ℕ) (Δ : ℝ) (hT : 3 ≤ T)
    (ω : Draft0.BadEvent.RawArrays T) : RawStream Arm (T - 3) :=
  fun a j => observation Δ ω a ⟨j.val + 1, by have := j.isLt; omega⟩

def initializedInput (T : ℕ) (Δ : ℝ) (hT : 3 ≤ T)
    (w : Junpei.Main.JointSample T) : Input (T - 3) :=
  (initialMeans T Δ hT w.1, (seedWindow T hT w.2, remainingStack T Δ hT w.1))

/-- Literal equivalence between Main's reward-stack execution and the raw
continuation tree started after the common one-sample-per-arm initialization. -/
theorem recommendation_eq_stackRun (π : StatePolicy) (T : ℕ) (hT : 3 ≤ T)
    (Δ : ℝ) (ω : Draft0.BadEvent.RawArrays T) (u : SeedArray (T + 1)) :
    recommendation π T (empiricalPath Δ ω) (randomPath T u) =
      stackRun (rawTree π 4 (T - 3) (fun _ => 1) (initialMeans T Δ hT ω)
        (seedWindow T hT u)) (remainingStack T Δ hT ω) := by
  have hi := rawTree_stack_recommendation π T Δ ω (randomPath T u) 0 (T - 3) (by omega)
  have hm : Draft0.PolicyPaths.meansAt (empiricalPath Δ ω) (fun _ => 1) =
      initialMeans T Δ hT ω := by
    funext a
    simp only [Draft0.PolicyPaths.meansAt, empiricalPath, initialMeans, observation]
    rw [Draft0.BadEvent.sampleMean_one (by omega)]
  have hu : pathSeeds (randomPath T u) 0 (T - 3) = seedWindow T hT u := by
    funext j
    unfold pathSeeds randomPath extendSeeds seedWindow
    rw [dif_pos (by have := j.isLt; omega)]
    congr 1
    apply Fin.ext
    change 0 + 3 + j.val = j.val + 3
    omega
  have hr : remainingRewards Δ ω (fun _ => 1) (T - 3)
      (fun _ => by change 1 + (T - 3) ≤ T; omega) = remainingStack T Δ hT ω := by
    funext a j
    unfold remainingRewards remainingStack
    congr 2
    change 1 + j.val = j.val + 1
    omega
  simp only [Nat.zero_add, afterInitial, hm, hu, hr] at hi
  unfold recommendation counts
  have hlast : randomPath T u (T - 3 + 3) = randomPath T u T := by congr 1; omega
  rw [hlast] at hi
  exact hi.symm

theorem actualRecommendation_eq_stackRun (π : StatePolicy) (T : ℕ) (hT : 3 ≤ T)
    (Δ : ℝ) (ω : Junpei.Main.JointSample T) :
    Junpei.Main.actualRecommendation π T Δ ω =
      stackRun (rawTree π 4 (T - 3) (fun _ => 1) (initialMeans T Δ hT ω.1)
        (seedWindow T hT ω.2)) (remainingStack T Δ hT ω.1) :=
  recommendation_eq_stackRun π T hT Δ ω.1 ω.2

end
end Junpei.FixedGaussian.StackPath
