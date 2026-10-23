import Junpei.InitialOptimalHistory

set_option autoImplicit false
set_option warningAsError true
set_option maxHeartbeats 1000000

open MeasureTheory ProbabilityTheory

namespace Junpei.PosteriorStateHistory

open Draft0.BayesianModel
open Junpei.Policy Junpei.PolicyContinuation Junpei.CountModelBridge
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution Junpei.SequentialRisk

noncomputable section

def updateData (ca : Counts × Arm) (my : Means × ℝ) : Counts × Means :=
  (advance ca.1 ca.2, shift my.1 ca.2 ((my.2 - my.1 ca.2) / ((ca.1 ca.2 : ℝ) + 1)))

theorem updateData_measurable :
    Measurable (fun p : (Counts × Arm) × (Means × ℝ) => updateData p.1 p.2) := by
  apply measurable_from_prod_countable_right
  intro ca
  change Measurable (fun y : Means × ℝ =>
    (advance ca.1 ca.2, shift y.1 ca.2 ((y.2 - y.1 ca.2) / ((ca.1 ca.2 : ℝ) + 1))))
  refine (measurable_const : Measurable (fun _ : Means × ℝ => advance ca.1 ca.2)).prodMk ?_
  apply measurable_pi_lambda
  intro a
  unfold shift
  split_ifs <;> fun_prop

/-- Fold only observed arm/reward pairs into the sufficient statistic.
The recorded past randomizers are available but the state update ignores them. -/
def foldState (c0 : Counts) (m0 : Means) : (n : ℕ) →
    HistoryPolicy.History Arm n → Counts × Means
  | 0, _ => (c0, m0)
  | n + 1, h =>
      let p := foldState c0 m0 n (Fin.init h)
      let z := h (Fin.last n)
      updateData (p.1, z.1) (p.2, z.2.1)

theorem foldState_append (c0 : Counts) (m0 : Means) (n : ℕ)
    (h : HistoryPolicy.History Arm n) (a : Arm) (y u : ℝ) :
    foldState c0 m0 (n + 1) (HistoryPolicy.append h a y u) =
      updateData ((foldState c0 m0 n h).1, a) ((foldState c0 m0 n h).2, y) := by
  simp only [foldState, HistoryPolicy.append, Fin.init_snoc, Fin.snoc_last]

theorem foldState_measurable (c0 : Counts) (m0 : Means) (n : ℕ) :
    Measurable (foldState c0 m0 n) := by
  induction n with
  | zero => exact measurable_const
  | succ n ih =>
    have hh : Measurable (fun h : HistoryPolicy.History Arm (n + 1) => Fin.init h) := by
      unfold Fin.init
      fun_prop
    have hp := ih.comp hh
    have hz : Measurable (fun h : HistoryPolicy.History Arm (n + 1) => h (Fin.last n)) :=
      measurable_pi_apply _
    exact updateData_measurable.comp
      ((hp.fst.prodMk hz.fst).prodMk (hp.snd.prodMk hz.snd.fst))

/-- A posterior-state policy is explicitly a Borel history policy. The
compiler uses no latent parameter, unobserved reward, or future seed. -/
def compile (π : StatePolicy) (t : ℕ) (c0 : Counts) (m0 : Means) : HistoryPolicy.Policy Arm where
  draw := fun n h u =>
    let p := foldState c0 m0 n h
    π.draw (t + n) p.1 p.2 u
  recommend := fun n h u =>
    let p := foldState c0 m0 n h
    π.recommend p.1 p.2 u
  draw_measurable := by
    intro n
    have hp := (foldState_measurable c0 m0 n).comp
      (measurable_fst : Measurable (fun p : HistoryPolicy.History Arm n × ℝ => p.1))
    exact (π.draw_measurable (t + n)).comp (hp.fst.prodMk (hp.snd.prodMk measurable_snd))
  recommend_measurable := by
    intro n
    have hp := (foldState_measurable c0 m0 n).comp
      (measurable_fst : Measurable (fun p : HistoryPolicy.History Arm n × ℝ => p.1))
    exact π.recommend_measurable.comp (hp.fst.prodMk (hp.snd.prodMk measurable_snd))

/-- The compiled history tree and the original state-policy continuation
are exactly the same raw strategy, including every posterior update. -/
theorem toRaw_compile (π : StatePolicy) (t n H : ℕ) (c0 : Counts) (m0 : Means)
    (h : HistoryPolicy.History Arm n) (u : Seeds H) :
    HistoryPolicy.toRaw (compile π t c0 m0) n H h u =
      rawTree π (t + n) H (foldState c0 m0 n h).1 (foldState c0 m0 n h).2 u := by
  induction H generalizing n with
  | zero => rfl
  | succ H ih =>
    let p := foldState c0 m0 n h
    let a := π.draw (t + n) p.1 p.2 (u ⟨0, by omega⟩)
    change Junpei.FiniteGaussian.Strategy.sample a
      (fun y => HistoryPolicy.toRaw (compile π t c0 m0) (n + 1) H
        (HistoryPolicy.append h a y (u ⟨0, by omega⟩)) (seedTail u)) =
      Junpei.FiniteGaussian.Strategy.sample a
        (fun y => rawTree π (t + n + 1) H (advance p.1 a)
          (shift p.2 a ((y - p.2 a) / ((p.1 a : ℝ) + 1))) (seedTail u))
    congr 1
    funext y
    rw [ih, foldState_append]
    have ht : t + (n + 1) = t + n + 1 := by omega
    simp only [updateData, ht, p]

def emptyHistory : HistoryPolicy.History Arm 0 := Fin.elim0

theorem actualRisk_compile (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means) :
    HistoryPolicy.actualRisk (compile π t c m) 0 H emptyHistory (natPrecisions c) m =
      continuationRisk π t H c m := by
  rw [← HistoryPolicy.familyRisk_eq_actualRisk]
  unfold continuationRisk
  congr 1
  funext u
  simpa only [Nat.add_zero, foldState] using toRaw_compile π t 0 H c m emptyHistory u

/-- Comparison against the paper's concrete prefix-randomized, Borel
history policies, rather than a raw family with the entire seed array as a
parameter. -/
def MinimizesHistoryRisk (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means) : Prop :=
  ∀ ρ : HistoryPolicy.Policy Arm,
    continuationRisk π t H c m ≤ HistoryPolicy.actualRisk ρ 0 H emptyHistory (natPrecisions c) m

/-- The two minimization predicates agree. The reverse implication uses a
concrete compiled Bellman history competitor that attains the actual loss;
the larger raw-family class cannot beat that loss by its proved lower bound. -/
theorem minimizesHistoryRisk_iff (π : StatePolicy) (T t H : ℕ) (c : Counts) (m : Means)
    (horizon : t + H = T + 1) (hc : ∀ a, 0 < c a) :
    MinimizesHistoryRisk π t H c m ↔ MinimizesActualRisk π t H c m := by
  constructor
  · intro hπ s hs
    have hp := hπ (compile (deterministicBellman T) t c m)
    rw [actualRisk_compile] at hp
    have he := minimizing_risk_eq_loss (deterministicBellman T) t H c m hc
      (deterministicBellman_minimizes_actual_risk T t H c m horizon hc)
    rw [he] at hp
    exact hp.trans (familyRisk_ge_loss s c m hs hc)
  · intro hπ ρ
    exact HistoryPolicy.minimizing_actualRisk_le_history π t H c m hπ ρ 0 emptyHistory

/-- Initialization-only posterior optimality against all concrete Borel
history algorithms. Each possible initialized mean is quantified separately. -/
def InitialHistoryRiskOptimal (π : StatePolicy) (T : ℕ) : Prop :=
  ∀ m : Means, MinimizesHistoryRisk π 4 (T - 3) (fun _ => 1) m

theorem initialHistoryRiskOptimal_iff (π : StatePolicy) (T : ℕ) (hT : 3 ≤ T) :
    InitialHistoryRiskOptimal π T ↔ InitialOptimal.InitialRiskOptimal π T := by
  constructor <;> intro hπ m
  · exact (minimizesHistoryRisk_iff π T 4 (T - 3) (fun _ => 1) m
      (by omega) (by intro a; norm_num)).mp (hπ m)
  · exact (minimizesHistoryRisk_iff π T 4 (T - 3) (fun _ => 1) m
      (by omega) (by intro a; norm_num)).mpr (hπ m)

end
end Junpei.PosteriorStateHistory
