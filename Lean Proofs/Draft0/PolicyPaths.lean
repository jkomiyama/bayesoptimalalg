import Draft0.BayesianModel
import Draft0.PosteriorRegret
import Draft0.MainBridge
import Mathlib.MeasureTheory.Constructions.Pi

open MeasureTheory ProbabilityTheory

namespace Draft0.PolicyPaths

set_option maxHeartbeats 1000000

open BayesianModel

abbrev Path := Arm → ℕ → ℝ

noncomputable section

def meansAt (p : Path) (c : Counts) : Means := fun i => p i (c i)

def afterInitial (T : ℕ) (p : Path) : ℕ → Counts
  | 0 => fun _ => 1
  | r + 1 =>
    let c := afterInitial T p r
    advance c (choose (T - (r + 4)) c (meansAt p c))

def counts (T : ℕ) (p : Path) (s : ℕ) : Counts := afterInitial T p (s - 3)

def selected (T : ℕ) (p : Path) (t : ℕ) : Arm :=
  choose (T - t) (counts T p (t - 1)) (meansAt p (counts T p (t - 1)))

def recommendation (T : ℕ) (p : Path) : Arm := recommend (meansAt p (counts T p T))

theorem counts_initial (T : ℕ) (p : Path) (i : Arm) : counts T p 3 i = 1 := rfl

theorem counts_update (T : ℕ) (p : Path) (s : ℕ) (hs : 3 ≤ s) :
    counts T p (s + 1) = advance (counts T p s) (selected T p (s + 1)) := by
  have hindex : s + 1 - 3 = (s - 3) + 1 := by omega
  have htime : (s - 3) + 4 = s + 1 := by omega
  simp only [counts, hindex, afterInitial, htime, selected, Nat.add_sub_cancel]

theorem counts_positive (T : ℕ) (p : Path) (s : ℕ) (i : Arm) :
    0 < counts T p s i := by
  have h (r : ℕ) : ∀ i : Arm, 0 < afterInitial T p r i := by
    induction r with
    | zero => intro j; norm_num [afterInitial]
    | succ r ih =>
      intro j
      simp only [afterInitial, advance]
      exact lt_of_lt_of_le (ih j) (Nat.le_add_right _ _)
  exact h (s - 3) i

theorem selected_optimal (T : ℕ) (p : Path) (t : ℕ) (i : Arm) :
    action (T - t) (counts T p (t - 1)) (meansAt p (counts T p (t - 1))) i ≤
    action (T - t) (counts T p (t - 1)) (meansAt p (counts T p (t - 1)))
      (selected T p t) := by
  unfold selected
  rw [choose_optimal]
  exact action_le_value _ _ _ _

theorem selected_minimizes_regret (T : ℕ) (p : Path) (t : ℕ) (i : Arm) :
    PosteriorRegret.actionLoss (T - t) (counts T p (t - 1))
      (meansAt p (counts T p (t - 1))) (selected T p t) ≤
    PosteriorRegret.actionLoss (T - t) (counts T p (t - 1))
      (meansAt p (counts T p (t - 1))) i :=
  PosteriorRegret.chosen_loss_minimal _ _ _ (counts_positive T p (t - 1)) i

theorem recommendation_greedy (T : ℕ) (p : Path) (i : Arm) :
    p i (counts T p T i) ≤ p (recommendation T p) (counts T p T (recommendation T p)) :=
  recommend_greedy (meansAt p (counts T p T)) i

theorem choose_measurable (k : ℕ) (c : Counts) : Measurable (choose k c) := by
  unfold choose
  exact Measurable.ite
    (measurableSet_le ((action_lipschitz k c 1).continuous.max
      (action_lipschitz k c 2).continuous).measurable
      (action_lipschitz k c 0).continuous.measurable)
    measurable_const
    (Measurable.ite
      (measurableSet_le (action_lipschitz k c 2).continuous.measurable
        (action_lipschitz k c 1).continuous.measurable)
      measurable_const measurable_const)

theorem recommend_measurable : Measurable recommend := by
  unfold recommend
  exact Measurable.ite
    (measurableSet_le ((continuous_apply 1).max (continuous_apply 2)).measurable
      (continuous_apply 0).measurable)
    measurable_const
    (Measurable.ite (measurableSet_le (measurable_pi_apply 2) (measurable_pi_apply 1))
      measurable_const measurable_const)

theorem meansAt_measurable (c : Counts) : Measurable (meansAt · c) := by
  apply measurable_pi_lambda
  intro i
  exact (measurable_pi_apply (c i)).comp (measurable_pi_apply i)

theorem meansAt_joint_measurable :
    Measurable (fun x : Counts × Path => meansAt x.2 x.1) :=
  measurable_from_prod_countable_right meansAt_measurable

theorem choose_joint_measurable (k : ℕ) :
    Measurable (fun x : Counts × Means => choose k x.1 x.2) :=
  measurable_from_prod_countable_right (choose_measurable k)

theorem afterInitial_measurable (T r : ℕ) : Measurable (afterInitial T · r) := by
  induction r with
  | zero => exact measurable_const
  | succ r ih =>
    have hm : Measurable (fun p : Path => meansAt p (afterInitial T p r)) :=
      meansAt_joint_measurable.comp (ih.prodMk measurable_id)
    have hp : Measurable (fun p : Path =>
        (afterInitial T p r, meansAt p (afterInitial T p r))) := ih.prodMk hm
    have ha := (choose_joint_measurable (T - (r + 4))).comp hp
    change Measurable (fun p : Path =>
      choose (T - (r + 4)) (afterInitial T p r) (meansAt p (afterInitial T p r))) at ha
    exact (measurable_of_countable (fun x : Counts × Arm => advance x.1 x.2)).comp
      (ih.prodMk ha)

theorem counts_measurable (T s : ℕ) : Measurable (counts T · s) :=
  afterInitial_measurable T (s - 3)

theorem recommendation_measurable (T : ℕ) : Measurable (recommendation T) :=
  recommend_measurable.comp
    (meansAt_joint_measurable.comp ((counts_measurable T T).prodMk measurable_id))

theorem wrong_measurable (T : ℕ) : MeasurableSet {p : Path | recommendation T p ≠ 2} :=
  ((recommendation_measurable T) (measurableSet_singleton 2)).compl

end

end Draft0.PolicyPaths
