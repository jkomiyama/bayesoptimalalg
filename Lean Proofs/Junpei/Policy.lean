import Draft0.PolicyPaths

open MeasureTheory ProbabilityTheory

namespace Junpei.Policy

set_option maxHeartbeats 800000

open Draft0.BayesianModel

noncomputable section

/-- Posterior-state policies can use the fresh external real randomizer. The
decision maps contain no latent parameter or unrevealed reward argument. -/
structure StatePolicy where
  draw : ℕ → Counts → Means → ℝ → Arm
  recommend : Counts → Means → ℝ → Arm
  draw_measurable : ∀ t, Measurable (fun x : Counts × Means × ℝ => draw t x.1 x.2.1 x.2.2)
  recommend_measurable : Measurable (fun x : Counts × Means × ℝ => recommend x.1 x.2.1 x.2.2)

def PointwiseOptimal (π : StatePolicy) (T : ℕ) : Prop :=
  (∀ (t : ℕ) (c : Counts) (m : Means) (u : ℝ), 4 ≤ t → t ≤ T →
    (∀ i, 0 < c i) → action (T - t) c m (π.draw t c m u) = value (T - t + 1) c m) ∧
  (∀ (c : Counts) (m : Means) (u : ℝ), (∀ i, 0 < c i) →
    m (π.recommend c m u) = terminal m)

abbrev Path := Draft0.PolicyPaths.Path
abbrev RandomPath := ℕ → ℝ

def afterInitial (π : StatePolicy) (p : Path) (u : RandomPath) : ℕ → Counts
  | 0 => fun _ => 1
  | r + 1 =>
    let c := afterInitial π p u r
    advance c (π.draw (r + 4) c (Draft0.PolicyPaths.meansAt p c) (u (r + 3)))

def counts (π : StatePolicy) (p : Path) (u : RandomPath) (s : ℕ) : Counts :=
  afterInitial π p u (s - 3)

def selected (π : StatePolicy) (p : Path) (u : RandomPath) (t : ℕ) : Arm :=
  let c := counts π p u (t - 1)
  π.draw t c (Draft0.PolicyPaths.meansAt p c) (u (t - 1))

def recommendation (π : StatePolicy) (T : ℕ) (p : Path) (u : RandomPath) : Arm :=
  let c := counts π p u T
  π.recommend c (Draft0.PolicyPaths.meansAt p c) (u T)

theorem counts_initial (π : StatePolicy) (p : Path) (u : RandomPath) (i : Arm) :
    counts π p u 3 i = 1 := rfl

theorem counts_update (π : StatePolicy) (p : Path) (u : RandomPath) (s : ℕ) (hs : 3 ≤ s) :
    counts π p u (s + 1) = advance (counts π p u s) (selected π p u (s + 1)) := by
  have he : s + 1 - 3 = (s - 3) + 1 := by omega
  have ht : (s - 3) + 4 = s + 1 := by omega
  have hu : (s - 3) + 3 = s := by omega
  simp only [counts, he, afterInitial, selected, Nat.add_sub_cancel, ht, hu]

theorem counts_positive (π : StatePolicy) (p : Path) (u : RandomPath) (s : ℕ) (i : Arm) :
    0 < counts π p u s i := by
  unfold counts
  generalize s - 3 = r
  induction r with
  | zero => norm_num [afterInitial]
  | succ r ih =>
    simp only [afterInitial, advance]
    omega

theorem selected_optimal (π : StatePolicy) (T : ℕ) (hπ : PointwiseOptimal π T)
    (p : Path) (u : RandomPath) (t : ℕ) (ht : 4 ≤ t) (htT : t ≤ T) :
    action (T - t) (counts π p u (t - 1))
      (Draft0.PolicyPaths.meansAt p (counts π p u (t - 1))) (selected π p u t) =
    value (T - t + 1) (counts π p u (t - 1))
      (Draft0.PolicyPaths.meansAt p (counts π p u (t - 1))) := by
  exact hπ.1 t _ _ _ ht htT (counts_positive π p u (t - 1))

theorem recommendation_greedy (π : StatePolicy) (T : ℕ) (hπ : PointwiseOptimal π T)
    (p : Path) (u : RandomPath) (i : Arm) :
    Draft0.PolicyPaths.meansAt p (counts π p u T) i ≤
      Draft0.PolicyPaths.meansAt p (counts π p u T) (recommendation π T p u) := by
  rw [show Draft0.PolicyPaths.meansAt p (counts π p u T) (recommendation π T p u) =
      terminal (Draft0.PolicyPaths.meansAt p (counts π p u T)) from
    hπ.2 _ _ _ (counts_positive π p u T)]
  exact (recommend_greedy _ i).trans_eq (recommend_max _)

theorem afterInitial_joint_measurable (π : StatePolicy) (r : ℕ) :
    Measurable (fun x : Path × RandomPath => afterInitial π x.1 x.2 r) := by
  induction r with
  | zero => exact measurable_const
  | succ r ih =>
    have hp : Measurable (fun x : Path × RandomPath => x.1) := measurable_fst
    have hm := Draft0.PolicyPaths.meansAt_joint_measurable.comp (ih.prodMk hp)
    have hu : Measurable (fun x : Path × RandomPath => x.2 (r + 3)) :=
      (measurable_pi_apply (r + 3)).comp measurable_snd
    have ha := (π.draw_measurable (r + 4)).comp (ih.prodMk (hm.prodMk hu))
    simp only [afterInitial]
    exact (measurable_of_countable (fun x : Counts × Arm => advance x.1 x.2)).comp
      (ih.prodMk ha)

theorem counts_joint_measurable (π : StatePolicy) (s : ℕ) :
    Measurable (fun x : Path × RandomPath => counts π x.1 x.2 s) :=
  afterInitial_joint_measurable π (s - 3)

theorem recommendation_joint_measurable (π : StatePolicy) (T : ℕ) :
    Measurable (fun x : Path × RandomPath => recommendation π T x.1 x.2) := by
  have hc := counts_joint_measurable π T
  have hp : Measurable (fun x : Path × RandomPath => x.1) := measurable_fst
  have hm := Draft0.PolicyPaths.meansAt_joint_measurable.comp (hc.prodMk hp)
  have hu : Measurable (fun x : Path × RandomPath => x.2 T) :=
    (measurable_pi_apply T).comp measurable_snd
  exact π.recommend_measurable.comp (hc.prodMk (hm.prodMk hu))

theorem bellman_draw_measurable (T t : ℕ) :
    Measurable (fun x : Counts × Means × ℝ =>
      Draft0.BayesianModel.choose (T - t) x.1 x.2.1) := by
  apply measurable_from_prod_countable_right
  intro c
  exact (Draft0.PolicyPaths.choose_measurable (T - t) c).comp
    (measurable_fst : Measurable (fun x : Means × ℝ => x.1))

theorem bellman_recommend_measurable :
    Measurable (fun x : Counts × Means × ℝ => Draft0.BayesianModel.recommend x.2.1) := by
  have hm : Measurable (fun x : Counts × Means × ℝ => x.2.1) :=
    measurable_fst.comp measurable_snd
  exact Draft0.PolicyPaths.recommend_measurable.comp hm

def deterministicBellman (T : ℕ) : StatePolicy where
  draw := fun t c m _ => Draft0.BayesianModel.choose (T - t) c m
  recommend := fun _ m _ => Draft0.BayesianModel.recommend m
  draw_measurable := bellman_draw_measurable T
  recommend_measurable := bellman_recommend_measurable

theorem deterministicBellman_optimal (T : ℕ) : PointwiseOptimal (deterministicBellman T) T := by
  constructor
  · intro t c m u _ _ _
    exact choose_optimal (T - t) c m
  · intro c m u _
    exact recommend_max m

end
end Junpei.Policy
