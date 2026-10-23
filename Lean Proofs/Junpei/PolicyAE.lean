import Junpei.Policy

open MeasureTheory ProbabilityTheory

namespace Junpei.Policy

set_option maxHeartbeats 800000

open Draft0.BayesianModel

noncomputable section

/-- Randomization may choose suboptimal actions on a Gaussian-null subset
of the external seed, separately at each proper posterior state. -/
def AEOptimal (π : StatePolicy) (T : ℕ) : Prop :=
  (∀ (t : ℕ) (c : Counts) (m : Means), 4 ≤ t → t ≤ T → (∀ i, 0 < c i) →
    ∀ᵐ u : ℝ ∂gaussianReal 0 1,
      action (T - t) c m (π.draw t c m u) = value (T - t + 1) c m) ∧
  (∀ (c : Counts) (m : Means), (∀ i, 0 < c i) →
    ∀ᵐ u : ℝ ∂gaussianReal 0 1, m (π.recommend c m u) = terminal m)

theorem pointwiseOptimal_aeOptimal (π : StatePolicy) (T : ℕ) (hπ : PointwiseOptimal π T) :
    AEOptimal π T := by
  constructor
  · intro t c m ht htT hc
    exact ae_of_all _ (fun u => hπ.1 t c m u ht htT hc)
  · intro c m hc
    exact ae_of_all _ (fun u => hπ.2 c m u hc)

theorem action_joint_measurable (k : ℕ) :
    Measurable (fun x : Counts × Means × Arm => action k x.1 x.2.1 x.2.2) := by
  apply measurable_from_prod_countable_right
  intro c
  apply measurable_from_prod_countable_left
  intro i
  exact (action_lipschitz k c i).continuous.measurable

theorem value_joint_measurable (k : ℕ) :
    Measurable (fun x : Counts × Means => value k x.1 x.2) := by
  apply measurable_from_prod_countable_right
  intro c
  exact (value_lipschitz k c).continuous.measurable

theorem evaluation_joint_measurable :
    Measurable (fun x : Means × Arm => x.1 x.2) := by
  apply measurable_from_prod_countable_left
  intro i
  exact measurable_pi_apply i

theorem draw_good_measurable (π : StatePolicy) (T t : ℕ) :
    MeasurableSet {x : Counts × Means × ℝ |
      action (T - t) x.1 x.2.1 (π.draw t x.1 x.2.1 x.2.2) =
        value (T - t + 1) x.1 x.2.1} := by
  have hc : Measurable (fun x : Counts × Means × ℝ => x.1) := measurable_fst
  have hm : Measurable (fun x : Counts × Means × ℝ => x.2.1) :=
    measurable_fst.comp measurable_snd
  have hd := π.draw_measurable t
  have ha := (action_joint_measurable (T - t)).comp (hc.prodMk (hm.prodMk hd))
  have hv := (value_joint_measurable (T - t + 1)).comp (hc.prodMk hm)
  exact measurableSet_eq_fun ha hv

theorem recommend_good_measurable (π : StatePolicy) :
    MeasurableSet {x : Counts × Means × ℝ |
      x.2.1 (π.recommend x.1 x.2.1 x.2.2) = terminal x.2.1} := by
  have hm : Measurable (fun x : Counts × Means × ℝ => x.2.1) :=
    measurable_fst.comp measurable_snd
  have hv := evaluation_joint_measurable.comp (hm.prodMk π.recommend_measurable)
  exact measurableSet_eq_fun hv (terminal_lipschitz.continuous.measurable.comp hm)

def repairedDraw (π : StatePolicy) (T t : ℕ) (c : Counts) (m : Means) (u : ℝ) : Arm :=
  if action (T - t) c m (π.draw t c m u) = value (T - t + 1) c m then
    π.draw t c m u else choose (T - t) c m

def repairedRecommend (π : StatePolicy) (c : Counts) (m : Means) (u : ℝ) : Arm :=
  if m (π.recommend c m u) = terminal m then π.recommend c m u else recommend m

theorem repairedDraw_measurable (π : StatePolicy) (T t : ℕ) :
    Measurable (fun x : Counts × Means × ℝ => repairedDraw π T t x.1 x.2.1 x.2.2) := by
  exact Measurable.ite (draw_good_measurable π T t) (π.draw_measurable t)
    (bellman_draw_measurable T t)

theorem repairedRecommend_measurable (π : StatePolicy) :
    Measurable (fun x : Counts × Means × ℝ => repairedRecommend π x.1 x.2.1 x.2.2) := by
  exact Measurable.ite (recommend_good_measurable π) π.recommend_measurable
    bellman_recommend_measurable

/-- A measurable canonical version that repairs only seed values giving a
nonoptimal action or recommendation at the current state. -/
def repair (π : StatePolicy) (T : ℕ) : StatePolicy where
  draw := repairedDraw π T
  recommend := repairedRecommend π
  draw_measurable := repairedDraw_measurable π T
  recommend_measurable := repairedRecommend_measurable π

theorem repair_pointwiseOptimal (π : StatePolicy) (T : ℕ) : PointwiseOptimal (repair π T) T := by
  constructor
  · intro t c m u _ _ _
    change action (T - t) c m (repairedDraw π T t c m u) = value (T - t + 1) c m
    unfold repairedDraw
    split_ifs with h
    · exact h
    · exact choose_optimal (T - t) c m
  · intro c m u _
    change m (repairedRecommend π c m u) = terminal m
    unfold repairedRecommend
    split_ifs with h
    · exact h
    · exact recommend_max m

theorem repair_draw_ae (π : StatePolicy) (T : ℕ) (hπ : AEOptimal π T)
    (t : ℕ) (c : Counts) (m : Means) (ht : 4 ≤ t) (htT : t ≤ T) (hc : ∀ i, 0 < c i) :
    π.draw t c m =ᵐ[gaussianReal 0 1] (repair π T).draw t c m := by
  filter_upwards [hπ.1 t c m ht htT hc] with u hu
  simp only [repair, repairedDraw, if_pos hu]

theorem repair_recommend_ae (π : StatePolicy) (T : ℕ) (hπ : AEOptimal π T)
    (c : Counts) (m : Means) (hc : ∀ i, 0 < c i) :
    π.recommend c m =ᵐ[gaussianReal 0 1] (repair π T).recommend c m := by
  filter_upwards [hπ.2 c m hc] with u hu
  simp only [repair, repairedRecommend, if_pos hu]

end

end Junpei.Policy
