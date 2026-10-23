import Junpei.FinitePosterior

/-! Information cannot increase optimal posterior expected regret.
The proof uses the actual finite-arm Gaussian Bellman recursion. -/

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.Information

open FiniteGaussian

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

omit [Fintype A] [Nonempty A] in
theorem advance_commute (c : Precisions A) (i j : A) :
    advance (advance c i) j = advance (advance c j) i := by
  funext a
  simp only [advance]
  exact add_right_comm _ _ _

omit [Fintype A] [Nonempty A] in
theorem shift_commute (m : Means A) (i j : A) (x y : ℝ) :
    shift (shift m i x) j y = shift (shift m j y) i x := by
  ext a
  simp only [shift]
  ring

omit [Fintype A] [Nonempty A] in
theorem innovation_other (c : Precisions A) (i j : A) (h : j ≠ i) :
    innovation (advance c i) j = innovation c j := by
  simp [innovation, advance, h]

omit [Nonempty A] in
theorem double_shift_integrable {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (m : Means A) (i j : A) (v w : ℝ≥0) :
    Integrable (fun p : ℝ × ℝ => f (shift (shift m i p.1) j p.2))
      ((gaussianReal 0 v).prod (gaussianReal 0 w)) := by
  have hb (p : ℝ × ℝ) : ‖f (shift (shift m i p.1) j p.2)‖ ≤
      |f m| + |p.1| + |p.2| := by
    have h := hf.dist_le_mul (shift (shift m i p.1) j p.2) m
    simp only [NNReal.coe_one, one_mul, dist_eq_norm] at h
    have hn : ‖shift (shift m i p.1) j p.2 - m‖ ≤ |p.1| + |p.2| := by
      have he : shift (shift m i p.1) j p.2 - m =
          (shift (shift m i p.1) j p.2 - shift m i p.1) + (shift m i p.1 - m) := by
        abel
      rw [he]
      exact (norm_add_le _ _).trans (by
        have h1 := shift_norm_bound (shift m i p.1) j p.2
        have h2 := shift_norm_bound m i p.1
        linarith)
    have ht := norm_add_le (f (shift (shift m i p.1) j p.2) - f m) (f m)
    rw [sub_add_cancel, Real.norm_eq_abs (f m)] at ht
    linarith
  have hcont : Continuous (fun p : ℝ × ℝ => shift (shift m i p.1) j p.2) := by
    apply continuous_pi
    intro a
    simp only [shift]
    split_ifs <;> fun_prop
  exact (((integrable_const |f m|).add
    ((Draft0.Gaussian.gaussian_id_integrable 0 v).norm.comp_fst (gaussianReal 0 w))).add
    ((Draft0.Gaussian.gaussian_id_integrable 0 w).norm.comp_snd (gaussianReal 0 v))).mono'
    (hf.continuous.comp hcont).measurable.aestronglyMeasurable
    (Filter.Eventually.of_forall hb)

theorem distinct_actions_commute (k : ℕ) (c : Precisions A) (m : Means A)
    (i j : A) (hij : i ≠ j) :
    (∫ x : ℝ, action k (advance c i) (shift m i x) j ∂gaussianReal 0 (innovation c i)) =
      ∫ y : ℝ, action k (advance c j) (shift m j y) i ∂gaussianReal 0 (innovation c j) := by
  simp only [action, innovation_other c i j hij.symm, innovation_other c j i hij]
  rw [integral_integral_swap
    (double_shift_integrable (value_lipschitz k (advance (advance c i) j)) m i j _ _)]
  simp_rw [advance_commute c i j, shift_commute m i j]

omit [Fintype A] [Nonempty A] in
theorem terminal_coordinate_mean (c : Precisions A) (m : Means A) (i j : A) :
    (∫ z : ℝ, shift m i z j ∂gaussianReal 0 (innovation c i)) = m j := by
  by_cases h : j = i
  · simp only [shift, if_pos h]
    rw [integral_add (integrable_const _) (Draft0.Gaussian.gaussian_id_integrable _ _)]
    simp [integral_id_gaussianReal]
  · simp [shift, h]

theorem terminal_action_ge (c : Precisions A) (m : Means A) (i : A) :
    value 0 c m ≤ action 0 c m i := by
  obtain ⟨j, hj⟩ := maxAll_attained m
  change maxAll m ≤ ∫ z : ℝ, maxAll (shift m i z) ∂gaussianReal 0 (innovation c i)
  rw [← hj, ← terminal_coordinate_mean c m i j]
  apply integral_mono _ (action_integrable 0 c m i)
    (fun z => coordinate_le_maxAll (shift m i z) j)
  by_cases h : j = i
  · simpa only [shift, if_pos h] using
      (integrable_const (m j)).add (Draft0.Gaussian.gaussian_id_integrable 0 (innovation c i))
  · simp [shift, h]

/-- Any first observation followed by k optimal observations is at least as
valuable as just the k optimal observations, for every finite arm set. -/
theorem value_le_action (k : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    value k c m ≤ action k c m i := by
  induction k generalizing c m with
  | zero => exact terminal_action_ge c m i
  | succ k ih =>
    rw [value_succ]
    apply maxAll_le
    intro j
    by_cases hij : i = j
    · subst j
      unfold action
      exact integral_mono (action_integrable k c m i)
        (action_integrable (k + 1) c m i) (fun z =>
          (ih (advance c i) (shift m i z)).trans
            (action_le_value k (advance c i) (shift m i z) i))
    · have hcomm := distinct_actions_commute k c m i j hij
      have hli : Integrable (fun x : ℝ => action k (advance c i) (shift m i x) j)
          (gaussianReal 0 (innovation c i)) :=
        shifted_integrable (integral_shift_lipschitz
          (value_lipschitz k (advance (advance c i) j)) j (innovation (advance c i) j))
          m i (innovation c i)
      have hlj : Integrable (fun y : ℝ => action k (advance c j) (shift m j y) i)
          (gaussianReal 0 (innovation c j)) :=
        shifted_integrable (integral_shift_lipschitz
          (value_lipschitz k (advance (advance c j) i)) i (innovation (advance c j) i))
          m j (innovation c j)
      calc
        action k c m j ≤ ∫ y : ℝ, action k (advance c j) (shift m j y) i
            ∂gaussianReal 0 (innovation c j) :=
          integral_mono (action_integrable k c m j) hlj
            (fun y => ih (advance c j) (shift m j y))
        _ = ∫ x : ℝ, action k (advance c i) (shift m i x) j
            ∂gaussianReal 0 (innovation c i) := hcomm.symm
        _ ≤ action (k + 1) c m i :=
          integral_mono hli (action_integrable (k + 1) c m i)
            (fun x => action_le_value k (advance c i) (shift m i x) j)

/-- Definition 4's nonnegative expected Bellman improvement, for arbitrary
positive real precision and every arm, follows by canceling the true
posterior expected terminal maximum. -/
theorem expected_bellman_improvement_nonnegative (k : ℕ) (c : Precisions A)
    (m : Means A) (i : A) (hc : 0 < c i) :
    0 ≤ FinitePosterior.loss k c m - FinitePosterior.actionLoss k c m i := by
  rw [FinitePosterior.actionLoss_eq _ _ _ _ hc]
  unfold FinitePosterior.loss
  have h := value_le_action k c m i
  linarith

end
end Junpei.Information
