import Draft0.BayesianModel

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Draft0.PosteriorRegret

open BayesianModel

noncomputable section

def smooth (f : Means → ℝ) (i : Arm) (v : ℝ≥0) (m : Means) : ℝ :=
  ∫ z : ℝ, f (shift m i z) ∂gaussianReal 0 v

theorem smooth_lipschitz {f : Means → ℝ} (hf : LipschitzWith 1 f)
    (i : Arm) (v : ℝ≥0) : LipschitzWith 1 (smooth f i v) :=
  integral_shift_lipschitz hf i v

theorem shift_same (m : Means) (i : Arm) (x y : ℝ) :
    shift (shift m i x) i y = shift m i (x + y) := by
  ext j
  by_cases h : j = i
  · simp [shift, h, add_assoc]
  · simp [shift, h]

theorem shift_comm (m : Means) (i j : Arm) (x y : ℝ) :
    shift (shift m i x) j y = shift (shift m j y) i x := by
  ext a
  simp only [shift]
  ring

theorem double_shift_integrable {f : Means → ℝ} (hf : LipschitzWith 1 f)
    (m : Means) (i j : Arm) (v w : ℝ≥0) :
    Integrable (fun p : ℝ × ℝ => f (shift (shift m i p.1) j p.2))
      ((gaussianReal 0 v).prod (gaussianReal 0 w)) := by
  have hb (p : ℝ × ℝ) : ‖f (shift (shift m i p.1) j p.2)‖ ≤
      ‖f m‖ + |p.1| + |p.2| := by
    have h := hf.dist_le_mul (shift (shift m i p.1) j p.2) m
    simp only [dist_eq_norm, NNReal.coe_one, one_mul] at h
    have hn := norm_sub_le_norm_sub_add_norm_sub
      (shift (shift m i p.1) j p.2) (shift m i p.1) m
    have h₁ := shift_norm_bound m i p.1
    have h₂ := shift_norm_bound (shift m i p.1) j p.2
    have ht := norm_add_le (f (shift (shift m i p.1) j p.2) - f m) (f m)
    simp only [sub_add_cancel] at ht
    linarith
  have hc : Continuous (fun p : ℝ × ℝ => shift (shift m i p.1) j p.2) := by
    apply continuous_pi
    intro a
    unfold shift
    split_ifs <;> fun_prop
  exact ((integrable_const ‖f m‖).add
    ((Gaussian.gaussian_id_integrable 0 v).norm.comp_fst (gaussianReal 0 w))).add
    ((Gaussian.gaussian_id_integrable 0 w).norm.comp_snd (gaussianReal 0 v)) |>.mono'
      (hf.continuous.comp hc).measurable.aestronglyMeasurable
      (Filter.Eventually.of_forall hb)

theorem smooth_comm {f : Means → ℝ} (hf : LipschitzWith 1 f)
    (i j : Arm) (v w : ℝ≥0) :
    smooth (smooth f j w) i v = smooth (smooth f i v) j w := by
  funext m
  unfold smooth
  rw [integral_integral_swap (double_shift_integrable hf m i j v w)]
  congr 1
  funext y
  congr 1
  funext x
  rw [shift_comm]

theorem smooth_add {f : Means → ℝ} (hf : LipschitzWith 1 f)
    (i : Arm) (v w : ℝ≥0) :
    smooth (smooth f i w) i v = smooth f i (v + w) := by
  funext m
  unfold smooth
  have hconv : (gaussianReal 0 v) ∗ (gaussianReal 0 w) = gaussianReal 0 (v + w) := by
    simpa using (gaussianReal_conv_gaussianReal (m₁ := (0 : ℝ)) (m₂ := (0 : ℝ)))
  have hi : Integrable (fun z : ℝ => f (shift m i z))
      ((gaussianReal 0 v) ∗ (gaussianReal 0 w)) := by
    rw [hconv]
    exact shifted_integrable hf m i (v + w)
  rw [← hconv, integral_conv hi]
  congr 1
  funext x
  congr 1
  funext y
  rw [shift_same]

def potential (c : Counts) : Means → ℝ :=
  smooth (smooth (smooth terminal 2 (c 2 : ℝ≥0)⁻¹)
    1 (c 1 : ℝ≥0)⁻¹) 0 (c 0 : ℝ≥0)⁻¹

theorem potential_lipschitz (c : Counts) : LipschitzWith 1 (potential c) :=
  smooth_lipschitz (smooth_lipschitz (smooth_lipschitz terminal_lipschitz _ _) _ _) _ _

theorem count_variance_split (n : ℕ) (hn : 0 < n) :
    (n : ℝ≥0)⁻¹ = Gaussian.updateVariance (n : ℝ≥0) + ((n + 1 : ℕ) : ℝ≥0)⁻¹ := by
  have h := Gaussian.posteriorVariance_after_draw (n : ℝ≥0) 0
    (by exact_mod_cast hn)
  simpa [Gaussian.posteriorVariance] using h

/-- The posterior expected best latent mean is a martingale under every arm
draw. This checks the additive constant connecting utility and regret. -/
theorem potential_tower (c : Counts) (i : Arm) (hc : 0 < c i) :
    smooth (potential (advance c i)) i (innovation c i) = potential c := by
  have hsplit := count_variance_split (c i) hc
  fin_cases i
  · change smooth (potential (advance c 0)) 0 (innovation c 0) = potential c
    norm_num [potential, advance, innovation] at hsplit ⊢
    rw [smooth_add (smooth_lipschitz (smooth_lipschitz terminal_lipschitz _ _) _ _)]
    rw [← hsplit]
  · change smooth (potential (advance c 1)) 1 (innovation c 1) = potential c
    norm_num [potential, advance, innovation] at hsplit ⊢
    rw [smooth_comm (smooth_lipschitz (smooth_lipschitz terminal_lipschitz _ _) _ _)
      1 0]
    congr 1
    rw [smooth_add (smooth_lipschitz terminal_lipschitz _ _), ← hsplit]
  · change smooth (potential (advance c 2)) 2 (innovation c 2) = potential c
    norm_num [potential, advance, innovation] at hsplit ⊢
    change (c 2 : ℝ≥0)⁻¹ = Gaussian.updateVariance (c 2 : ℝ≥0) +
      ((c 2 : ℝ≥0) + 1)⁻¹ at hsplit
    rw [smooth_comm (smooth_lipschitz (smooth_lipschitz terminal_lipschitz _ _) _ _)
      2 0]
    congr 1
    rw [smooth_comm (smooth_lipschitz terminal_lipschitz _ _) 2 1]
    congr 1
    rw [smooth_add terminal_lipschitz, ← hsplit]

def loss (k : ℕ) (c : Counts) (m : Means) : ℝ := potential c m - value k c m

def actionLoss (k : ℕ) (c : Counts) (m : Means) (i : Arm) : ℝ :=
  ∫ z : ℝ, loss k (advance c i) (shift m i z)
    ∂gaussianReal 0 (innovation c i)

theorem actionLoss_eq (k : ℕ) (c : Counts) (m : Means) (i : Arm)
    (hc : 0 < c i) : actionLoss k c m i = potential c m - action k c m i := by
  unfold actionLoss loss action
  rw [integral_sub (shifted_integrable (potential_lipschitz _) _ _ _)
    (action_integrable k c m i)]
  have ht := congrFun (potential_tower c i hc) m
  exact congrArg (fun x : ℝ => x - ∫ z : ℝ, value k (advance c i) (shift m i z)
    ∂gaussianReal 0 (innovation c i)) ht

theorem chosen_loss_minimal (k : ℕ) (c : Counts) (m : Means)
    (hc : ∀ i, 0 < c i) (i : Arm) :
    actionLoss k c m (choose k c m) ≤ actionLoss k c m i := by
  rw [actionLoss_eq _ _ _ _ (hc _), actionLoss_eq _ _ _ _ (hc _), choose_optimal]
  exact sub_le_sub_left (action_le_value k c m i) _

end

end Draft0.PosteriorRegret
