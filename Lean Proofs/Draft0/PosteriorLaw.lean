import Draft0.Optimality

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Draft0.PosteriorLaw

open BayesianModel PosteriorRegret

noncomputable section

abbrev Triple := ℝ × (ℝ × ℝ)

/-- Independent centered posterior errors, with the paper's inverse-count variances. -/
def centeredTripleLaw (c : Counts) : Measure Triple :=
  (gaussianReal 0 (c 0 : ℝ≥0)⁻¹).prod
    ((gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹))

def tripleShift (m : Means) (x : Triple) : Means :=
  shift (shift (shift m 0 x.1) 1 x.2.1) 2 x.2.2

theorem tripleShift_continuous (m : Means) : Continuous (tripleShift m) := by
  apply continuous_pi
  intro i
  fin_cases i <;> simp [tripleShift, shift] <;> fun_prop

theorem tripleShift_measurable (m : Means) : Measurable (tripleShift m) :=
  (tripleShift_continuous m).measurable

/-- The actual joint Gaussian posterior of the three latent arm means. -/
def posteriorLaw (c : Counts) (m : Means) : Measure Means :=
  Measure.map (tripleShift m) (centeredTripleLaw c)

instance centeredTripleLaw_probability (c : Counts) :
    IsProbabilityMeasure (centeredTripleLaw c) := by
  unfold centeredTripleLaw
  infer_instance

instance posteriorLaw_probability (c : Counts) (m : Means) :
    IsProbabilityMeasure (posteriorLaw c m) :=
  Measure.isProbabilityMeasure_map (tripleShift_measurable m).aemeasurable

theorem centered_first_law (c : Counts) :
    Measure.map Prod.fst (centeredTripleLaw c) = gaussianReal 0 (c 0 : ℝ≥0)⁻¹ := by
  simp [centeredTripleLaw]

theorem centered_second_law (c : Counts) :
    Measure.map (fun x : Triple => x.2.1) (centeredTripleLaw c) =
      gaussianReal 0 (c 1 : ℝ≥0)⁻¹ := by
  change Measure.map (Prod.fst ∘ Prod.snd) (centeredTripleLaw c) = _
  rw [← Measure.map_map measurable_fst measurable_snd]
  simp [centeredTripleLaw]

theorem centered_third_law (c : Counts) :
    Measure.map (fun x : Triple => x.2.2) (centeredTripleLaw c) =
      gaussianReal 0 (c 2 : ℝ≥0)⁻¹ := by
  change Measure.map (Prod.snd ∘ Prod.snd) (centeredTripleLaw c) = _
  rw [← Measure.map_map measurable_snd measurable_snd]
  simp [centeredTripleLaw]

theorem posterior_coordinate_law (c : Counts) (m : Means) (i : Arm) :
    Measure.map (fun u : Means => u i) (posteriorLaw c m) =
      gaussianReal (m i) (c i : ℝ≥0)⁻¹ := by
  unfold posteriorLaw
  rw [Measure.map_map (measurable_pi_apply i) (tripleShift_measurable m)]
  fin_cases i
  · have h := gaussianReal_map_const_add (μ := 0) (v := (c 0 : ℝ≥0)⁻¹) (m 0)
    rw [← centered_first_law c,
      Measure.map_map (by fun_prop) measurable_fst] at h
    simpa [Function.comp_def, tripleShift, shift] using h
  · have h := gaussianReal_map_const_add (μ := 0) (v := (c 1 : ℝ≥0)⁻¹) (m 1)
    rw [← centered_second_law c,
      Measure.map_map (by fun_prop) (by fun_prop)] at h
    simpa [Function.comp_def, tripleShift, shift] using h
  · have h := gaussianReal_map_const_add (μ := 0) (v := (c 2 : ℝ≥0)⁻¹) (m 2)
    rw [← centered_third_law c,
      Measure.map_map (by fun_prop) (by fun_prop)] at h
    simpa [Function.comp_def, tripleShift, shift] using h

theorem posterior_joint_law (c : Counts) (m : Means) :
    Measure.map (fun u : Means => (u 0, (u 1, u 2))) (posteriorLaw c m) =
      (gaussianReal (m 0) (c 0 : ℝ≥0)⁻¹).prod
        ((gaussianReal (m 1) (c 1 : ℝ≥0)⁻¹).prod
          (gaussianReal (m 2) (c 2 : ℝ≥0)⁻¹)) := by
  unfold posteriorLaw
  rw [Measure.map_map (by fun_prop) (tripleShift_measurable m)]
  have he : (fun u : Means => (u 0, (u 1, u 2))) ∘ tripleShift m =
      Prod.map (fun x : ℝ => m 0 + x)
        (Prod.map (fun x : ℝ => m 1 + x) (fun x : ℝ => m 2 + x)) := by
    funext x
    simp [tripleShift, shift, Prod.map]
  rw [he]
  unfold centeredTripleLaw
  rw [← Measure.map_prod_map _ _ (by fun_prop) (by fun_prop),
    ← Measure.map_prod_map _ _ (by fun_prop) (by fun_prop)]
  simp only [gaussianReal_map_const_add, zero_add]

theorem triple_shift_integrable {f : Means → ℝ} (hf : LipschitzWith 1 f)
    (m : Means) (c : Counts) :
    Integrable (fun x : Triple => f (tripleShift m x)) (centeredTripleLaw c) := by
  have hb (x : Triple) : ‖f (tripleShift m x)‖ ≤
      ‖f m‖ + |x.1| + |x.2.1| + |x.2.2| := by
    have h := hf.dist_le_mul (tripleShift m x) m
    simp only [dist_eq_norm, NNReal.coe_one, one_mul] at h
    have h₀ := shift_norm_bound m 0 x.1
    have h₁ := shift_norm_bound (shift m 0 x.1) 1 x.2.1
    have h₂ := shift_norm_bound (shift (shift m 0 x.1) 1 x.2.1) 2 x.2.2
    have ht₁ := norm_sub_le_norm_sub_add_norm_sub
      (shift (shift m 0 x.1) 1 x.2.1) (shift m 0 x.1) m
    have ht₂ := norm_sub_le_norm_sub_add_norm_sub
      (tripleShift m x) (shift (shift m 0 x.1) 1 x.2.1) m
    have ht := norm_add_le (f (tripleShift m x) - f m) (f m)
    simp only [sub_add_cancel] at ht
    unfold tripleShift at h ht ht₂ ⊢
    linarith
  have hi₀ := (Gaussian.gaussian_id_integrable 0 (c 0 : ℝ≥0)⁻¹).norm.comp_fst
    ((gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹))
  have hi₁ := ((Gaussian.gaussian_id_integrable 0 (c 1 : ℝ≥0)⁻¹).norm.comp_fst
    (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)).comp_snd (gaussianReal 0 (c 0 : ℝ≥0)⁻¹)
  have hi₂ := ((Gaussian.gaussian_id_integrable 0 (c 2 : ℝ≥0)⁻¹).norm.comp_snd
    (gaussianReal 0 (c 1 : ℝ≥0)⁻¹)).comp_snd (gaussianReal 0 (c 0 : ℝ≥0)⁻¹)
  exact (((integrable_const ‖f m‖).add hi₀).add hi₁).add hi₂ |>.mono'
    (hf.continuous.comp (tripleShift_continuous m)).measurable.aestronglyMeasurable
    (Filter.Eventually.of_forall hb)

theorem posterior_terminal_integrable (c : Counts) (m : Means) :
    Integrable terminal (posteriorLaw c m) := by
  apply (integrable_map_measure terminal_lipschitz.continuous.measurable.aestronglyMeasurable
    (tripleShift_measurable m).aemeasurable).mpr
  exact triple_shift_integrable terminal_lipschitz m c

theorem potential_eq_posterior_integral (c : Counts) (m : Means) :
    potential c m = ∫ u : Means, terminal u ∂posteriorLaw c m := by
  rw [posteriorLaw, integral_map (tripleShift_measurable m).aemeasurable
    terminal_lipschitz.continuous.measurable.aestronglyMeasurable]
  unfold centeredTripleLaw
  rw [integral_prod _ (triple_shift_integrable terminal_lipschitz m c)]
  unfold potential smooth
  congr 1
  funext x
  unfold tripleShift
  rw [integral_prod _ (double_shift_integrable terminal_lipschitz
    (shift m 0 x) 1 2 _ _)]

theorem posterior_coordinate_integrable (c : Counts) (m : Means) (i : Arm) :
    Integrable (fun u : Means => u i) (posteriorLaw c m) := by
  have hi : Integrable (fun x : ℝ => x)
      (Measure.map (fun u : Means => u i) (posteriorLaw c m)) := by
    rw [posterior_coordinate_law]
    exact Gaussian.gaussian_id_integrable _ _
  exact hi.comp_measurable (measurable_pi_apply i)

theorem posterior_coordinate_mean (c : Counts) (m : Means) (i : Arm) :
    ∫ u : Means, u i ∂posteriorLaw c m = m i := by
  have h := integral_id_gaussianReal (μ := m i) (v := (c i : ℝ≥0)⁻¹)
  rw [← posterior_coordinate_law c m i,
    integral_map (measurable_pi_apply i).aemeasurable (by fun_prop)] at h
  exact h

/-- Actual posterior expected simple regret of recommending a fixed arm. -/
def terminalRegret (c : Counts) (m : Means) (i : Arm) : ℝ :=
  ∫ u : Means, terminal u - u i ∂posteriorLaw c m

theorem terminalRegret_integrable (c : Counts) (m : Means) (i : Arm) :
    Integrable (fun u : Means => terminal u - u i) (posteriorLaw c m) :=
  (posterior_terminal_integrable c m).sub (posterior_coordinate_integrable c m i)

theorem terminalRegret_nonnegative (c : Counts) (m : Means) (i : Arm) :
    0 ≤ terminalRegret c m i := by
  apply integral_nonneg
  intro u
  exact sub_nonneg.mpr ((recommend_greedy u i).trans_eq (recommend_max u))

theorem terminalRegret_eq (c : Counts) (m : Means) (i : Arm) :
    terminalRegret c m i = potential c m - m i := by
  unfold terminalRegret
  rw [integral_sub (posterior_terminal_integrable c m)
    (posterior_coordinate_integrable c m i),
    ← potential_eq_posterior_integral, posterior_coordinate_mean]

theorem terminalRegret_recommend_eq_loss (c : Counts) (m : Means) :
    terminalRegret c m (recommend m) = loss 0 c m := by
  rw [terminalRegret_eq, recommend_max m]
  rfl

theorem terminalRegret_recommend_minimal (c : Counts) (m : Means) (i : Arm) :
    terminalRegret c m (recommend m) ≤ terminalRegret c m i := by
  rw [terminalRegret_eq, terminalRegret_eq]
  exact sub_le_sub_left (recommend_greedy m i) _

theorem terminalRegret_stop_eq_expectedRegret (c : Counts) (m : Means) (i : Arm) :
    terminalRegret c m i = Optimality.expectedRegret (.stop i) c m := by
  exact terminalRegret_eq c m i

end

end Draft0.PosteriorLaw
