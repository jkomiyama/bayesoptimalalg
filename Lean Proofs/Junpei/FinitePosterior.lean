import Junpei.FiniteGaussian
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.Independence.Basic

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FinitePosterior

open Junpei.FiniteGaussian

variable {A : Type*} [Fintype A] [DecidableEq A]

noncomputable section

def centeredLaw (c : Precisions A) : Measure (Means A) :=
  Measure.pi (fun a => gaussianReal 0 (c a)⁻¹)

instance centeredLaw_probability (c : Precisions A) :
    IsProbabilityMeasure (centeredLaw c) := by unfold centeredLaw; infer_instance

def posteriorLaw (c : Precisions A) (m : Means A) : Measure (Means A) :=
  Measure.map (fun x => m + x) (centeredLaw c)

instance posteriorLaw_probability (c : Precisions A) (m : Means A) :
    IsProbabilityMeasure (posteriorLaw c m) :=
  Measure.isProbabilityMeasure_map (by fun_prop)

omit [DecidableEq A] in
theorem centered_coordinate_law (c : Precisions A) (i : A) :
    (centeredLaw c).map (fun x => x i) = gaussianReal 0 (c i)⁻¹ :=
  (measurePreserving_eval (fun a => gaussianReal 0 (c a)⁻¹) i).map_eq

omit [DecidableEq A] in
theorem centered_coordinate_integrable (c : Precisions A) (i : A) :
    Integrable (fun x : Means A => x i) (centeredLaw c) := by
  have h := Draft0.Gaussian.gaussian_id_integrable 0 (c i)⁻¹
  rw [← centered_coordinate_law c i] at h
  exact h.comp_measurable (measurable_pi_apply i)

omit [DecidableEq A] in
theorem norm_le_sum_abs (x : Means A) : ‖x‖ ≤ ∑ a, |x a| := by
  have hs : 0 ≤ ∑ a, |x a| := Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  apply (pi_norm_le_iff_of_nonneg hs).mpr
  intro i
  rw [Real.norm_eq_abs]
  exact Finset.single_le_sum (fun a (_ : a ∈ Finset.univ) => abs_nonneg (x a))
    (Finset.mem_univ i)

omit [DecidableEq A] in
theorem centered_sum_integrable (c : Precisions A) :
    Integrable (fun x : Means A => ∑ a, |x a|) (centeredLaw c) := by
  apply integrable_finset_sum
  intro i _
  simpa only [Real.norm_eq_abs] using (centered_coordinate_integrable c i).norm

omit [DecidableEq A] in
theorem translated_integrable {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) :
    Integrable (fun x => f (m + x)) (centeredLaw c) := by
  have hb (x : Means A) : ‖f (m + x)‖ ≤ |f m| + ∑ a, |x a| := by
    have hl := hf.dist_le_mul (m + x) m
    simp only [NNReal.coe_one, one_mul, dist_eq_norm, add_sub_cancel_left] at hl
    have ht := norm_add_le (f (m + x) - f m) (f m)
    rw [sub_add_cancel, Real.norm_eq_abs (f m)] at ht
    linarith [norm_le_sum_abs x]
  exact ((integrable_const |f m|).add (centered_sum_integrable c)).mono'
    (hf.continuous.comp (by fun_prop)).measurable.aestronglyMeasurable
    (Filter.Eventually.of_forall hb)

omit [DecidableEq A] in
theorem posterior_integrable {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) : Integrable f (posteriorLaw c m) := by
  apply (integrable_map_measure hf.continuous.measurable.aestronglyMeasurable
    (by fun_prop : AEMeasurable (fun x : Means A => m + x) (centeredLaw c))).mpr
  exact translated_integrable hf c m

omit [DecidableEq A] in
theorem posterior_integral {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) :
    (∫ u, f u ∂posteriorLaw c m) = ∫ x, f (m + x) ∂centeredLaw c := by
  exact integral_map (by fun_prop) hf.continuous.measurable.aestronglyMeasurable

omit [DecidableEq A] in
theorem posterior_coordinate_law (c : Precisions A) (m : Means A) (i : A) :
    (posteriorLaw c m).map (fun x => x i) = gaussianReal (m i) (c i)⁻¹ := by
  unfold posteriorLaw
  rw [Measure.map_map (measurable_pi_apply i) (by fun_prop)]
  have h := gaussianReal_map_const_add (μ := (0 : ℝ)) (v := (c i)⁻¹) (m i)
  rw [← centered_coordinate_law c i] at h
  rw [Measure.map_map (g := fun x : ℝ => m i + x) (f := fun x : Means A => x i)
    (by fun_prop) (measurable_pi_apply i)] at h
  simpa only [Function.comp_def, Pi.add_apply, zero_add] using h

omit [DecidableEq A] in
theorem posterior_coordinate_integrable (c : Precisions A) (m : Means A) (i : A) :
    Integrable (fun x : Means A => x i) (posteriorLaw c m) :=
  posterior_integrable (LipschitzWith.eval i) c m

omit [DecidableEq A] in
theorem posterior_coordinate_mean (c : Precisions A) (m : Means A) (i : A) :
    (∫ u, u i ∂posteriorLaw c m) = m i := by
  have h := integral_id_gaussianReal (μ := m i) (v := (c i)⁻¹)
  rw [← posterior_coordinate_law c m i,
    integral_map (measurable_pi_apply i).aemeasurable (by fun_prop)] at h
  exact h

def otherLaw (c : Precisions A) (i : A) : Measure ({a : A // a ≠ i} → ℝ) :=
  Measure.pi (fun a => gaussianReal 0 (c a)⁻¹)

instance otherLaw_probability (c : Precisions A) (i : A) :
    IsProbabilityMeasure (otherLaw c i) := by unfold otherLaw; infer_instance

def join (i : A) (x : ℝ × ({a : A // a ≠ i} → ℝ)) : Means A :=
  fun a => if h : a = i then x.1 else x.2 ⟨a, h⟩

omit [Fintype A] in
theorem join_continuous (i : A) : Continuous (join i) := by
  apply continuous_pi
  intro a
  by_cases h : a = i <;> simp only [join, h, dite_true, dite_false] <;> fun_prop

theorem join_law (c : Precisions A) (i : A) :
    Measure.map (join i) ((gaussianReal 0 (c i)⁻¹).prod (otherLaw c i)) = centeredLaw c := by
  symm
  apply Measure.pi_eq
  intro s hs
  rw [Measure.map_apply (join_continuous i).measurable (MeasurableSet.univ_pi hs)]
  have he : join i ⁻¹' Set.pi Set.univ s =
      s i ×ˢ Set.pi Set.univ (fun a : {a : A // a ≠ i} => s a) := by
    ext x
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_true_left,
      Set.mem_prod, join]
    constructor
    · intro h
      exact ⟨by simpa using h i, fun a => by simpa [a.property] using h a⟩
    · rintro ⟨hi, ho⟩ a
      split_ifs with ha
      · subst a; exact hi
      · exact ho ⟨a, ha⟩
  rw [he, Measure.prod_prod, otherLaw, Measure.pi_pi]
  letI : Unique {a : A // a = i} :=
    ⟨⟨i, rfl⟩, fun a => Subtype.ext a.property⟩
  have hprod := Fintype.prod_subtype_mul_prod_subtype (fun a : A => a = i)
    (fun a => (gaussianReal 0 (c a)⁻¹) (s a))
  simpa only [Fintype.prod_unique] using hprod

theorem join_integrable {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) (i : A) :
    Integrable (fun x => f (m + join i x))
      ((gaussianReal 0 (c i)⁻¹).prod (otherLaw c i)) := by
  have h := translated_integrable hf c m
  rw [← join_law c i] at h
  exact h.comp_measurable (join_continuous i).measurable

def expectation (f : Means A → ℝ) (c : Precisions A) (m : Means A) : ℝ :=
  ∫ x, f (m + x) ∂centeredLaw c

omit [DecidableEq A] in
theorem expectation_lipschitz {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) : LipschitzWith 1 (expectation f c) := by
  apply LipschitzWith.of_dist_le_mul
  intro m n
  simp only [NNReal.coe_one, one_mul, dist_eq_norm]
  unfold expectation
  rw [← integral_sub (translated_integrable hf c m) (translated_integrable hf c n)]
  have hb : ∀ᵐ x ∂centeredLaw c, ‖f (m + x) - f (n + x)‖ ≤ ‖m - n‖ := by
    apply Filter.Eventually.of_forall
    intro x
    have h := hf.dist_le_mul (m + x) (n + x)
    simpa only [dist_eq_norm, NNReal.coe_one, one_mul, add_sub_add_right_eq_sub] using h
  simpa using norm_integral_le_of_norm_le_const hb

theorem expectation_split {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) (i : A) :
    expectation f c m = ∫ u, ∫ x : ℝ, f (m + join i (x, u))
      ∂gaussianReal 0 (c i)⁻¹ ∂otherLaw c i := by
  unfold expectation
  rw [← join_law c i]
  rw [integral_map (f := fun x : Means A => f (m + x))
    (join_continuous i).measurable.aemeasurable
    (hf.continuous.comp (by fun_prop)).measurable.aestronglyMeasurable,
    integral_prod _ (join_integrable hf c m i),
    integral_integral_swap (join_integrable hf c m i)]

def smooth (f : Means A → ℝ) (i : A) (v : ℝ≥0) (m : Means A) : ℝ :=
  ∫ z : ℝ, f (shift m i z) ∂gaussianReal 0 v

theorem smooth_lipschitz {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (i : A) (v : ℝ≥0) : LipschitzWith 1 (smooth f i v) :=
  integral_shift_lipschitz hf i v

omit [Fintype A] in
theorem shift_same (m : Means A) (i : A) (x y : ℝ) :
    shift (shift m i x) i y = shift m i (x + y) := by
  ext a
  by_cases h : a = i <;> simp [shift, h, add_assoc]

omit [Fintype A] in
theorem shift_zero (m : Means A) (i : A) : shift m i 0 = m := by
  ext a
  simp [shift]

theorem smooth_add {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (i : A) (v w : ℝ≥0) :
    smooth (smooth f i w) i v = smooth f i (v + w) := by
  funext m
  unfold smooth
  have hconv : gaussianReal 0 v ∗ gaussianReal 0 w = gaussianReal 0 (v + w) := by
    simpa using gaussianReal_conv_gaussianReal (m₁ := (0 : ℝ)) (m₂ := (0 : ℝ))
  have hi : Integrable (fun z : ℝ => f (shift m i z))
      (gaussianReal 0 v ∗ gaussianReal 0 w) := by
    rw [hconv]
    exact shifted_integrable hf m i (v + w)
  rw [← hconv, integral_conv hi]
  congr 1
  funext x
  congr 1
  funext y
  rw [shift_same]

omit [Fintype A] in
theorem shift_join (m : Means A) (i : A) (u : {a : A // a ≠ i} → ℝ) (z x : ℝ) :
    shift (shift m i z + join i (0, u)) i x = shift m i z + join i (x, u) := by
  ext a
  by_cases h : a = i <;> simp [shift, join, h, Pi.add_apply]

omit [Fintype A] in
theorem shift_add_join (m : Means A) (i : A) (u : {a : A // a ≠ i} → ℝ) (z : ℝ) :
    shift m i z + join i (0, u) = shift (m + join i (0, u)) i z := by
  ext a
  simp only [shift, Pi.add_apply]
  ring

theorem expectation_split_smooth {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) (i : A) :
    expectation f c m = ∫ u, smooth f i (c i)⁻¹ (m + join i (0, u)) ∂otherLaw c i := by
  rw [expectation_split hf c m i]
  congr 1
  funext u
  unfold smooth
  congr 1
  funext x
  have h := shift_join m i u 0 x
  simpa only [shift_zero] using congrArg f h.symm

theorem join_zero_norm_bound (i : A) (u : {a : A // a ≠ i} → ℝ) :
    ‖join i (0, u)‖ ≤ ∑ a, |u a| := by
  apply (pi_norm_le_iff_of_nonneg (Finset.sum_nonneg (fun _ _ => abs_nonneg _))).mpr
  intro a
  by_cases h : a = i
  · simp only [join, h, dite_true, norm_zero]
    exact Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  · simp only [join, h, dite_false, Real.norm_eq_abs]
    exact Finset.single_le_sum (fun b (_ : b ∈ Finset.univ) => abs_nonneg (u b))
      (Finset.mem_univ (⟨a, h⟩ : {a : A // a ≠ i}))

theorem shift_join_integrable {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) (i : A) (v : ℝ≥0) :
    Integrable (fun p : ℝ × ({a : A // a ≠ i} → ℝ) =>
      f (shift m i p.1 + join i (0, p.2)))
      ((gaussianReal 0 v).prod (otherLaw c i)) := by
  have hiu : Integrable (fun u : {a : A // a ≠ i} → ℝ => ∑ a, |u a|)
      (otherLaw c i) := centered_sum_integrable (fun a : {a : A // a ≠ i} => c a)
  have hb (p : ℝ × ({a : A // a ≠ i} → ℝ)) :
      ‖f (shift m i p.1 + join i (0, p.2))‖ ≤ |f m| + |p.1| + ∑ a, |p.2 a| := by
    have hl := hf.dist_le_mul (shift m i p.1 + join i (0, p.2)) m
    simp only [NNReal.coe_one, one_mul, dist_eq_norm] at hl
    have hn : ‖shift m i p.1 + join i (0, p.2) - m‖ ≤ |p.1| + ∑ a, |p.2 a| := by
      have he : shift m i p.1 + join i (0, p.2) - m =
          (shift m i p.1 - m) + join i (0, p.2) := by abel
      rw [he]
      exact (norm_add_le _ _).trans (add_le_add (shift_norm_bound m i p.1)
        (join_zero_norm_bound i p.2))
    have ht := norm_add_le (f (shift m i p.1 + join i (0, p.2)) - f m) (f m)
    rw [sub_add_cancel, Real.norm_eq_abs (f m)] at ht
    linarith
  have hcont : Continuous (fun p : ℝ × ({a : A // a ≠ i} → ℝ) =>
      shift m i p.1 + join i (0, p.2)) :=
    ((shift_continuous m i).comp continuous_fst).add
      ((join_continuous i).comp (continuous_const.prodMk continuous_snd))
  exact (((integrable_const |f m|).add
    ((Draft0.Gaussian.gaussian_id_integrable 0 v).norm.comp_fst (otherLaw c i))).add
    (hiu.comp_snd (gaussianReal 0 v))).mono'
      (hf.continuous.comp hcont).measurable.aestronglyMeasurable
      (Filter.Eventually.of_forall hb)

theorem otherLaw_advance (c : Precisions A) (i : A) :
    otherLaw (advance c i) i = otherLaw c i := by
  unfold otherLaw
  congr 1
  funext a
  simp [advance, a.property]

theorem variance_split (c : ℝ≥0) (hc : 0 < c) :
    c⁻¹ = Draft0.Gaussian.updateVariance c + (c + 1)⁻¹ := by
  have h := Draft0.Gaussian.posteriorVariance_after_draw c 0 hc
  simpa [Draft0.Gaussian.posteriorVariance] using h

/-- Exact posterior tower for every arm and every integrable Lipschitz payoff,
under the actual independent finite product Gaussian posterior. -/
theorem expectation_tower {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (c : Precisions A) (m : Means A) (i : A) (hc : 0 < c i) :
    (∫ z : ℝ, expectation f (advance c i) (shift m i z)
      ∂gaussianReal 0 (innovation c i)) = expectation f c m := by
  have he (z : ℝ) := expectation_split_smooth hf (advance c i) (shift m i z) i
  simp_rw [he]
  rw [otherLaw_advance]
  have hci : advance c i i = c i + 1 := by simp [advance]
  rw [hci]
  rw [integral_integral_swap
    (shift_join_integrable (smooth_lipschitz hf i (c i + 1)⁻¹) c m i (innovation c i))]
  rw [expectation_split_smooth hf c m i]
  congr 1
  funext u
  simp_rw [shift_add_join]
  change smooth (smooth f i (c i + 1)⁻¹) i (innovation c i) _ =
    smooth f i (c i)⁻¹ _
  rw [smooth_add hf]
  unfold innovation
  rw [← variance_split (c i) hc]

section Nonempty
variable [Nonempty A]

def potential (c : Precisions A) (m : Means A) : ℝ := expectation maxAll c m

omit [DecidableEq A] in
theorem potential_lipschitz (c : Precisions A) : LipschitzWith 1 (potential c) :=
  expectation_lipschitz maxAll_lipschitz c

omit [DecidableEq A] in
theorem potential_eq_posterior (c : Precisions A) (m : Means A) :
    potential c m = ∫ u, maxAll u ∂posteriorLaw c m :=
  (posterior_integral maxAll_lipschitz c m).symm

theorem potential_tower (c : Precisions A) (m : Means A) (i : A) (hc : 0 < c i) :
    (∫ z : ℝ, potential (advance c i) (shift m i z)
      ∂gaussianReal 0 (innovation c i)) = potential c m :=
  expectation_tower maxAll_lipschitz c m i hc

def terminalRegret (c : Precisions A) (m : Means A) (i : A) : ℝ :=
  ∫ u, maxAll u - u i ∂posteriorLaw c m

omit [DecidableEq A] in
theorem terminalRegret_eq (c : Precisions A) (m : Means A) (i : A) :
    terminalRegret c m i = potential c m - m i := by
  unfold terminalRegret
  rw [integral_sub (posterior_integrable maxAll_lipschitz c m)
    (posterior_coordinate_integrable c m i), ← potential_eq_posterior,
    posterior_coordinate_mean]

def loss (k : ℕ) (c : Precisions A) (m : Means A) : ℝ := potential c m - value k c m

def actionLoss (k : ℕ) (c : Precisions A) (m : Means A) (i : A) : ℝ :=
  ∫ z : ℝ, loss k (advance c i) (shift m i z) ∂gaussianReal 0 (innovation c i)

theorem actionLoss_eq (k : ℕ) (c : Precisions A) (m : Means A) (i : A) (hc : 0 < c i) :
    actionLoss k c m i = potential c m - action k c m i := by
  unfold actionLoss loss action
  rw [integral_sub (shifted_integrable (potential_lipschitz _) m i _)
    (action_integrable k c m i), potential_tower c m i hc]

def expectedRegret : {k : ℕ} → Strategy (A := A) k → Precisions A → Means A → ℝ
  | 0, .stop i, c, m => terminalRegret c m i
  | _ + 1, .sample i next, c, m =>
      ∫ z : ℝ, expectedRegret (next z) (advance c i) (shift m i z)
        ∂gaussianReal 0 (innovation c i)

omit [Fintype A] [Nonempty A] in
theorem advance_positive (c : Precisions A) (i : A) (hc : ∀ a, 0 < c a) :
    ∀ a, 0 < advance c i a := by
  intro a
  unfold advance
  exact lt_of_lt_of_le (hc a) (le_add_of_nonneg_right (by positivity))

/-- Recursive expected regret has the actual latent Gaussian simple-regret
integral at its leaves; the additive potential is proved, rather than assumed. -/
theorem expectedRegret_eq {k : ℕ} (s : Strategy (A := A) k)
    (c : Precisions A) (m : Means A) (hs : Admissible s c m) (hc : ∀ a, 0 < c a) :
    expectedRegret s c m = potential c m - utility s c m := by
  induction k generalizing c m with
  | zero => cases s with | stop i => exact terminalRegret_eq c m i
  | succ k ih =>
    cases s with
    | sample i next =>
      change (∫ z, expectedRegret (next z) (advance c i) (shift m i z)
        ∂gaussianReal 0 (innovation c i)) = _
      have he (z : ℝ) := ih (next z) (advance c i) (shift m i z) (hs.1 z)
        (advance_positive c i hc)
      simp_rw [he]
      rw [integral_sub (shifted_integrable (potential_lipschitz _) m i _) hs.2,
        potential_tower c m i (hc i)]
      rfl

/-- Theorem1 for any finite nonempty arm set and arbitrary positive real
posterior precisions, using actual posterior expected simple regret. -/
theorem theorem1_global_bayes_optimality {k : ℕ} (s : Strategy (A := A) k)
    (c : Precisions A) (m : Means A) (hs : Admissible s c m) (hc : ∀ a, 0 < c a) :
    expectedRegret (optimal k c m) c m ≤ expectedRegret s c m := by
  rw [expectedRegret_eq _ _ _ (optimal_admissible k c m) hc,
    expectedRegret_eq s c m hs hc, optimal_utility]
  exact sub_le_sub_left (utility_le_value s c m hs) _

/-- External randomization may choose any admissible adaptive strategy tree.
Its actual expected regret is the integral over that independent seed. -/
theorem theorem1_randomized_bayes_optimality {S : Type*} [MeasurableSpace S]
    (seedLaw : Measure S) [IsProbabilityMeasure seedLaw] {k : ℕ}
    (s : S → Strategy (A := A) k) (c : Precisions A) (m : Means A)
    (hs : ∀ᵐ r ∂seedLaw, Admissible (s r) c m) (hc : ∀ a, 0 < c a)
    (hi : Integrable (fun r => expectedRegret (s r) c m) seedLaw) :
    expectedRegret (optimal k c m) c m ≤ ∫ r, expectedRegret (s r) c m ∂seedLaw := by
  have h := integral_mono_ae (integrable_const (expectedRegret (optimal k c m) c m)) hi
    (hs.mono (fun r hr => theorem1_global_bayes_optimality (s r) c m hr hc))
  simpa using h

end Nonempty

end
end Junpei.FinitePosterior
