import Draft0.Gaussian
import Mathlib.Topology.MetricSpace.Lipschitz

/-!
The three-arm Gaussian posterior decision process. Counts and means are the
paper's sufficient statistics. Values are expected terminal posterior means;
the posterior expectation of the best latent mean is an action-independent
additive constant in expected regret.
-/

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Draft0.BayesianModel

abbrev Arm := Fin 3
abbrev Counts := Arm → ℕ
abbrev Means := Arm → ℝ

noncomputable section

def advance (c : Counts) (i : Arm) : Counts :=
  fun j => c j + if j = i then 1 else 0

def shift (m : Means) (i : Arm) (z : ℝ) : Means :=
  fun j => m j + if j = i then z else 0

def innovation (c : Counts) (i : Arm) : ℝ≥0 :=
  Gaussian.updateVariance (c i : ℝ≥0)

def terminal (m : Means) : ℝ := max (m 0) (max (m 1) (m 2))

def value : ℕ → Counts → Means → ℝ
  | 0, _, m => terminal m
  | k + 1, c, m =>
    let a := fun i : Arm => ∫ z : ℝ,
      value k (advance c i) (shift m i z) ∂gaussianReal 0 (innovation c i)
    max (a 0) (max (a 1) (a 2))

def action (k : ℕ) (c : Counts) (m : Means) (i : Arm) : ℝ :=
  ∫ z : ℝ, value k (advance c i) (shift m i z)
    ∂gaussianReal 0 (innovation c i)

theorem value_succ (k : ℕ) (c : Counts) (m : Means) :
    value (k + 1) c m = max (action k c m 0)
      (max (action k c m 1) (action k c m 2)) := rfl

theorem shift_sub_shift (m n : Means) (i : Arm) (z : ℝ) :
    shift m i z - shift n i z = m - n := by
  ext j
  simp only [shift, Pi.sub_apply]
  ring

theorem shift_lipschitz (i : Arm) (z : ℝ) :
    LipschitzWith 1 (fun m : Means => shift m i z) := by
  apply LipschitzWith.of_dist_le_mul
  intro m n
  simp only [dist_eq_norm, shift_sub_shift, NNReal.coe_one, one_mul, le_refl]

theorem shift_norm_bound (m : Means) (i : Arm) (z : ℝ) :
    ‖shift m i z - m‖ ≤ |z| := by
  apply (pi_norm_le_iff_of_nonneg (abs_nonneg z)).mpr
  intro j
  by_cases h : j = i <;> simp [shift, h, Real.norm_eq_abs]

theorem shift_continuous_noise (m : Means) (i : Arm) :
    Continuous (shift m i) := by
  apply continuous_pi
  intro j
  by_cases h : j = i <;> simp [shift, h] <;> fun_prop

theorem terminal_lipschitz : LipschitzWith 1 terminal := by
  have h (i : Arm) : LipschitzWith 1 (fun m : Means => m i) :=
    LipschitzWith.eval i
  simpa only [max_self] using (h 0).max ((h 1).max (h 2))

theorem shifted_integrable {f : Means → ℝ} (hf : LipschitzWith 1 f)
    (m : Means) (i : Arm) (v : ℝ≥0) :
    Integrable (fun z : ℝ => f (shift m i z)) (gaussianReal 0 v) := by
  have hb (z : ℝ) : ‖f (shift m i z)‖ ≤ |f m| + |z| := by
    have h := hf.dist_le_mul (shift m i z) m
    simp only [NNReal.coe_one, one_mul, dist_eq_norm, Real.norm_eq_abs] at h
    have hn := shift_norm_bound m i z
    calc
      ‖f (shift m i z)‖ = |f (shift m i z)| := Real.norm_eq_abs _
      _ ≤ |f (shift m i z) - f m| + |f m| := by
        simpa only [sub_add_cancel] using abs_add_le (f (shift m i z) - f m) (f m)
      _ ≤ |f m| + |z| := by linarith
  exact ((integrable_const |f m|).add (Gaussian.gaussian_id_integrable 0 v).norm).mono'
    (hf.continuous.comp (shift_continuous_noise m i)).measurable.aestronglyMeasurable
    (Filter.Eventually.of_forall hb)

theorem integral_shift_lipschitz {f : Means → ℝ} (hf : LipschitzWith 1 f)
    (i : Arm) (v : ℝ≥0) :
    LipschitzWith 1 (fun m : Means =>
      ∫ z : ℝ, f (shift m i z) ∂gaussianReal 0 v) := by
  apply LipschitzWith.of_dist_le_mul
  intro m n
  simp only [NNReal.coe_one, one_mul, dist_eq_norm]
  rw [← integral_sub (shifted_integrable hf m i v) (shifted_integrable hf n i v)]
  have hb : ∀ᵐ z : ℝ ∂gaussianReal 0 v,
      ‖f (shift m i z) - f (shift n i z)‖ ≤ ‖m - n‖ := by
    apply Filter.Eventually.of_forall
    intro z
    have h := hf.dist_le_mul (shift m i z) (shift n i z)
    simpa only [dist_eq_norm, shift_sub_shift, NNReal.coe_one, one_mul] using h
  simpa using norm_integral_le_of_norm_le_const hb

theorem value_lipschitz (k : ℕ) (c : Counts) : LipschitzWith 1 (value k c) := by
  induction k generalizing c with
  | zero => exact terminal_lipschitz
  | succ k ih =>
    have h (i : Arm) : LipschitzWith 1 (action k c · i) :=
      integral_shift_lipschitz (ih (advance c i)) i (innovation c i)
    simpa only [value_succ, max_self] using (h 0).max ((h 1).max (h 2))

theorem action_integrable (k : ℕ) (c : Counts) (m : Means) (i : Arm) :
    Integrable (fun z : ℝ => value k (advance c i) (shift m i z))
      (gaussianReal 0 (innovation c i)) :=
  shifted_integrable (value_lipschitz k (advance c i)) m i (innovation c i)

theorem action_lipschitz (k : ℕ) (c : Counts) (i : Arm) :
    LipschitzWith 1 (action k c · i) :=
  integral_shift_lipschitz (value_lipschitz k (advance c i)) i (innovation c i)

theorem action_le_value (k : ℕ) (c : Counts) (m : Means) (i : Arm) :
    action k c m i ≤ value (k + 1) c m := by
  rw [value_succ]
  fin_cases i
  · exact le_max_left _ _
  · exact (le_max_left _ _).trans (le_max_right _ _)
  · exact (le_max_right _ _).trans (le_max_right _ _)

def choose (k : ℕ) (c : Counts) (m : Means) : Arm :=
  if action k c m 0 ≥ max (action k c m 1) (action k c m 2) then 0
  else if action k c m 1 ≥ action k c m 2 then 1 else 2

theorem choose_optimal (k : ℕ) (c : Counts) (m : Means) :
    action k c m (choose k c m) = value (k + 1) c m := by
  rw [value_succ]
  unfold choose
  split_ifs with h₀ h₁
  · exact (max_eq_left h₀).symm
  · rw [max_eq_right (le_of_not_ge h₀), max_eq_left h₁]
  · rw [max_eq_right (le_of_not_ge h₀), max_eq_right (le_of_not_ge h₁)]

def recommend (m : Means) : Arm :=
  if m 0 ≥ max (m 1) (m 2) then 0 else if m 1 ≥ m 2 then 1 else 2

theorem recommend_max (m : Means) : m (recommend m) = terminal m := by
  unfold recommend terminal
  split_ifs with h₀ h₁
  · exact (max_eq_left h₀).symm
  · rw [max_eq_right (le_of_not_ge h₀), max_eq_left h₁]
  · rw [max_eq_right (le_of_not_ge h₀), max_eq_right (le_of_not_ge h₁)]

theorem recommend_greedy (m : Means) (i : Arm) : m i ≤ m (recommend m) := by
  rw [recommend_max m]
  fin_cases i
  · exact le_max_left _ _
  · exact (le_max_left _ _).trans (le_max_right _ _)
  · exact (le_max_right _ _).trans (le_max_right _ _)

end

end Draft0.BayesianModel
