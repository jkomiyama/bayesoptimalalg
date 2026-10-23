import Draft0.Gaussian
import Mathlib.Topology.MetricSpace.Lipschitz

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FiniteGaussian

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

abbrev Means (A : Type*) := A → ℝ
abbrev Precisions (A : Type*) := A → ℝ≥0

noncomputable section

def maxAll (m : Means A) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty m

omit [DecidableEq A] in
theorem coordinate_le_maxAll (m : Means A) (i : A) : m i ≤ maxAll m :=
  Finset.le_sup' m (Finset.mem_univ i)

omit [DecidableEq A] in
theorem maxAll_le (m : Means A) (a : ℝ) (h : ∀ i, m i ≤ a) : maxAll m ≤ a :=
  Finset.sup'_le Finset.univ_nonempty m (fun i _ => h i)

omit [DecidableEq A] in
theorem maxAll_attained (m : Means A) : ∃ i, m i = maxAll m := by
  obtain ⟨i, _, hi⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty m
  exact ⟨i, hi.symm⟩

def advance (c : Precisions A) (i : A) : Precisions A :=
  fun j => c j + if j = i then 1 else 0

def shift (m : Means A) (i : A) (z : ℝ) : Means A :=
  fun j => m j + if j = i then z else 0

def innovation (c : Precisions A) (i : A) : ℝ≥0 := Draft0.Gaussian.updateVariance (c i)

omit [DecidableEq A] in
theorem maxAll_lipschitz : LipschitzWith 1 (maxAll (A := A)) := by
  apply LipschitzWith.of_dist_le_mul
  intro m n
  simp only [NNReal.coe_one, one_mul, Real.dist_eq]
  apply abs_le.mpr
  have hu (m n : Means A) : maxAll m ≤ maxAll n + dist m n := by
    apply maxAll_le
    intro i
    have h := (LipschitzWith.eval i : LipschitzWith 1 (fun x : Means A => x i)).dist_le_mul m n
    simp only [NNReal.coe_one, one_mul, Real.dist_eq] at h
    have hupper := (abs_le.mp h).2
    have hcoord := coordinate_le_maxAll n i
    linarith
  have hm := hu m n
  have hn := hu n m
  rw [dist_comm n m] at hn
  constructor <;> linarith

omit [Fintype A] [Nonempty A] in
theorem shift_sub_shift (m n : Means A) (i : A) (z : ℝ) :
    shift m i z - shift n i z = m - n := by
  ext j
  simp only [shift, Pi.sub_apply]
  ring

omit [Nonempty A] in
theorem shift_norm_bound (m : Means A) (i : A) (z : ℝ) : ‖shift m i z - m‖ ≤ |z| := by
  apply (pi_norm_le_iff_of_nonneg (abs_nonneg z)).mpr
  intro j
  by_cases h : j = i <;> simp [shift, h, Real.norm_eq_abs]

omit [Fintype A] [Nonempty A] in
theorem shift_continuous (m : Means A) (i : A) : Continuous (shift m i) := by
  apply continuous_pi
  intro j
  by_cases h : j = i <;> simp [shift, h] <;> fun_prop

omit [Nonempty A] in
theorem shifted_integrable {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (m : Means A) (i : A) (v : ℝ≥0) :
    Integrable (fun z : ℝ => f (shift m i z)) (gaussianReal 0 v) := by
  have hb (z : ℝ) : ‖f (shift m i z)‖ ≤ |f m| + |z| := by
    have h := hf.dist_le_mul (shift m i z) m
    simp only [NNReal.coe_one, one_mul, dist_eq_norm, Real.norm_eq_abs] at h
    have hn := shift_norm_bound m i z
    have ht := abs_add_le (f (shift m i z) - f m) (f m)
    rw [sub_add_cancel] at ht
    rw [Real.norm_eq_abs]
    linarith
  exact ((integrable_const |f m|).add (Draft0.Gaussian.gaussian_id_integrable 0 v).norm).mono'
    (hf.continuous.comp (shift_continuous m i)).measurable.aestronglyMeasurable
    (Filter.Eventually.of_forall hb)

omit [Nonempty A] in
theorem integral_shift_lipschitz {f : Means A → ℝ} (hf : LipschitzWith 1 f)
    (i : A) (v : ℝ≥0) :
    LipschitzWith 1 (fun m => ∫ z : ℝ, f (shift m i z) ∂gaussianReal 0 v) := by
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

omit [DecidableEq A] in
theorem maxAll_family_lipschitz {f : A → Means A → ℝ}
    (hf : ∀ i, LipschitzWith 1 (f i)) :
    LipschitzWith 1 (fun m => maxAll (fun i => f i m)) := by
  apply LipschitzWith.of_dist_le_mul
  intro m n
  simp only [NNReal.coe_one, one_mul, Real.dist_eq]
  have hu (m n : Means A) : maxAll (fun i => f i m) ≤
      maxAll (fun i => f i n) + dist m n := by
    apply maxAll_le
    intro i
    have h := (hf i).dist_le_mul m n
    simp only [NNReal.coe_one, one_mul, Real.dist_eq] at h
    have hb := (abs_le.mp h).2
    have hi := coordinate_le_maxAll (fun i => f i n) i
    linarith
  have hm := hu m n
  have hn := hu n m
  rw [dist_comm n m] at hn
  exact abs_le.mpr ⟨by linarith, by linarith⟩

def value : ℕ → Precisions A → Means A → ℝ
  | 0, _, m => maxAll m
  | k + 1, c, m => maxAll (fun i =>
      ∫ z : ℝ, value k (advance c i) (shift m i z) ∂gaussianReal 0 (innovation c i))

def action (k : ℕ) (c : Precisions A) (m : Means A) (i : A) : ℝ :=
  ∫ z : ℝ, value k (advance c i) (shift m i z) ∂gaussianReal 0 (innovation c i)

theorem value_succ (k : ℕ) (c : Precisions A) (m : Means A) :
    value (k + 1) c m = maxAll (action k c m) := rfl

theorem value_lipschitz (k : ℕ) (c : Precisions A) : LipschitzWith 1 (value k c) := by
  induction k generalizing c with
  | zero => exact maxAll_lipschitz
  | succ k ih =>
    exact maxAll_family_lipschitz (fun i =>
      integral_shift_lipschitz (ih (advance c i)) i (innovation c i))

theorem action_integrable (k : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    Integrable (fun z => value k (advance c i) (shift m i z))
      (gaussianReal 0 (innovation c i)) :=
  shifted_integrable (value_lipschitz k (advance c i)) m i (innovation c i)

theorem action_le_value (k : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    action k c m i ≤ value (k + 1) c m := coordinate_le_maxAll _ i

def choose (k : ℕ) (c : Precisions A) (m : Means A) : A :=
  Classical.choose (maxAll_attained (action k c m))

def recommend (m : Means A) : A := Classical.choose (maxAll_attained m)

theorem choose_optimal (k : ℕ) (c : Precisions A) (m : Means A) :
    action k c m (choose k c m) = value (k + 1) c m :=
  Classical.choose_spec (maxAll_attained (action k c m))

omit [DecidableEq A] in
theorem recommend_max (m : Means A) : m (recommend m) = maxAll m :=
  Classical.choose_spec (maxAll_attained m)

inductive Strategy : ℕ → Type _
  | stop (i : A) : Strategy 0
  | sample {k : ℕ} (i : A) (next : ℝ → Strategy k) : Strategy (k + 1)

def utility : {k : ℕ} → Strategy (A := A) k → Precisions A → Means A → ℝ
  | 0, .stop i, _, m => m i
  | _ + 1, .sample i next, c, m =>
      ∫ z : ℝ, utility (next z) (advance c i) (shift m i z)
        ∂gaussianReal 0 (innovation c i)

def Admissible : {k : ℕ} → Strategy (A := A) k → Precisions A → Means A → Prop
  | 0, .stop _, _, _ => True
  | _ + 1, .sample i next, c, m =>
      (∀ z, Admissible (next z) (advance c i) (shift m i z)) ∧
      Integrable (fun z => utility (next z) (advance c i) (shift m i z))
        (gaussianReal 0 (innovation c i))

def optimal : (k : ℕ) → Precisions A → Means A → Strategy (A := A) k
  | 0, _, m => .stop (recommend m)
  | k + 1, c, m =>
    let i := choose k c m
    .sample i (fun z => optimal k (advance c i) (shift m i z))

theorem optimal_utility (k : ℕ) (c : Precisions A) (m : Means A) :
    utility (optimal k c m) c m = value k c m := by
  induction k generalizing c m with
  | zero => exact recommend_max m
  | succ k ih => simp only [optimal, utility, ih]; exact choose_optimal k c m

theorem optimal_admissible (k : ℕ) (c : Precisions A) (m : Means A) :
    Admissible (optimal k c m) c m := by
  induction k generalizing c m with
  | zero => trivial
  | succ k ih =>
    constructor
    · intro z; exact ih _ _
    · simp only [optimal_utility]
      exact action_integrable k c m (choose k c m)

theorem utility_le_value {k : ℕ} (s : Strategy (A := A) k) (c : Precisions A) (m : Means A)
    (hs : Admissible s c m) : utility s c m ≤ value k c m := by
  induction k generalizing c m with
  | zero => cases s with | stop i => exact coordinate_le_maxAll m i
  | succ k ih =>
    cases s with
    | sample i next =>
      exact (integral_mono hs.2 (action_integrable k c m i)
        (fun z => ih (next z) _ _ (hs.1 z))).trans (action_le_value k c m i)

/-- Theorem1's global finite-horizon optimality, for any finite nonempty arm
set and arbitrary positive real Gaussian precisions. -/
theorem optimal_dominates {k : ℕ} (s : Strategy (A := A) k)
    (c : Precisions A) (m : Means A) (hs : Admissible s c m) :
    utility s c m ≤ utility (optimal k c m) c m := by
  rw [optimal_utility]
  exact utility_le_value s c m hs

end
end Junpei.FiniteGaussian
