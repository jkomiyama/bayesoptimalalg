import Junpei.MockTie

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.JointGaussianMeasurable

noncomputable section

set_option linter.unusedSectionVars false

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

open Junpei.FiniteGaussian Junpei.FinitePosterior

/-- A parameter-dependent centered Gaussian is a measurable scaling of one
fixed standard Gaussian, including the zero-variance case. -/
theorem standard_scaled_law (v : ℝ≥0) :
    (gaussianReal 0 1).map (fun x : ℝ => Real.sqrt (v : ℝ) * x) =
      gaussianReal 0 v := by
  have hv : (⟨(Real.sqrt (v : ℝ)) ^ 2, sq_nonneg _⟩ : ℝ≥0) * 1 = v := by
    ext
    simp only [mul_one]
    exact Real.sq_sqrt v.property
  rw [gaussianReal_map_const_mul, mul_zero, hv]

theorem gaussian_integral_measurable {X : Type*} [MeasurableSpace X]
    (v : X → ℝ≥0) (hv : Measurable v) (f : X × ℝ → ℝ) (hf : Measurable f) :
    Measurable (fun x => ∫ z, f (x, z) ∂gaussianReal 0 (v x)) := by
  have hg : Measurable (fun p : X × ℝ =>
      f (p.1, Real.sqrt (v p.1 : ℝ) * p.2)) := by
    apply hf.comp
    exact measurable_fst.prodMk (((hv.comp measurable_fst).coe_nnreal_real.sqrt).mul
      measurable_snd)
  have h := hg.stronglyMeasurable.integral_prod_right' (ν := gaussianReal 0 1)
  convert h.measurable using 1
  funext x
  rw [← standard_scaled_law (v x), integral_map (by fun_prop)]
  exact (hf.comp (measurable_const.prodMk measurable_id)).aestronglyMeasurable

theorem advance_measurable (i : A) : Measurable (fun c : Precisions A => advance c i) := by
  apply measurable_pi_lambda
  intro a
  exact (measurable_pi_apply a).add measurable_const

theorem innovation_measurable (i : A) : Measurable (fun c : Precisions A => innovation c i) := by
  unfold innovation Draft0.Gaussian.updateVariance
  fun_prop

theorem shift_joint_measurable (i : A) :
    Measurable (fun p : Means A × ℝ => shift p.1 i p.2) := by
  apply measurable_pi_lambda
  intro a
  by_cases h : a = i <;> simp only [shift, h, ite_true, ite_false]
  · fun_prop
  · fun_prop

theorem transition_joint_measurable (i : A) :
    Measurable (fun p : (Precisions A × Means A) × ℝ =>
      (advance p.1.1 i, shift p.1.2 i p.2)) :=
  ((advance_measurable i).comp (measurable_fst.comp measurable_fst)).prodMk
    ((shift_joint_measurable i).comp
      ((measurable_snd.comp measurable_fst).prodMk measurable_snd))

/-- The Bellman values are jointly Borel in all real Gaussian precisions and
all posterior means; the precision vector is not fixed. -/
theorem value_joint_measurable (k : ℕ) :
    Measurable (fun p : Precisions A × Means A => value k p.1 p.2) := by
  induction k with
  | zero => exact maxAll_lipschitz.continuous.measurable.comp measurable_snd
  | succ k ih =>
    apply maxAll_lipschitz.continuous.measurable.comp
    apply measurable_pi_lambda
    intro i
    apply gaussian_integral_measurable
      (fun p : Precisions A × Means A => innovation p.1 i)
      ((innovation_measurable i).comp measurable_fst)
      (fun p => value k (advance p.1.1 i) (shift p.1.2 i p.2))
    exact ih.comp (transition_joint_measurable i)

theorem action_joint_measurable (k : ℕ) (i : A) :
    Measurable (fun p : Precisions A × Means A => action k p.1 p.2 i) := by
  apply gaussian_integral_measurable
    (fun p : Precisions A × Means A => innovation p.1 i)
    ((innovation_measurable i).comp measurable_fst)
    (fun p => value k (advance p.1.1 i) (shift p.1.2 i p.2))
  exact (value_joint_measurable k).comp (transition_joint_measurable i)

def standardArrayLaw : Measure (Means A) := Measure.pi (fun _ => gaussianReal 0 1)

instance standardArrayLaw_probability : IsProbabilityMeasure (standardArrayLaw (A := A)) := by
  unfold standardArrayLaw
  infer_instance

theorem posterior_standard_law (c : Precisions A) (m : Means A) :
    (standardArrayLaw (A := A)).map
        (fun x a => m a + Real.sqrt ((c a)⁻¹ : ℝ≥0) * x a) = posteriorLaw c m := by
  have h : MeasurePreserving (fun x a => Real.sqrt ((c a)⁻¹ : ℝ≥0) * x a)
      (standardArrayLaw (A := A)) (centeredLaw c) := by
    apply measurePreserving_pi
    intro a
    exact ⟨by fun_prop, standard_scaled_law (c a)⁻¹⟩
  unfold posteriorLaw
  rw [← h.map_eq, Measure.map_map (by fun_prop) h.measurable]
  rfl

theorem potential_joint_measurable :
    Measurable (fun p : Precisions A × Means A => potential p.1 p.2) := by
  have hf : Measurable (fun p : (Precisions A × Means A) × Means A =>
      maxAll (fun a => p.1.2 a + Real.sqrt ((p.1.1 a)⁻¹ : ℝ≥0) * p.2 a)) := by
    apply maxAll_lipschitz.continuous.measurable.comp
    apply measurable_pi_lambda
    intro a
    fun_prop
  have hm := hf.stronglyMeasurable.integral_prod_right'
    (ν := standardArrayLaw (A := A))
  convert hm.measurable using 1
  funext p
  change expectation maxAll p.1 p.2 = _
  unfold expectation
  rw [← posterior_integral maxAll_lipschitz p.1 p.2]
  rw [← posterior_standard_law p.1 p.2, integral_map (by
    apply Measurable.aemeasurable
    apply measurable_pi_lambda
    intro a
    fun_prop)
    maxAll_lipschitz.continuous.measurable.aestronglyMeasurable]

theorem loss_joint_measurable (k : ℕ) :
    Measurable (fun p : Precisions A × Means A => loss k p.1 p.2) :=
  potential_joint_measurable.sub (value_joint_measurable k)

theorem actionLoss_joint_measurable (k : ℕ) (i : A) :
    Measurable (fun p : Precisions A × Means A => actionLoss k p.1 p.2 i) := by
  apply gaussian_integral_measurable
    (fun p : Precisions A × Means A => innovation p.1 i)
    ((innovation_measurable i).comp measurable_fst)
    (fun p => loss k (advance p.1.1 i) (shift p.1.2 i p.2))
  exact (loss_joint_measurable k).comp (transition_joint_measurable i)

section Ordered
variable [LinearOrder A] [MeasurableSpace A] [MeasurableSingletonClass A]

def minimumOptimal (k : ℕ) (c : Precisions A) (m : Means A) : A :=
  Junpei.MockTie.minimumMaximizer (action k c m)

theorem minimumOptimal_joint_measurable (k : ℕ) :
    Measurable (fun p : Precisions A × Means A => minimumOptimal k p.1 p.2) := by
  apply Junpei.MockTie.minimumMaximizer_measurable.comp
  exact measurable_pi_lambda _ (action_joint_measurable k)

theorem minimumOptimal_attains (k : ℕ) (c : Precisions A) (m : Means A) :
    action k c m (minimumOptimal k c m) = value (k + 1) c m :=
  Junpei.MockTie.minimumMaximizer_attains _

theorem minimumOptimal_le (k : ℕ) (c : Precisions A) (m : Means A) (i : A)
    (hi : action k c m i = value (k + 1) c m) : minimumOptimal k c m ≤ i :=
  Junpei.MockTie.minimumMaximizer_le _ _ hi

theorem minimumOptimal_actionLoss (k : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) :
    actionLoss k c m (minimumOptimal k c m) = loss (k + 1) c m := by
  rw [actionLoss_eq _ _ _ _ (hc _), minimumOptimal_attains]
  rfl

theorem minimumOptimal_minimizes (k : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) (i : A) :
    actionLoss k c m (minimumOptimal k c m) ≤ actionLoss k c m i := by
  rw [minimumOptimal_actionLoss _ _ _ hc, actionLoss_eq _ _ _ _ (hc i)]
  exact sub_le_sub_left (action_le_value k c m i) _

theorem minimumOptimal_le_minimizer (k : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) (i : A)
    (hi : actionLoss k c m i = loss (k + 1) c m) : minimumOptimal k c m ≤ i := by
  apply minimumOptimal_le
  rw [actionLoss_eq _ _ _ _ (hc i)] at hi
  unfold loss at hi
  linarith

end Ordered
end
end Junpei.JointGaussianMeasurable
