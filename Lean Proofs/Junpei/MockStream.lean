import Junpei.FiniteModelExclusion

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockStream

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.FiniteModelExclusion

variable {A : Type*} [Fintype A] [DecidableEq A]

noncomputable section

/-- A finite reward array, indexed by arm and calendar sampling opportunity.
Unused entries remain independent Gaussian noise conditional on the latent means. -/
abbrev RawStream (A : Type*) (H : ℕ) := A → Fin H → ℝ
abbrev Block (H : ℕ) := ℝ × (Fin H → ℝ)
abbrev OtherData (A : Type*) (H : ℕ) (i : A) :=
  ({a : A // a ≠ i} → ℝ) × ({a : A // a ≠ i} → Fin H → ℝ)
abbrev World (A : Type*) (H : ℕ) (i : A) := (Block H × Block H) × OtherData A H i

def noiseLaw (H : ℕ) : Measure (Fin H → ℝ) :=
  Measure.pi (fun _ => gaussianReal 0 1)

instance noiseLaw_probability (H : ℕ) : IsProbabilityMeasure (noiseLaw H) := by
  unfold noiseLaw; infer_instance

def blockLaw (H : ℕ) (v : ℝ≥0) : Measure (Block H) :=
  (gaussianReal 0 v).prod (noiseLaw H)

instance blockLaw_probability (H : ℕ) (v : ℝ≥0) :
    IsProbabilityMeasure (blockLaw H v) := by unfold blockLaw; infer_instance

def otherDataLaw (H : ℕ) (c : Precisions A) (i : A) : Measure (OtherData A H i) :=
  (otherLaw c i).prod (Measure.pi (fun _ : {a : A // a ≠ i} => noiseLaw H))

instance otherDataLaw_probability (H : ℕ) (c : Precisions A) (i : A) :
    IsProbabilityMeasure (otherDataLaw H c i) := by unfold otherDataLaw; infer_instance

/-- The real arm-i latent/noise block and its independent posterior-predictive
copy are independent of each other and all the remaining arms' latent/noise data. -/
def law (H : ℕ) (c : Precisions A) (i : A) : Measure (World A H i) :=
  ((blockLaw H (c i)⁻¹).prod (blockLaw H (c i)⁻¹)).prod (otherDataLaw H c i)

instance law_probability (H : ℕ) (c : Precisions A) (i : A) :
    IsProbabilityMeasure (law H c i) := by unfold law; infer_instance

def swap (H : ℕ) (i : A) (w : World A H i) : World A H i :=
  ((w.1.2, w.1.1), w.2)

theorem swap_preserving (H : ℕ) (c : Precisions A) (i : A) :
    MeasurePreserving (swap H i) (law H c i) (law H c i) := by
  exact (Measure.measurePreserving_swap (μ := blockLaw H (c i)⁻¹)
    (ν := blockLaw H (c i)⁻¹)).prod (MeasurePreserving.id _)

omit [Fintype A] [DecidableEq A] in
theorem swap_involution (H : ℕ) (i : A) (w : World A H i) :
    swap H i (swap H i w) = w := rfl

def theta (H : ℕ) (m : Means A) (i : A) (w : World A H i) : Means A :=
  m + join i (w.1.1.1, w.2.1)

def copyMean (H : ℕ) (m : Means A) (i : A) (w : World A H i) : ℝ :=
  m i + w.1.2.1

def realStream (H : ℕ) (m : Means A) (i : A) (w : World A H i) : RawStream A H :=
  fun a t => if h : a = i then m a + w.1.1.1 + w.1.1.2 t
    else m a + w.2.1 ⟨a, h⟩ + w.2.2 ⟨a, h⟩ t

def mockStream (H : ℕ) (m : Means A) (i : A) (w : World A H i) : RawStream A H :=
  realStream H m i (swap H i w)

theorem theta_measurable (H : ℕ) (m : Means A) (i : A) : Measurable (theta H m i) := by
  unfold theta
  exact measurable_const.add ((join_continuous i).measurable.comp (by fun_prop))

omit [Fintype A] in
theorem realStream_measurable (H : ℕ) (m : Means A) (i : A) :
    Measurable (realStream H m i) := by
  apply measurable_pi_lambda
  intro a
  apply measurable_pi_lambda
  intro t
  by_cases h : a = i <;> simp only [realStream, h, dite_true, dite_false] <;> fun_prop

omit [Fintype A] in
theorem mockStream_measurable (H : ℕ) (m : Means A) (i : A) :
    Measurable (mockStream H m i) :=
  (realStream_measurable H m i).comp (by unfold swap; fun_prop)

omit [Fintype A] in
theorem stream_swap (H : ℕ) (m : Means A) (i : A) (w : World A H i) :
    realStream H m i (swap H i w) = mockStream H m i w := rfl

omit [Fintype A] in
theorem stream_other_unchanged (H : ℕ) (m : Means A) (i : A) (a : A) (hai : a ≠ i)
    (w : World A H i) : mockStream H m i w a = realStream H m i w a := by
  funext t
  simp [mockStream, realStream, swap, hai]

omit [Fintype A] in
theorem theta_other_unchanged (H : ℕ) (m : Means A) (i : A) (a : A) (hai : a ≠ i)
    (w : World A H i) : theta H m i (swap H i w) a = theta H m i w a := by
  simp [theta, join, swap, hai]

theorem latent_projection_preserving (H : ℕ) (c : Precisions A) (i : A) :
    MeasurePreserving (fun w : World A H i => (w.1.1.1, w.2.1)) (law H c i)
      ((gaussianReal 0 (c i)⁻¹).prod (otherLaw c i)) := by
  exact ((measurePreserving_fst (μ := gaussianReal 0 (c i)⁻¹) (ν := noiseLaw H)).comp
    (measurePreserving_fst (μ := blockLaw H (c i)⁻¹) (ν := blockLaw H (c i)⁻¹))).prod
    (measurePreserving_fst (μ := otherLaw c i)
      (ν := Measure.pi (fun _ : {a : A // a ≠ i} => noiseLaw H)))

theorem theta_law (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    (law H c i).map (theta H m i) = posteriorLaw c m := by
  have hj : MeasurePreserving (join i)
      ((gaussianReal 0 (c i)⁻¹).prod (otherLaw c i)) (centeredLaw c) :=
    ⟨(join_continuous i).measurable, join_law c i⟩
  have h := hj.comp (latent_projection_preserving H c i)
  unfold theta posteriorLaw
  rw [← h.map_eq, Measure.map_map (by fun_prop) h.measurable]
  rfl

theorem theta_preserving (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    MeasurePreserving (theta H m i) (law H c i) (posteriorLaw c m) :=
  ⟨theta_measurable H m i, theta_law H c m i⟩

/-- The copy latent and its noise array have exactly the same joint law as
the real block; this equality includes the dependence shared across all rewards. -/
theorem copy_block_law (H : ℕ) (c : Precisions A) (i : A) :
    (law H c i).map (fun w => w.1.2) = blockLaw H (c i)⁻¹ := by
  exact ((measurePreserving_snd (μ := blockLaw H (c i)⁻¹) (ν := blockLaw H (c i)⁻¹)).comp
    (measurePreserving_fst (μ := (blockLaw H (c i)⁻¹).prod (blockLaw H (c i)⁻¹))
      (ν := otherDataLaw H c i))).map_eq

/-- Rearrangement isolates the auxiliary copy from the entire real world. -/
theorem isolate_copy_preserving (H : ℕ) (c : Precisions A) (i : A) :
    MeasurePreserving (fun w : World A H i => ((w.1.1, w.2), w.1.2)) (law H c i)
      (((blockLaw H (c i)⁻¹).prod (otherDataLaw H c i)).prod (blockLaw H (c i)⁻¹)) := by
  let b := blockLaw H (c i)⁻¹
  let o := otherDataLaw H c i
  have h1 := measurePreserving_prodAssoc b b o
  have h2 := (MeasurePreserving.id b).prod (Measure.measurePreserving_swap (μ := b) (ν := o))
  have h3 := MeasurePreserving.symm MeasurableEquiv.prodAssoc
    (measurePreserving_prodAssoc b o b)
  exact h3.comp (h2.comp h1)

/-- The posterior-predictive block is independent of all real parameters,
real observation noises, and the remaining arms' data. -/
theorem copy_independent_of_real_data (H : ℕ) (c : Precisions A) (i : A) :
    IndepFun (fun w : World A H i => w.1.2) (fun w => (w.1.1, w.2)) (law H c i) := by
  have h := isolate_copy_preserving H c i
  apply (indepFun_iff_map_prod_eq_prod_map_map (by fun_prop) (by fun_prop)).mpr
  have hfst := measurePreserving_fst
    (μ := (blockLaw H (c i)⁻¹).prod (otherDataLaw H c i)) (ν := blockLaw H (c i)⁻¹)
  have hsnd := measurePreserving_snd
    (μ := (blockLaw H (c i)⁻¹).prod (otherDataLaw H c i)) (ν := blockLaw H (c i)⁻¹)
  have hswap := (Measure.measurePreserving_swap
    (μ := (blockLaw H (c i)⁻¹).prod (otherDataLaw H c i)) (ν := blockLaw H (c i)⁻¹)).comp h
  have ha := hsnd.comp h
  have hb := hfst.comp h
  have haEq : (law H c i).map (fun w => w.1.2) = blockLaw H (c i)⁻¹ := by
    simpa only [Function.comp_def] using ha.map_eq
  have hbEq : (law H c i).map (fun w => (w.1.1, w.2)) =
      (blockLaw H (c i)⁻¹).prod (otherDataLaw H c i) := by
    simpa only [Function.comp_def] using hb.map_eq
  rw [haEq, hbEq]
  simpa only [Function.comp_def, Prod.swap] using hswap.map_eq

/-- Given a latent mean, the finite pseudo-noise array produces iid Gaussian
rewards of variance one; the common latent is drawn only once for the stream. -/
theorem rewards_given_mean_law (H : ℕ) (u : ℝ) :
    (noiseLaw H).map (fun e : Fin H → ℝ => fun t => u + e t) =
      Measure.pi (fun _ : Fin H => gaussianReal u 1) := by
  unfold noiseLaw
  rw [Measure.pi_map_pi (fun _ => (by fun_prop :
    AEMeasurable (fun x : ℝ => u + x) (gaussianReal 0 1)))]
  simp only [gaussianReal_map_const_add, zero_add]

theorem copy_mean_law (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    (law H c i).map (copyMean H m i) = gaussianReal (m i) (c i)⁻¹ := by
  have hm := (measurePreserving_fst (μ := gaussianReal 0 (c i)⁻¹) (ν := noiseLaw H)).map_eq
  change (blockLaw H (c i)⁻¹).map Prod.fst = gaussianReal 0 (c i)⁻¹ at hm
  have ht := gaussianReal_map_const_add (μ := (0 : ℝ)) (v := (c i)⁻¹) (m i)
  rw [← hm, ← copy_block_law H c i,
    Measure.map_map (by fun_prop) (by fun_prop),
    Measure.map_map (by fun_prop) (by fun_prop)] at ht
  simpa [copyMean, Function.comp_def] using ht

theorem copy_mean_independent_of_theta (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    IndepFun (copyMean H m i) (theta H m i) (law H c i) := by
  have hf : Measurable (fun b : Block H => m i + b.1) := by fun_prop
  have hg : Measurable (fun p : Block H × OtherData A H i =>
      m + join i (p.1.1, p.2.1)) :=
    measurable_const.add ((join_continuous i).measurable.comp (by fun_prop))
  simpa only [copyMean, theta, Function.comp_def] using
    (copy_independent_of_real_data H c i).comp hf hg

/-- The complete adaptive input reward array has the same distribution after
substituting the independent arm-i copy, including its shared-latent dependence. -/
theorem streams_same_law (H : ℕ) (c : Precisions A) (m : Means A) (i : A) :
    (law H c i).map (mockStream H m i) = (law H c i).map (realStream H m i) := by
  change (law H c i).map ((realStream H m i) ∘ (swap H i)) = _
  rw [← Measure.map_map (realStream_measurable H m i)
    (swap_preserving H c i).measurable, (swap_preserving H c i).map_eq]

end
end Junpei.MockStream
