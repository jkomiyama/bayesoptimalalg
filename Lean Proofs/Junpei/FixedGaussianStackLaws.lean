import Junpei.FixedGaussianLaws
import Junpei.MockMeasure

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FixedGaussian

open FiniteGaussian EmbeddingRaw EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

def arraySplit {E : Type*} [MeasurableSpace E] (a : A) :
    (A → E) ≃ᵐ E × ({b : A // b ≠ a} → E) where
  toFun r := (r a, fun b => r b)
  invFun := MockMeasure.join a
  left_inv r := by ext b; by_cases hb : b = a <;> simp [MockMeasure.join, hb]
  right_inv p := by
    apply Prod.ext
    · simp [MockMeasure.join]
    · funext b
      simp [MockMeasure.join, b.property]
  measurable_toFun := (measurable_pi_apply a).prodMk
    (measurable_pi_lambda _ (fun b => measurable_pi_apply (b : A)))
  measurable_invFun := MockMeasure.join_measurable a

theorem arraySplit_preserving {E : Type*} [MeasurableSpace E]
    (ν : A → Measure E) [∀ a, IsProbabilityMeasure (ν a)] (a : A) :
    MeasurePreserving (arraySplit a) (Measure.pi ν)
      ((ν a).prod (Measure.pi (fun b : {b : A // b ≠ a} => ν b))) := by
  exact (MockMeasure.join_preserving ν a).symm (arraySplit a).symm

theorem gaussian_prefix_preserving (H : ℕ) (x : ℝ) :
    MeasurePreserving (fun r : Fin (H + 1) → ℝ => fun j : Fin H => r j.castSucc)
      (Measure.pi (fun _ : Fin (H + 1) => gaussianReal x 1))
      (Measure.pi (fun _ : Fin H => gaussianReal x 1)) := by
  have hp := (measurePreserving_snd (μ := gaussianReal x 1)
    (ν := Measure.pi (fun _ : Fin H => gaussianReal x 1))).comp
      (measurePreserving_piFinSuccAbove (fun _ : Fin (H + 1) => gaussianReal x 1)
        (Fin.last H))
  convert hp using 1
  funext r j
  simp [MeasurableEquiv.piFinSuccAbove, Fin.init]

def stackTail {H : ℕ} (a : A) (r : RawStream A (H + 1)) : RawStream A H :=
  fun b j => if b = a then r b j.succ else r b j.castSucc

theorem stackTail_measurable (H : ℕ) (a : A) :
    Measurable (stackTail (H := H) a) := by
  unfold stackTail
  apply measurable_pi_lambda
  intro b
  by_cases hb : b = a <;> simp only [hb, if_true, if_false] <;> fun_prop

/-- Taking the next reward from the selected arm and retaining every other
arm's unused prefix leaves the same independent per-arm Gaussian stack law. -/
theorem selected_stack_split_preserving (H : ℕ) (μ : Means A) (a : A) :
    MeasurePreserving (fun r : RawStream A (H + 1) =>
      (r a ⟨0, by omega⟩, stackTail a r)) (fixedLaw (H + 1) μ)
      ((gaussianReal (μ a) 1).prod (fixedLaw H μ)) := by
  have hsplit := arraySplit_preserving
    (fun b : A => Measure.pi (fun _ : Fin (H + 1) => gaussianReal (μ b) 1)) a
  have hhead := measurePreserving_piFinSuccAbove
    (fun _ : Fin (H + 1) => gaussianReal (μ a) 1) ⟨0, by omega⟩
  have hothers := measurePreserving_pi
    (fun b : {b : A // b ≠ a} =>
      Measure.pi (fun _ : Fin (H + 1) => gaussianReal (μ b) 1))
    (fun b : {b : A // b ≠ a} =>
      Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1))
    (fun b => gaussian_prefix_preserving H (μ b))
  have hpair := (hhead.prod hothers).comp hsplit
  have hassoc := measurePreserving_prodAssoc (gaussianReal (μ a) 1)
    (Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1))
    (Measure.pi (fun b : {b : A // b ≠ a} =>
      Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))
  letI : IsProbabilityMeasure
      ((Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1)).prod
        (Measure.pi (fun b : {b : A // b ≠ a} =>
          Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))) := by infer_instance
  letI : IsFiniteMeasure
      ((Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1)).prod
        (Measure.pi (fun b : {b : A // b ≠ a} =>
          Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))) :=
    IsZeroOrProbabilityMeasure.toIsFiniteMeasure _
  letI : SigmaFinite
      ((Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1)).prod
        (Measure.pi (fun b : {b : A // b ≠ a} =>
          Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))) :=
    IsFiniteMeasure.toSigmaFinite _
  have hj : MeasurePreserving (MockMeasure.join (E := Fin H → ℝ) a)
      ((Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1)).prod
        (Measure.pi (fun b : {b : A // b ≠ a} =>
          Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))) (fixedLaw H μ) :=
    MockMeasure.join_preserving
      (fun b : A => Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)) a
  have hjoin : MeasurePreserving
      (Prod.map (id : ℝ → ℝ) (MockMeasure.join (E := Fin H → ℝ) a))
      ((gaussianReal (μ a) 1).prod
        ((Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1)).prod
          (Measure.pi (fun b : {b : A // b ≠ a} =>
            Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))))
      ((gaussianReal (μ a) 1).prod (fixedLaw H μ)) := by
    exact (MeasurePreserving.id (gaussianReal (μ a) 1)).prod hj
  convert hjoin.comp (hassoc.comp hpair) using 1
  funext r
  apply Prod.ext
  · rfl
  · funext b j
    by_cases hb : b = a
    · subst b
      simp [stackTail, arraySplit, MockMeasure.join, MeasurableEquiv.prodAssoc, Fin.tail]
    · simp [stackTail, arraySplit, MockMeasure.join, hb, MeasurableEquiv.prodAssoc]

/-- Calendar-indexed sampling discards one coordinate from every arm. -/
theorem selected_calendar_split_preserving (H : ℕ) (μ : Means A) (a : A) :
    MeasurePreserving (fun r : RawStream A (H + 1) =>
      (r a ⟨0, by omega⟩, tail r)) (fixedLaw (H + 1) μ)
      ((gaussianReal (μ a) 1).prod (fixedLaw H μ)) := by
  have h1 := measurePreserving_pi
    (fun b : A => Measure.pi (fun _ : Fin (H + 1) => gaussianReal (μ b) 1))
    (fun b : A => (gaussianReal (μ b) 1).prod
      (Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))
    (fun b => measurePreserving_piFinSuccAbove
      (fun _ : Fin (H + 1) => gaussianReal (μ b) 1) ⟨0, by omega⟩)
  have h2 := measurePreserving_arrowProdEquivProdArrow ℝ (Fin H → ℝ) A
    (fun b => gaussianReal (μ b) 1)
    (fun b => Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1))
  have h3 := (measurePreserving_eval (fun b : A => gaussianReal (μ b) 1) a).prod
    (MeasurePreserving.id (fixedLaw H μ))
  convert h3.comp (h2.comp h1) using 1

end
end Junpei.FixedGaussian
