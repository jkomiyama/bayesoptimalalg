import Junpei.EmbeddingExecution

open MeasureTheory ProbabilityTheory

namespace Junpei.EmbeddingMeasurable

open Junpei.FiniteGaussian Junpei.FiniteModelMeasurable Junpei.EmbeddingRaw
open Junpei.EmbeddingDirect Junpei.EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

@[fun_prop] theorem shift_param_measurable {B X : Type*} [DecidableEq B]
    [MeasurableSpace X] (m : X → Means B) (z : X → ℝ) (i : B)
    (hm : Measurable m) (hz : Measurable z) :
    Measurable (fun x => shift (m x) i (z x)) := by
  apply measurable_pi_lambda
  intro a
  by_cases h : a = i
  · simpa only [shift, if_pos h] using ((measurable_pi_apply a).comp hm).add hz
  · simpa only [shift, if_neg h, add_zero] using (measurable_pi_apply a).comp hm

variable (i : A) [Nonempty (Retained i)] (dummy : Retained i)

abbrev Input (H : ℕ) := (Means A × Means (Retained i)) ×
  ((Fin H → ℝ) × RawStream (Retained i) H)

def result (H : ℕ) (c : Precisions A) (d : Precisions (Retained i))
    (p : Input i H) : Retained i :=
  run (simulate i dummy (rawOfInnovation (optimalFirst H c p.1.1) c p.1.1)
    d p.1.2 p.2.1) p.2.2

theorem result_zero_measurable (c : Precisions A) (d : Precisions (Retained i)) :
    Measurable (result i dummy 0 c d) := by
  have hm : Measurable (fun p : Input i 0 => firstMax p.1.1) :=
    firstMax_measurable.comp (measurable_fst.comp measurable_fst)
  have hn : Measurable (fun p : Input i 0 => (firstMax (A := Retained i) p.1.2 : A)) :=
    measurable_subtype_coe.comp
      (firstMax_measurable.comp (measurable_snd.comp measurable_fst))
  have h : Measurable (fun p : Input i 0 =>
      if firstMax p.1.1 = i then (firstMax (A := Retained i) p.1.2 : A) else firstMax p.1.1) :=
    hn.ite (hm (measurableSet_singleton i)) hm
  have hr : Measurable (fun p : Input i 0 => (result i dummy 0 c d p : A)) := by
    convert h using 1
    funext p
    unfold result optimalFirst rawOfInnovation simulate run
    split_ifs <;> rfl
  exact hr.subtype_mk

theorem result_measurable (H : ℕ) (c : Precisions A) (d : Precisions (Retained i)) :
    Measurable (result i dummy H c d) := by
  induction H generalizing c d with
  | zero => exact result_zero_measurable i dummy c d
  | succ H ih =>
    let g : Input i (H+1) × A → Retained i := fun x =>
      let p := x.1
      let a := x.2
      if h : a = i then
        result i dummy H (advance c a) d
          ((shift p.1.1 a ((p.2.1 ⟨0, by omega⟩-p.1.1 a)/((c a : ℝ)+1)),p.1.2),
            ((fun t => p.2.1 t.succ),tail p.2.2))
      else
        let b : Retained i := ⟨a,h⟩
        let y := p.2.2 b ⟨0, by omega⟩
        result i dummy H (advance c a) (advance d b)
          ((shift p.1.1 a ((y-p.1.1 a)/((c a : ℝ)+1)),
            shift p.1.2 b ((y-p.1.2 b)/((d b : ℝ)+1))),
            ((fun t => p.2.1 t.succ),tail p.2.2))
    have hg : Measurable g := by
      apply measurable_from_prod_countable_left
      intro a
      by_cases h : a = i
      · simp only [g, dif_pos h]
        apply (ih (advance c a) d).comp
        unfold tail
        fun_prop
      · simp only [g, dif_neg h]
        have hy : Measurable (fun p : Input i (H+1) => p.2.2 ⟨a,h⟩ ⟨0,by omega⟩) :=
          (measurable_pi_apply (⟨0,by omega⟩ : Fin (H+1))).comp
            ((measurable_pi_apply (⟨a,h⟩ : Retained i)).comp
              (measurable_snd.comp measurable_snd))
        apply (ih (advance c a) (advance d ⟨a,h⟩)).comp
        have hm : Measurable (fun p : Input i (H+1) => p.1.1) :=
          measurable_fst.comp measurable_fst
        have hn : Measurable (fun p : Input i (H+1) => p.1.2) :=
          measurable_snd.comp measurable_fst
        have hz := (hy.sub ((measurable_pi_apply a).comp hm)).div_const ((c a : ℝ)+1)
        have hw := (hy.sub ((measurable_pi_apply (⟨a,h⟩ : Retained i)).comp hn)).div_const
          ((d ⟨a,h⟩ : ℝ)+1)
        apply Measurable.prodMk
        · exact (shift_param_measurable _ _ a hm hz).prodMk
            (shift_param_measurable _ _ (⟨a,h⟩ : Retained i) hn hw)
        · unfold tail
          fun_prop
    have hchoose : Measurable (fun p : Input i (H+1) => chooseFirst H c p.1.1) :=
      (chooseFirst_measurable H c).comp (measurable_fst.comp measurable_fst)
    have h := hg.comp (measurable_id.prodMk hchoose)
    convert h using 1
    funext p
    simp only [result, optimalFirst, rawOfInnovation, simulate, g, Function.comp_def, id]
    split_ifs <;> rfl

theorem simulated_optimal_regular (H : ℕ) (c : Precisions A) (m : Means A)
    (d : Precisions (Retained i)) (n : Means (Retained i)) (imaginary : Fin H → ℝ) :
    Regular (simulate i dummy (rawOfInnovation (optimalFirst H c m) c m) d n imaginary) := by
  induction H generalizing c m d n with
  | zero => simp only [optimalFirst, rawOfInnovation, simulate]; split_ifs <;> trivial
  | succ H ih =>
    simp only [optimalFirst, rawOfInnovation, simulate]
    split_ifs with h
    · constructor
      · intro y; exact ih _ _ _ _ _
      · change Measurable (result i dummy H (advance c (chooseFirst H c m)) d ∘
          fun p : ℝ × RawStream (Retained i) H =>
          ((shift m (chooseFirst H c m)
            ((imaginary ⟨0,by omega⟩-m (chooseFirst H c m))/
              ((c (chooseFirst H c m) : ℝ)+1)),n),
            ((fun t => imaginary t.succ),p.2)))
        apply (result_measurable i dummy H (advance c (chooseFirst H c m)) d).comp
        fun_prop
    · constructor
      · intro y; exact ih _ _ _ _ _
      · let b : Retained i := ⟨chooseFirst H c m,h⟩
        change Measurable (result i dummy H (advance c b) (advance d b) ∘
          fun p : ℝ × RawStream (Retained i) H =>
          ((shift m b ((p.1-m b)/((c b : ℝ)+1)),
            shift n b ((p.1-n b)/((d b : ℝ)+1))),
            ((fun t => imaginary t.succ),p.2)))
        apply (result_measurable i dummy H (advance c b) (advance d b)).comp
        fun_prop

end
end Junpei.EmbeddingMeasurable
