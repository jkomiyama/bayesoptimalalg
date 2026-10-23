import Junpei.EmbeddingExecution

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FixedGaussian

noncomputable section

theorem fin_pi_absolutelyContinuous {B : Type*} [MeasurableSpace B] (n : ℕ)
    (μ ν : Fin n → Measure B) [∀ i, SigmaFinite (μ i)] [∀ i, SigmaFinite (ν i)]
    (h : ∀ i, μ i ≪ ν i) : Measure.pi μ ≪ Measure.pi ν := by
  induction n with
  | zero =>
      rw [Measure.pi_of_empty μ, Measure.pi_of_empty ν]
  | succ n ih =>
      let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => B)
        ⟨0, Nat.zero_lt_succ n⟩
      have hμ := measurePreserving_piFinSuccAbove μ ⟨0, Nat.zero_lt_succ n⟩
      have hν := measurePreserving_piFinSuccAbove ν ⟨0, Nat.zero_lt_succ n⟩
      have hrest := ih (fun j => μ ((⟨0, Nat.zero_lt_succ n⟩ : Fin (n + 1)).succAbove j))
        (fun j => ν ((⟨0, Nat.zero_lt_succ n⟩ : Fin (n + 1)).succAbove j))
        (fun j => h _)
      have hp := ((h ⟨0, Nat.zero_lt_succ n⟩).prod hrest).map e.symm.measurable
      simpa only [(hμ.symm e).map_eq, (hν.symm e).map_eq] using hp

theorem finite_pi_absolutelyContinuous {I B : Type*} [Fintype I] [MeasurableSpace B]
    (μ ν : I → Measure B) [∀ i, SigmaFinite (μ i)] [∀ i, SigmaFinite (ν i)]
    (h : ∀ i, μ i ≪ ν i) : Measure.pi μ ≪ Measure.pi ν := by
  let e := Fintype.equivFin I
  have hf := fin_pi_absolutelyContinuous (Fintype.card I)
    (fun j => μ (e.symm j)) (fun j => ν (e.symm j)) (fun j => h _)
  have hm := hf.map (MeasurableEquiv.piCongrLeft (fun _ : I => B) e.symm).measurable
  simpa only [Measure.pi_map_piCongrLeft] using hm

theorem gaussian_unit_equivalent (x y : ℝ) :
    gaussianReal x 1 ≪ gaussianReal y 1 ∧ gaussianReal y 1 ≪ gaussianReal x 1 := by
  constructor
  · exact (gaussianReal_absolutelyContinuous x one_ne_zero).trans
      (gaussianReal_absolutelyContinuous' y one_ne_zero)
  · exact (gaussianReal_absolutelyContinuous y one_ne_zero).trans
      (gaussianReal_absolutelyContinuous' x one_ne_zero)

end
end Junpei.FixedGaussian
