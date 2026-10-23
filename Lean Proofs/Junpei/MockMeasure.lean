import Junpei.MockStream

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockMeasure

noncomputable section

/-- Exchange the middle two independent factors without changing their laws. -/
theorem cross_preserving {X Y Z W : Type*}
    [MeasurableSpace X] [MeasurableSpace Y] [MeasurableSpace Z] [MeasurableSpace W]
    (μ : Measure X) (ν : Measure Y) (ρ : Measure Z) (τ : Measure W)
    [SFinite μ] [SFinite ν] [SFinite ρ] [SFinite τ] :
    MeasurePreserving (fun p : (X × Y) × (Z × W) => ((p.1.1, p.2.1), (p.1.2, p.2.2)))
      ((μ.prod ν).prod (ρ.prod τ)) ((μ.prod ρ).prod (ν.prod τ)) := by
  have h1 := measurePreserving_prodAssoc μ ν (ρ.prod τ)
  have h2 := (MeasurePreserving.id μ).prod
    (MeasurePreserving.symm MeasurableEquiv.prodAssoc (measurePreserving_prodAssoc ν ρ τ))
  have h3 := (MeasurePreserving.id μ).prod
    ((Measure.measurePreserving_swap (μ := ν) (ν := ρ)).prod (MeasurePreserving.id τ))
  have h4 := (MeasurePreserving.id μ).prod (measurePreserving_prodAssoc ρ ν τ)
  have h5 := MeasurePreserving.symm MeasurableEquiv.prodAssoc
    (measurePreserving_prodAssoc μ ρ (ν.prod τ))
  exact h5.comp (h4.comp (h3.comp (h2.comp h1)))

variable {A E : Type*} [Fintype A] [DecidableEq A] [MeasurableSpace E]

def join (i : A) (p : E × ({a : A // a ≠ i} → E)) : A → E :=
  fun a => if h : a = i then p.1 else p.2 ⟨a, h⟩

omit [Fintype A] in
theorem join_measurable (i : A) : Measurable (join (E := E) i) := by
  apply measurable_pi_lambda
  intro a
  by_cases h : a = i <;> simp only [join, h, dite_true, dite_false] <;> fun_prop

/-- Reassemble independent single-arm and retained-arm arrays into the full
product law. This applies equally to latent means and finite noise arrays. -/
theorem join_preserving (μ : A → Measure E) [∀ a, IsProbabilityMeasure (μ a)] (i : A) :
    MeasurePreserving (join i) ((μ i).prod (Measure.pi (fun a : {a : A // a ≠ i} => μ a)))
      (Measure.pi μ) := by
  refine ⟨join_measurable i, ?_⟩
  symm
  apply Measure.pi_eq
  intro s hs
  rw [Measure.map_apply (join_measurable i) (MeasurableSet.univ_pi hs)]
  have he : join i ⁻¹' Set.pi Set.univ s =
      s i ×ˢ Set.pi Set.univ (fun a : {a : A // a ≠ i} => s a) := by
    ext p
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_true_left,
      Set.mem_prod, join]
    constructor
    · intro h
      exact ⟨by simpa using h i, fun a => by simpa [a.property] using h a⟩
    · rintro ⟨hi, ho⟩ a
      split_ifs with ha
      · subst a; exact hi
      · exact ho ⟨a, ha⟩
  rw [he, Measure.prod_prod, Measure.pi_pi]
  letI : Unique {a : A // a = i} :=
    ⟨⟨i, rfl⟩, fun a => Subtype.ext a.property⟩
  have hprod := Fintype.prod_subtype_mul_prod_subtype (fun a : A => a = i)
    (fun a => μ a (s a))
  simpa only [Fintype.prod_unique] using hprod

end
end Junpei.MockMeasure
