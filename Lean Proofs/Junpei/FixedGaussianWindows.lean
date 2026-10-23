import Junpei.FixedGaussianStackLaws
import Junpei.PolicySeeds

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory

namespace Junpei.FixedGaussian

noncomputable section

/-- Restricting an independent finite array to distinct coordinates preserves
the corresponding product law. This checks the initialization/seed windows. -/
theorem pi_restrict_preserving {I J : Type*} [Fintype I] [Fintype J]
    (ν : I → Measure ℝ) [∀ i, IsProbabilityMeasure (ν i)]
    (f : J → I) (hf : Function.Injective f) :
    MeasurePreserving (fun r : I → ℝ => fun j : J => r (f j))
      (Measure.pi ν) (Measure.pi (fun j => ν (f j))) := by
  have hi : iIndepFun (fun i : I => fun r : I → ℝ => r i) (Measure.pi ν) :=
    iIndepFun_pi (fun _ => measurable_id.aemeasurable)
  have hif := hi.precomp hf
  refine ⟨measurable_pi_lambda _ (fun j => measurable_pi_apply (f j)), ?_⟩
  rw [(iIndepFun_iff_map_fun_eq_pi_map
    (fun j : J => (measurable_pi_apply (f j)).aemeasurable)).mp hif]
  congr 1
  funext j
  exact (measurePreserving_eval ν (f j)).map_eq

theorem gaussian_prefix_castLE_preserving (N H : ℕ) (hHN : H ≤ N) (x : ℝ) :
    MeasurePreserving (fun r : Fin N → ℝ => fun j : Fin H => r (j.castLE hHN))
      (Measure.pi (fun _ : Fin N => gaussianReal x 1))
      (Measure.pi (fun _ : Fin H => gaussianReal x 1)) :=
  pi_restrict_preserving _ _ (fun _ _ h => Fin.ext (congrArg (fun z : Fin N => z.val) h))

open Junpei.FiniteGaussian Junpei.EmbeddingRaw Junpei.EmbeddingExecution

theorem fixed_prefix_preserving {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (N H : ℕ) (hHN : H ≤ N) (μ : Means A) :
    MeasurePreserving (fun r : RawStream A N => fun a (j : Fin H) => r a (j.castLE hHN))
      (fixedLaw N μ) (fixedLaw H μ) := by
  exact measurePreserving_pi
    (fun a : A => Measure.pi (fun _ : Fin N => gaussianReal (μ a) 1))
    (fun a : A => Measure.pi (fun _ : Fin H => gaussianReal (μ a) 1))
    (fun a => gaussian_prefix_castLE_preserving N H hHN (μ a))

theorem all_heads_split_preserving {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (H : ℕ) (μ : Means A) :
    MeasurePreserving (fun r : RawStream A (H + 1) =>
      (fun a => r a ⟨0, Nat.zero_lt_succ H⟩, tail r)) (fixedLaw (H + 1) μ)
      ((Measure.pi (fun a : A => gaussianReal (μ a) 1)).prod (fixedLaw H μ)) := by
  have h1 := measurePreserving_pi
    (fun b : A => Measure.pi (fun _ : Fin (H + 1) => gaussianReal (μ b) 1))
    (fun b : A => (gaussianReal (μ b) 1).prod
      (Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1)))
    (fun b => measurePreserving_piFinSuccAbove
      (fun _ : Fin (H + 1) => gaussianReal (μ b) 1) ⟨0, by omega⟩)
  have h2 := measurePreserving_arrowProdEquivProdArrow ℝ (Fin H → ℝ) A
    (fun b => gaussianReal (μ b) 1)
    (fun b => Measure.pi (fun _ : Fin H => gaussianReal (μ b) 1))
  convert h2.comp h1 using 1

end
end Junpei.FixedGaussian
