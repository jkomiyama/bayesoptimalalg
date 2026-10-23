import Junpei.FinitePosterior
import Mathlib.MeasureTheory.Measure.CharacteristicFunction

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.ObservationBridge

noncomputable section

theorem charFunDual_centered_gaussian (v : ℝ≥0) (L : ℝ →L[ℝ] ℝ) :
    charFunDual (gaussianReal 0 v) L =
      Complex.exp (-((v : ℝ) * (L 1)^2 : ℝ) / 2) := by
  rw [charFunDual_eq_charFun_map_one, gaussianReal_map_continuousLinearMap,
    charFun_gaussianReal]
  simp only [map_zero, Complex.ofReal_zero, zero_mul, mul_zero, zero_sub,
    NNReal.coe_mul, Real.coe_toNNReal', max_eq_left (sq_nonneg _)]
  congr 1
  push_cast
  ring

def updateLinear (c : ℝ≥0) : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ) :=
  (((((c : ℝ)+1)⁻¹) • (ContinuousLinearMap.fst ℝ ℝ ℝ +
      ContinuousLinearMap.snd ℝ ℝ ℝ)).prod
    (((c : ℝ) / ((c : ℝ)+1)) • ContinuousLinearMap.fst ℝ ℝ ℝ -
      (((c : ℝ)+1)⁻¹) • ContinuousLinearMap.snd ℝ ℝ ℝ))

theorem updateLinear_apply (c : ℝ≥0) (x : ℝ × ℝ) :
    updateLinear c x = ((x.1+x.2)/((c : ℝ)+1),
      (c : ℝ)*x.1/((c : ℝ)+1)-x.2/((c : ℝ)+1)) := by
  ext <;> simp [updateLinear, div_eq_mul_inv] <;> ring

theorem update_joint_law (c : ℝ≥0) (hc : 0 < c) :
    (((gaussianReal 0 c⁻¹).prod (gaussianReal 0 1)).map (updateLinear c)) =
      (gaussianReal 0 (Draft0.Gaussian.updateVariance c)).prod
        (gaussianReal 0 (c+1)⁻¹) := by
  apply Measure.ext_of_charFunDual
  funext L
  rw [charFunDual_map, charFunDual_prod, charFunDual_prod]
  simp_rw [charFunDual_centered_gaussian]
  rw [← Complex.exp_add, ← Complex.exp_add]
  congr 1
  have hc0 : (c : ℝ) ≠ 0 := ne_of_gt (by exact_mod_cast hc)
  have hc1 : (c : ℝ)+1 ≠ 0 := by positivity
  have hleft : (L.comp (updateLinear c)).comp (ContinuousLinearMap.inl ℝ ℝ ℝ) 1 =
      ((c : ℝ)+1)⁻¹ * L (1,0) + (c : ℝ)/((c : ℝ)+1) * L (0,1) := by
    change L (updateLinear c (1,0)) = _
    rw [updateLinear_apply]
    have hx : ((1+0)/((c : ℝ)+1), (c : ℝ)*1/((c : ℝ)+1)-0/((c : ℝ)+1)) =
        ((c : ℝ)+1)⁻¹ • (1,0) + ((c : ℝ)/((c : ℝ)+1)) • (0,1) := by
      ext <;> simp
    rw [hx, map_add, map_smul, map_smul]
    rfl
  have hright : (L.comp (updateLinear c)).comp (ContinuousLinearMap.inr ℝ ℝ ℝ) 1 =
      ((c : ℝ)+1)⁻¹ * L (1,0) - ((c : ℝ)+1)⁻¹ * L (0,1) := by
    change L (updateLinear c (0,1)) = _
    rw [updateLinear_apply]
    have hx : ((0+1)/((c : ℝ)+1), (c : ℝ)*0/((c : ℝ)+1)-1/((c : ℝ)+1)) =
        ((c : ℝ)+1)⁻¹ • (1,0) - ((c : ℝ)+1)⁻¹ • (0,1) := by
      ext <;> simp
    rw [hx, map_sub, map_smul, map_smul]
    rfl
  rw [hleft, hright]
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.inl_apply,
    ContinuousLinearMap.inr_apply, NNReal.coe_inv, NNReal.coe_one,
    Draft0.Gaussian.updateVariance, NNReal.coe_mul, NNReal.coe_add]
  have he (a b : ℝ) :
      -((c : ℝ)⁻¹ * (((c : ℝ)+1)⁻¹*a + (c : ℝ)/((c : ℝ)+1)*b)^2)/2 +
      -((((c : ℝ)+1)⁻¹*a - ((c : ℝ)+1)⁻¹*b)^2)/2 =
      -(((c : ℝ)*((c : ℝ)+1))⁻¹*a^2)/2 + -(((c : ℝ)+1)⁻¹*b^2)/2 := by
    field_simp [hc0, hc1]
    ring
  have hz := congrArg (fun x : ℝ => (x : ℂ)) (he (L (1,0)) (L (0,1)))
  push_cast at hz
  push_cast
  simpa only [one_mul] using hz

def rawPair (m : ℝ) (x : ℝ × ℝ) : ℝ × ℝ :=
  (m+x.1, m+x.1+x.2)

def posteriorPair (c : ℝ≥0) (m : ℝ) (x : ℝ × ℝ) : ℝ × ℝ :=
  (m+x.1+x.2, m+((c : ℝ)+1)*x.1)

@[fun_prop] theorem rawPair_measurable (m : ℝ) : Measurable (rawPair m) := by
  unfold rawPair
  fun_prop

@[fun_prop] theorem posteriorPair_measurable (c : ℝ≥0) (m : ℝ) :
    Measurable (posteriorPair c m) := by
  unfold posteriorPair
  fun_prop

theorem rawPair_eq_posteriorPair_update (c : ℝ≥0) (m : ℝ) (x : ℝ × ℝ) :
    rawPair m x = posteriorPair c m (updateLinear c x) := by
  have hc1 : (c : ℝ)+1 ≠ 0 := by positivity
  rw [updateLinear_apply]
  ext <;> simp only [rawPair, posteriorPair] <;> field_simp [hc1] <;> ring

/-- The actual joint law of a Gaussian latent mean and its noisy observation
equals the posterior-innovation/posterior-residual representation. -/
theorem scalar_observation_joint_law (c : ℝ≥0) (hc : 0 < c) (m : ℝ) :
    (((gaussianReal 0 c⁻¹).prod (gaussianReal 0 1)).map (rawPair m)) =
      (((gaussianReal 0 (Draft0.Gaussian.updateVariance c)).prod
        (gaussianReal 0 (c+1)⁻¹)).map (posteriorPair c m)) := by
  rw [← update_joint_law c hc, Measure.map_map (by fun_prop) (by fun_prop)]
  congr 1
  funext x
  exact rawPair_eq_posteriorPair_update c m x

theorem scalar_observation_integral (c : ℝ≥0) (hc : 0 < c) (m : ℝ)
    (f : ℝ × ℝ → ℝ) (hf : Measurable f) :
    (∫ x : ℝ × ℝ, f (rawPair m x)
      ∂(gaussianReal 0 c⁻¹).prod (gaussianReal 0 1)) =
    ∫ x : ℝ × ℝ, f (posteriorPair c m x)
      ∂(gaussianReal 0 (Draft0.Gaussian.updateVariance c)).prod
        (gaussianReal 0 (c+1)⁻¹) := by
  rw [← integral_map (by fun_prop) hf.aestronglyMeasurable,
    scalar_observation_joint_law c hc m,
    integral_map (by fun_prop) hf.aestronglyMeasurable]

theorem prod_preserving {X Y U V : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    [MeasurableSpace U] [MeasurableSpace V] {μ : Measure X} {ν : Measure Y}
    {ρ : Measure U} {σ : Measure V} [SFinite μ] [SFinite ρ]
    {f : X → Y} {g : U → V} (hf : MeasurePreserving f μ ν)
    (hg : MeasurePreserving g ρ σ) :
    MeasurePreserving (Prod.map f g) (μ.prod ρ) (ν.prod σ) := by
  refine ⟨hf.measurable.prodMap hg.measurable, ?_⟩
  rw [← Measure.map_prod_map μ ρ hf.measurable hg.measurable, hf.map_eq, hg.map_eq]

theorem middle_swap_preserving {X Y U : Type*} [MeasurableSpace X]
    [MeasurableSpace Y] [MeasurableSpace U] (μ : Measure X) (ν : Measure Y)
    (ρ : Measure U) [SFinite μ] [SFinite ν] [SFinite ρ] :
    MeasurePreserving (fun x : (X × Y) × U => ((x.1.1,x.2),x.1.2))
      ((μ.prod ν).prod ρ) ((μ.prod ρ).prod ν) := by
  have h1 := measurePreserving_prodAssoc μ ν ρ
  have h2 := prod_preserving (MeasurePreserving.id μ)
    (Measure.measurePreserving_swap (μ := ν) (ν := ρ))
  have h3 := MeasurePreserving.symm MeasurableEquiv.prodAssoc
    (measurePreserving_prodAssoc μ ρ ν)
  exact h3.comp (h2.comp h1)

section Finite

open Junpei.FiniteGaussian Junpei.FinitePosterior

variable {A : Type*} [Fintype A] [DecidableEq A]

def translatedJoin (m : Means A) (i : A) (x : ℝ × ({a : A // a ≠ i} → ℝ)) :
    Means A := m + join i x

@[fun_prop] theorem translatedJoin_measurable (m : Means A) (i : A) :
    Measurable (translatedJoin m i) := by
  exact (continuous_const.add (join_continuous i)).measurable

theorem translatedJoin_preserving (c : Precisions A) (m : Means A) (i : A) :
    MeasurePreserving (translatedJoin m i)
      ((gaussianReal 0 (c i)⁻¹).prod (otherLaw c i)) (posteriorLaw c m) := by
  refine ⟨translatedJoin_measurable m i, ?_⟩
  unfold posteriorLaw
  rw [← join_law c i, Measure.map_map (by fun_prop) (join_continuous i).measurable]
  rfl

def rawObservation (i : A) (x : Means A × ℝ) : Means A × ℝ :=
  (x.1, x.1 i+x.2)

def reconstructedObservation (c : Precisions A) (m : Means A) (i : A)
    (x : ℝ × Means A) : Means A × ℝ :=
  (shift m i x.1+x.2, m i+((c i : ℝ)+1)*x.1)

omit [Fintype A] [DecidableEq A] in
@[fun_prop] theorem rawObservation_measurable (i : A) : Measurable (rawObservation i) := by
  unfold rawObservation
  fun_prop

omit [Fintype A] in
@[fun_prop] theorem reconstructedObservation_measurable
    (c : Precisions A) (m : Means A) (i : A) :
    Measurable (reconstructedObservation c m i) := by
  unfold reconstructedObservation
  apply Measurable.prodMk
  · apply measurable_pi_lambda
    intro a
    simp only [shift, Pi.add_apply]
    split_ifs <;> fun_prop
  · fun_prop

/-- Full finite-arm Gaussian observation/posterior disintegration as an
equality of joint laws. It applies to arbitrary measurable adaptive payoffs. -/
theorem observation_joint_law (c : Precisions A) (m : Means A) (i : A)
    (hc : 0 < c i) :
    ((posteriorLaw c m).prod (gaussianReal 0 1)).map (rawObservation i) =
      ((gaussianReal 0 (innovation c i)).prod (centeredLaw (advance c i))).map
        (reconstructedObservation c m i) := by
  let μ := gaussianReal 0 (c i)⁻¹
  let ν := gaussianReal (0 : ℝ) (1 : ℝ≥0)
  let ρ := otherLaw c i
  let ζ := gaussianReal 0 (innovation c i)
  let η := gaussianReal 0 (c i+1)⁻¹
  have hraw := (prod_preserving (translatedJoin_preserving c m i)
    (MeasurePreserving.id ν)).comp (middle_swap_preserving μ ν ρ)
  have hupdate : MeasurePreserving (updateLinear (c i)) (μ.prod ν) (ζ.prod η) :=
    ⟨by fun_prop, update_joint_law (c i) hc⟩
  have hcenter : MeasurePreserving (join i) (η.prod ρ) (centeredLaw (advance c i)) := by
    refine ⟨(join_continuous i).measurable, ?_⟩
    have hj := join_law (advance c i) i
    rw [otherLaw_advance] at hj
    simpa only [advance, if_pos rfl, add_zero, add_comm] using hj
  have hpost := (prod_preserving (MeasurePreserving.id ζ) hcenter).comp
    ((measurePreserving_prodAssoc ζ η ρ).comp
      (prod_preserving hupdate (MeasurePreserving.id ρ)))
  have hr := congrArg (fun l => l.map (rawObservation i)) hraw.map_eq
  dsimp only at hr
  rw [Measure.map_map (rawObservation_measurable i) hraw.measurable] at hr
  have hp := congrArg (fun l => l.map (reconstructedObservation c m i)) hpost.map_eq
  dsimp only at hp
  rw [Measure.map_map (reconstructedObservation_measurable c m i) hpost.measurable] at hp
  rw [← hr, ← hp]
  congr 1
  funext x
  dsimp only [Function.comp_def, Prod.map, id, MeasurableEquiv.prodAssoc,
    Equiv.prodAssoc, translatedJoin, rawObservation, reconstructedObservation]
  rw [updateLinear_apply]
  have hc1 : (c i : ℝ)+1 ≠ 0 := by positivity
  apply Prod.ext
  · ext a
    by_cases ha : a = i
    · subst a
      simp only [Pi.add_apply, join, shift, dite_true, ite_true]
      field_simp [hc1]
      ring
    · simp [Pi.add_apply, join, shift, ha]
  · simp only [Pi.add_apply, join, dite_true]
    field_simp [hc1]
    ring

theorem observation_integral (c : Precisions A) (m : Means A) (i : A)
    (hc : 0 < c i) (f : Means A × ℝ → ℝ) (hf : Measurable f) :
    (∫ x : Means A × ℝ, f (rawObservation i x)
      ∂(posteriorLaw c m).prod (gaussianReal 0 1)) =
    ∫ x : ℝ × Means A, f (reconstructedObservation c m i x)
      ∂(gaussianReal 0 (innovation c i)).prod (centeredLaw (advance c i)) := by
  rw [← integral_map (rawObservation_measurable i).aemeasurable hf.aestronglyMeasurable,
    observation_joint_law c m i hc,
    integral_map (reconstructedObservation_measurable c m i).aemeasurable
      hf.aestronglyMeasurable]

end Finite

end
end Junpei.ObservationBridge
