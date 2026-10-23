import Junpei.PolicySeeds

open MeasureTheory ProbabilityTheory

namespace Junpei.Policy

open Draft0.BayesianModel

noncomputable section

theorem recommendation_seed_joint_measurable (π : StatePolicy) (T : ℕ) :
    Measurable (fun x : Path × SeedArray (T + 1) =>
      recommendation π T x.1 (randomPath T x.2)) := by
  exact (recommendation_joint_measurable π T).comp
    (measurable_fst.prodMk ((randomPath_measurable T).comp measurable_snd))

theorem recommendation_execution_measurable {Ω : Type*} [MeasurableSpace Ω]
    (π : StatePolicy) (T : ℕ) (p : Ω → Path) (hp : Measurable p) :
    Measurable (fun x : Ω × SeedArray (T + 1) =>
      recommendation π T (p x.1) (randomPath T x.2)) := by
  exact (recommendation_seed_joint_measurable π T).comp
    ((hp.comp measurable_fst).prodMk measurable_snd)

/-- Independent fresh Gaussian randomizers preserve the entire recommendation
law when the state policy is replaced by its canonical pointwise-optimal version.
The reward stack may have any independent s-finite law. -/
theorem recommendation_repair_joint_ae {Ω : Type*} [MeasurableSpace Ω]
    (ν : Measure Ω) [SFinite ν] (π : StatePolicy) (T : ℕ) (hπ : AEOptimal π T)
    (p : Ω → Path) (hp : Measurable p) :
    (fun x : Ω × SeedArray (T + 1) => recommendation π T (p x.1) (randomPath T x.2))
      =ᵐ[ν.prod (randomizerLaw T)]
    (fun x : Ω × SeedArray (T + 1) => recommendation (repair π T) T (p x.1) (randomPath T x.2)) := by
  have hm : MeasurableSet {x : Ω × SeedArray (T + 1) |
      recommendation π T (p x.1) (randomPath T x.2) =
        recommendation (repair π T) T (p x.1) (randomPath T x.2)} :=
    measurableSet_eq_fun_of_countable (recommendation_execution_measurable π T p hp)
      (recommendation_execution_measurable (repair π T) T p hp)
  apply (Measure.ae_prod_iff_ae_ae hm).mpr
  exact ae_of_all _ (fun ω => recommendation_repair_ae π T hπ (p ω))

/-- Any function of the terminal recommendation, including true simple regret,
has exactly the same expectation under the original and repaired policies. -/
theorem recommendation_integral_repair_eq {Ω : Type*} [MeasurableSpace Ω]
    (ν : Measure Ω) [SFinite ν] (π : StatePolicy) (T : ℕ) (hπ : AEOptimal π T)
    (p : Ω → Path) (hp : Measurable p) (f : Arm → ℝ) :
    (∫ x : Ω × SeedArray (T + 1), f (recommendation π T (p x.1) (randomPath T x.2))
        ∂ν.prod (randomizerLaw T)) =
    ∫ x : Ω × SeedArray (T + 1), f (recommendation (repair π T) T (p x.1) (randomPath T x.2))
        ∂ν.prod (randomizerLaw T) := by
  apply integral_congr_ae
  filter_upwards [recommendation_repair_joint_ae ν π T hπ p hp] with x hx
  exact congrArg f hx

end

end Junpei.Policy
