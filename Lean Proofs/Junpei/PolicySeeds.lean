import Junpei.PolicyAE

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace Junpei.Policy

set_option maxHeartbeats 800000

open Draft0.BayesianModel

noncomputable section

abbrev SeedArray (n : ℕ) := Fin n → ℝ

def seedLaw (n : ℕ) : Measure (SeedArray n) :=
  Measure.pi (fun _ : Fin n => gaussianReal 0 1)

instance seedLaw_probability (n : ℕ) : IsProbabilityMeasure (seedLaw n) := by
  unfold seedLaw
  infer_instance

def randomizerLaw (T : ℕ) : Measure (SeedArray (T + 1)) := seedLaw (T + 1)

instance randomizerLaw_probability (T : ℕ) : IsProbabilityMeasure (randomizerLaw T) :=
  seedLaw_probability (T + 1)

def extendSeeds (n : ℕ) (u : SeedArray n) : RandomPath :=
  fun j => if h : j < n then u ⟨j, h⟩ else 0

def randomPath (T : ℕ) (u : SeedArray (T + 1)) : RandomPath := extendSeeds (T + 1) u

theorem extendSeeds_measurable (n : ℕ) : Measurable (extendSeeds n) := by
  apply measurable_pi_lambda
  intro j
  by_cases h : j < n
  · simpa only [extendSeeds, dif_pos h] using measurable_pi_apply (⟨j, h⟩ : Fin n)
  · simp only [extendSeeds, dif_neg h]
    exact measurable_const

theorem randomPath_measurable (T : ℕ) : Measurable (randomPath T) := extendSeeds_measurable (T + 1)

def splitSeeds (n : ℕ) : SeedArray (n + 1) ≃ᵐ SeedArray n × ℝ :=
  (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) (Fin.last n)).trans
    MeasurableEquiv.prodComm

theorem splitSeeds_apply (n : ℕ) (u : SeedArray (n + 1)) :
    splitSeeds n u = (Fin.init u, u (Fin.last n)) := by
  ext j <;> simp [splitSeeds, MeasurableEquiv.prodComm, Fin.init]

theorem splitSeeds_symm_apply (n : ℕ) (u : SeedArray n) (z : ℝ) :
    (splitSeeds n).symm (u, z) = Fin.snoc u z := by
  apply (splitSeeds n).injective
  rw [(splitSeeds n).apply_symm_apply, splitSeeds_apply, Fin.init_snoc, Fin.snoc_last]

theorem splitSeeds_preserving (n : ℕ) :
    MeasurePreserving (splitSeeds n) (seedLaw (n + 1)) ((seedLaw n).prod (gaussianReal 0 1)) := by
  exact Measure.measurePreserving_swap.comp
    (measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => gaussianReal 0 1) (Fin.last n))

/-- Counts after s rewards, with exactly the first s external seeds available.
The first three observations initialize the posterior and use no policy seed. -/
def prefixCounts (π : StatePolicy) (p : Path) : (s : ℕ) → SeedArray s → Counts
  | 0, _ => fun _ => 1
  | s + 1, u =>
    let c := prefixCounts π p s (Fin.init u)
    if 3 ≤ s then advance c (π.draw (s + 1) c (Draft0.PolicyPaths.meansAt p c) (u (Fin.last s)))
    else c

theorem prefixCounts_positive (π : StatePolicy) (p : Path) (s : ℕ) (u : SeedArray s) (i : Arm) :
    0 < prefixCounts π p s u i := by
  induction s with
  | zero => norm_num [prefixCounts]
  | succ s ih =>
    simp only [prefixCounts]
    split_ifs
    · simp only [advance]
      have h := ih (Fin.init u)
      omega
    · exact ih (Fin.init u)

theorem prefixCounts_measurable (π : StatePolicy) (p : Path) (s : ℕ) :
    Measurable (prefixCounts π p s) := by
  induction s with
  | zero => exact measurable_const
  | succ s ih =>
    have hinit : Measurable (fun u : SeedArray (s + 1) => Fin.init u) := by
      apply measurable_pi_lambda
      intro j
      exact measurable_pi_apply j.castSucc
    have hc := ih.comp hinit
    have hm := (measurable_of_countable (fun c : Counts => Draft0.PolicyPaths.meansAt p c)).comp hc
    have hu : Measurable (fun u : SeedArray (s + 1) => u (Fin.last s)) := measurable_pi_apply _
    have hd := (π.draw_measurable (s + 1)).comp (hc.prodMk (hm.prodMk hu))
    by_cases hs : 3 ≤ s
    · simp only [prefixCounts, if_pos hs]
      exact (measurable_of_countable (fun x : Counts × Arm => advance x.1 x.2)).comp
        (hc.prodMk hd)
    · simpa only [prefixCounts, if_neg hs] using hc

theorem prefixCounts_eq_counts (π : StatePolicy) (p : Path) (u : RandomPath) (s : ℕ) :
    prefixCounts π p s (fun j : Fin s => u j.val) = counts π p u s := by
  induction s with
  | zero => rfl
  | succ s ih =>
    have hi : Fin.init (fun j : Fin (s + 1) => u j.val) = fun j : Fin s => u j.val := rfl
    rw [prefixCounts, hi, ih]
    by_cases hs : 3 ≤ s
    · rw [if_pos hs, counts_update π p u s hs]
      rfl
    · rw [if_neg hs]
      have hs0 : s - 3 = 0 := by omega
      have hs1 : s + 1 - 3 = 0 := by omega
      simp only [counts, hs0, hs1]

theorem prefixCounts_repair_ae (π : StatePolicy) (T : ℕ) (hπ : AEOptimal π T)
    (p : Path) (s : ℕ) (hsT : s ≤ T) :
    prefixCounts π p s =ᵐ[seedLaw s] prefixCounts (repair π T) p s := by
  induction s with
  | zero => exact ae_of_all _ (fun _ => rfl)
  | succ s ih =>
    have hprev := ih (by omega : s ≤ T)
    have hm : MeasurableSet {x : SeedArray s × ℝ |
        prefixCounts π p (s + 1) ((splitSeeds s).symm x) =
          prefixCounts (repair π T) p (s + 1) ((splitSeeds s).symm x)} :=
      measurableSet_eq_fun_of_countable ((prefixCounts_measurable π p (s + 1)).comp (splitSeeds s).symm.measurable)
        ((prefixCounts_measurable (repair π T) p (s + 1)).comp (splitSeeds s).symm.measurable)
    have hp : ∀ᵐ x : SeedArray s × ℝ ∂(seedLaw s).prod (gaussianReal 0 1),
        prefixCounts π p (s + 1) ((splitSeeds s).symm x) =
          prefixCounts (repair π T) p (s + 1) ((splitSeeds s).symm x) := by
      apply (Measure.ae_prod_iff_ae_ae hm).mpr
      filter_upwards [hprev] with u hu
      by_cases hs : 3 ≤ s
      · have ha := repair_draw_ae π T hπ (s + 1) (prefixCounts π p s u)
          (Draft0.PolicyPaths.meansAt p (prefixCounts π p s u)) (by omega) (by omega)
          (prefixCounts_positive π p s u)
        filter_upwards [ha] with z hz
        rw [splitSeeds_symm_apply, prefixCounts, prefixCounts, Fin.init_snoc, Fin.snoc_last,
          if_pos hs, if_pos hs, ← hu]
        exact congrArg (advance (prefixCounts π p s u)) hz
      · exact ae_of_all _ (fun z => by
          rw [splitSeeds_symm_apply, prefixCounts, prefixCounts, Fin.init_snoc,
            if_neg hs, if_neg hs]
          exact hu)
    have h := (splitSeeds_preserving s).quasiMeasurePreserving.ae hp
    filter_upwards [h] with u hu
    simpa only [MeasurableEquiv.symm_apply_apply] using hu

def prefixRecommendation (π : StatePolicy) (T : ℕ) (p : Path) (u : SeedArray (T + 1)) : Arm :=
  let c := prefixCounts π p T (Fin.init u)
  π.recommend c (Draft0.PolicyPaths.meansAt p c) (u (Fin.last T))

theorem prefixRecommendation_measurable (π : StatePolicy) (T : ℕ) (p : Path) :
    Measurable (prefixRecommendation π T p) := by
  have hinit : Measurable (fun u : SeedArray (T + 1) => Fin.init u) := by
    apply measurable_pi_lambda
    intro j
    exact measurable_pi_apply j.castSucc
  have hc := (prefixCounts_measurable π p T).comp hinit
  have hm := (measurable_of_countable (fun c : Counts => Draft0.PolicyPaths.meansAt p c)).comp hc
  have hu : Measurable (fun u : SeedArray (T + 1) => u (Fin.last T)) := measurable_pi_apply _
  exact π.recommend_measurable.comp (hc.prodMk (hm.prodMk hu))

theorem prefixRecommendation_eq (π : StatePolicy) (T : ℕ) (p : Path) (u : SeedArray (T + 1)) :
    prefixRecommendation π T p u = recommendation π T p (randomPath T u) := by
  have hi : (fun j : Fin T => randomPath T u j.val) = Fin.init u := by
    funext j
    simp only [randomPath, extendSeeds, dif_pos (by omega : j.val < T + 1), Fin.init]
    rfl
  have hc := prefixCounts_eq_counts π p (randomPath T u) T
  rw [hi] at hc
  unfold prefixRecommendation recommendation
  rw [hc]
  simp only [randomPath, extendSeeds, dif_pos (by omega : T < T + 1)]
  rfl

/-- The original AE-optimal policy and its pointwise-optimal repair produce
the same recommendation almost surely under independent fresh Gaussian seeds. -/
theorem recommendation_repair_ae (π : StatePolicy) (T : ℕ) (hπ : AEOptimal π T) (p : Path) :
    (fun u : SeedArray (T + 1) => recommendation π T p (randomPath T u)) =ᵐ[randomizerLaw T]
      (fun u : SeedArray (T + 1) => recommendation (repair π T) T p (randomPath T u)) := by
  have hm : MeasurableSet {x : SeedArray T × ℝ |
      prefixRecommendation π T p ((splitSeeds T).symm x) =
        prefixRecommendation (repair π T) T p ((splitSeeds T).symm x)} :=
    measurableSet_eq_fun_of_countable ((prefixRecommendation_measurable π T p).comp (splitSeeds T).symm.measurable)
      ((prefixRecommendation_measurable (repair π T) T p).comp (splitSeeds T).symm.measurable)
  have hp : ∀ᵐ x : SeedArray T × ℝ ∂(seedLaw T).prod (gaussianReal 0 1),
      prefixRecommendation π T p ((splitSeeds T).symm x) =
        prefixRecommendation (repair π T) T p ((splitSeeds T).symm x) := by
    apply (Measure.ae_prod_iff_ae_ae hm).mpr
    filter_upwards [prefixCounts_repair_ae π T hπ p T le_rfl] with u hu
    have ha := repair_recommend_ae π T hπ (prefixCounts π p T u)
      (Draft0.PolicyPaths.meansAt p (prefixCounts π p T u)) (prefixCounts_positive π p T u)
    filter_upwards [ha] with z hz
    rw [splitSeeds_symm_apply, prefixRecommendation, prefixRecommendation,
      Fin.init_snoc, Fin.snoc_last, ← hu]
    exact hz
  have h := (splitSeeds_preserving T).quasiMeasurePreserving.ae hp
  filter_upwards [h] with u hu
  simp only [MeasurableEquiv.symm_apply_apply] at hu
  simpa only [prefixRecommendation_eq] using hu

end

end Junpei.Policy
