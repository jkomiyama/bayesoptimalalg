import Junpei.EmbeddingRaw
import Junpei.MockStream
import Junpei.FiniteModelMeasurable
import Junpei.ObservationBridge

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.EmbeddingExecution

open FiniteGaussian FinitePosterior EmbeddingRaw FiniteModelMeasurable

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

def arrayLaw (H : ℕ) : Measure (RawStream A H) :=
  Measure.pi (fun _ : A => MockStream.noiseLaw H)

instance arrayLaw_probability (H : ℕ) : IsProbabilityMeasure (arrayLaw (A := A) H) := by
  unfold arrayLaw
  infer_instance

def splitNoise {H : ℕ} (e : RawStream A (H + 1)) : (A → ℝ) × RawStream A H :=
  (fun a => e a ⟨0, Nat.zero_lt_succ H⟩, tail e)

theorem splitNoise_preserving (H : ℕ) :
    MeasurePreserving (splitNoise (A := A)) (arrayLaw (A := A) (H + 1))
      ((Measure.pi (fun _ : A => gaussianReal 0 1)).prod (arrayLaw (A := A) H)) := by
  have h1 := measurePreserving_pi
    (fun _ : A => MockStream.noiseLaw (H + 1))
    (fun _ : A => (gaussianReal 0 1).prod (MockStream.noiseLaw H))
    (fun _ => measurePreserving_piFinSuccAbove
      (fun _ : Fin (H + 1) => gaussianReal 0 1) ⟨0, Nat.zero_lt_succ H⟩)
  have h2 := measurePreserving_arrowProdEquivProdArrow ℝ (Fin H → ℝ) A
    (fun _ => gaussianReal 0 1) (fun _ => MockStream.noiseLaw H)
  convert h2.comp h1 using 1

theorem selected_noise_split_preserving (H : ℕ) (a : A) :
    MeasurePreserving (fun e : RawStream A (H + 1) =>
      (e a ⟨0, Nat.zero_lt_succ H⟩, tail e)) (arrayLaw (A := A) (H + 1))
      ((gaussianReal 0 1).prod (arrayLaw (A := A) H)) := by
  exact ((measurePreserving_eval (fun _ : A => gaussianReal 0 1) a).prod
    (MeasurePreserving.id (arrayLaw (A := A) H))).comp (splitNoise_preserving H)

def worldLaw (H : ℕ) (c : Precisions A) (m : Means A) :
    Measure (Means A × RawStream A H) := (posteriorLaw c m).prod (arrayLaw H)

instance worldLaw_probability (H : ℕ) (c : Precisions A) (m : Means A) :
    IsProbabilityMeasure (worldLaw H c m) := by unfold worldLaw; infer_instance

def observations {H : ℕ} (w : Means A × RawStream A H) : RawStream A H :=
  fun a t => w.1 a + w.2 a t

theorem observations_measurable (H : ℕ) : Measurable (observations (A := A) (H := H)) := by
  unfold observations
  fun_prop

section Measurable
variable [MeasurableSpace A] [MeasurableSingletonClass A]

def Regular : {H : ℕ} → Strategy (A := A) H → Prop
  | 0, .stop _ => True
  | _ + 1, .sample _ next =>
      (∀ y, Regular (next y)) ∧
      Measurable (fun p : ℝ × RawStream A _ => run (next p.1) p.2)

theorem regular_run_measurable {H : ℕ} (s : Strategy (A := A) H) (hs : Regular s) :
    Measurable (run s) := by
  cases s with
  | stop a => exact measurable_const
  | @sample H a next =>
    have hm : Measurable (fun r : RawStream A (H + 1) =>
        (r a ⟨0, Nat.zero_lt_succ H⟩, tail r)) := by unfold tail; fun_prop
    exact hs.2.comp hm

def rawRisk {H : ℕ} (s : Strategy (A := A) H) (c : Precisions A) (m : Means A) : ℝ :=
  ∫ w, maxAll w.1 - w.1 (run s (observations w)) ∂worldLaw H c m

theorem rawRisk_integrable {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hs : Measurable (run s)) :
    Integrable (fun w : Means A × RawStream A H => maxAll w.1 - w.1 (run s (observations w)))
      (worldLaw H c m) := by
  have hc (a : A) : Integrable (fun w : Means A × RawStream A H => w.1 a)
      (worldLaw H c m) :=
    (posterior_coordinate_integrable c m a).comp_fst (arrayLaw H)
  have hsum : Integrable (fun w : Means A × RawStream A H => ∑ a, |w.1 a|)
      (worldLaw H c m) := by
    apply integrable_finset_sum
    intro a _
    simpa only [Real.norm_eq_abs] using (hc a).norm
  have hcoord : Measurable (fun p : (Means A × RawStream A H) × A => p.1.1 p.2) :=
    measurable_from_prod_countable_left (fun a => by
      change Measurable (fun w : Means A × RawStream A H => w.1 a)
      exact (measurable_pi_apply a).comp measurable_fst)
  have hsel : Measurable (fun w : Means A × RawStream A H => w.1 (run s (observations w))) :=
    hcoord.comp (measurable_id.prodMk (hs.comp (observations_measurable H)))
  have hi : Integrable (fun w : Means A × RawStream A H => w.1 (run s (observations w)))
      (worldLaw H c m) := by
    apply hsum.mono' hsel.aestronglyMeasurable
    apply Filter.Eventually.of_forall
    intro w
    rw [Real.norm_eq_abs]
    exact Finset.single_le_sum (fun a _ => abs_nonneg (w.1 a)) (Finset.mem_univ _)
  exact ((posterior_integrable maxAll_lipschitz c m).comp_fst (arrayLaw H)).sub hi

theorem rawRisk_zero (a : A) (c : Precisions A) (m : Means A) :
    rawRisk (.stop a) c m = terminalRegret c m a := by
  unfold rawRisk worldLaw
  simp only [run]
  simpa only [measureReal_univ_eq_one, one_smul, terminalRegret] using
    integral_fun_fst (μ := posteriorLaw c m) (ν := arrayLaw (A := A) 0)
    (fun u : Means A => maxAll u - u a)

theorem optimalFirst_raw_run_measurable (H : ℕ) (c : Precisions A) :
    Measurable (fun p : Means A × RawStream A H =>
      run (rawOfInnovation (optimalFirst H c p.1) c p.1) p.2) := by
  induction H generalizing c with
  | zero => exact firstMax_measurable.comp measurable_fst
  | succ H ih =>
    let g : (Means A × RawStream A (H + 1)) × A → A := fun p =>
      run (rawOfInnovation (optimalFirst H (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2 p.2 ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1)))) (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2 p.2 ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1)))) (tail p.1.2)
    have hg : Measurable g := by
      apply measurable_from_prod_countable_left
      intro a
      have ht : Measurable (fun p : Means A × RawStream A (H + 1) =>
          (shift p.1 a ((p.2 a ⟨0, Nat.zero_lt_succ H⟩ - p.1 a) / ((c a : ℝ) + 1)),
            tail p.2)) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro b
          simp only [shift]
          split_ifs <;> fun_prop
        · unfold tail; fun_prop
      exact (ih (advance c a)).comp ht
    have hm := hg.comp (measurable_id.prodMk ((chooseFirst_measurable H c).comp measurable_fst))
    exact hm

theorem optimalFirst_raw_regular (H : ℕ) (c : Precisions A) (m : Means A) :
    Regular (rawOfInnovation (optimalFirst H c m) c m) := by
  induction H generalizing c m with
  | zero => trivial
  | succ H ih =>
    constructor
    · intro y; exact ih _ _
    · have ht : Measurable (fun p : ℝ × RawStream A H =>
          (shift m (chooseFirst H c m)
            ((p.1 - m (chooseFirst H c m)) / ((c (chooseFirst H c m) : ℝ) + 1)), p.2)) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro a
          simp only [shift]
          split_ifs <;> fun_prop
        · exact measurable_snd
      exact (optimalFirst_raw_run_measurable H (advance c (chooseFirst H c m))).comp ht

def continuationPayoff {H : ℕ} (next : ℝ → Strategy (A := A) H)
    (p : (Means A × ℝ) × RawStream A H) : ℝ :=
  maxAll p.1.1 - p.1.1 (run (next p.1.2) (observations (p.1.1, p.2)))

theorem continuationPayoff_measurable {H : ℕ} (next : ℝ → Strategy (A := A) H)
    (hnext : Measurable (fun p : ℝ × RawStream A H => run (next p.1) p.2)) :
    Measurable (continuationPayoff next) := by
  have hselect : Measurable (fun p : (Means A × ℝ) × RawStream A H =>
      run (next p.1.2) (observations (p.1.1, p.2))) := by
    have ht : Measurable (fun p : (Means A × ℝ) × RawStream A H =>
        (p.1.2, observations (p.1.1, p.2))) := by unfold observations; fun_prop
    exact hnext.comp ht
  have hcoord : Measurable (fun p : ((Means A × ℝ) × RawStream A H) × A =>
      p.1.1.1 p.2) := measurable_from_prod_countable_left (fun a => by
        change Measurable (fun p : (Means A × ℝ) × RawStream A H => p.1.1 a)
        fun_prop)
  exact ((maxAll_lipschitz (A := A)).continuous.measurable.comp
    (measurable_fst.comp measurable_fst)).sub
      (hcoord.comp (measurable_id.prodMk hselect))

theorem continuationPayoff_norm_le {H : ℕ} (next : ℝ → Strategy (A := A) H)
    (p : (Means A × ℝ) × RawStream A H) :
    ‖continuationPayoff next p‖ ≤ 2 * ∑ a, |p.1.1 a| := by
  obtain ⟨a, ha⟩ := maxAll_attained p.1.1
  have h1 := Finset.single_le_sum (fun a _ => abs_nonneg (p.1.1 a)) (Finset.mem_univ a)
  have h2 := Finset.single_le_sum (fun a _ => abs_nonneg (p.1.1 a))
    (Finset.mem_univ (run (next p.1.2) (observations (p.1.1, p.2))))
  unfold continuationPayoff
  rw [Real.norm_eq_abs, ← ha]
  have hb := abs_add_le (p.1.1 a) (-p.1.1 (run (next p.1.2) (observations (p.1.1, p.2))))
  rw [abs_neg, ← sub_eq_add_neg] at hb
  linarith

theorem continuationPayoff_raw_integrable {H : ℕ} (next : ℝ → Strategy (A := A) H)
    (hnext : Measurable (fun p : ℝ × RawStream A H => run (next p.1) p.2))
    (c : Precisions A) (m : Means A) (a : A) :
    Integrable (fun p : (Means A × ℝ) × RawStream A H =>
      continuationPayoff next (ObservationBridge.rawObservation a p.1, p.2))
      (((posteriorLaw c m).prod (gaussianReal 0 1)).prod (arrayLaw H)) := by
  have hi : Integrable (fun p : (Means A × ℝ) × RawStream A H =>
      2 * ∑ b, |p.1.1 b|)
      (((posteriorLaw c m).prod (gaussianReal 0 1)).prod (arrayLaw H)) := by
    apply Integrable.const_mul
    apply integrable_finset_sum
    intro b _
    simpa only [Real.norm_eq_abs] using
      (((posterior_coordinate_integrable c m b).norm.comp_fst (gaussianReal 0 1)).comp_fst
        (arrayLaw H))
  apply hi.mono' (((continuationPayoff_measurable next hnext).comp
    (by fun_prop)).aestronglyMeasurable)
  exact Filter.Eventually.of_forall (fun p => continuationPayoff_norm_le next
    (ObservationBridge.rawObservation a p.1, p.2))

/-- Exact raw reward disintegration for an arbitrary measurable finite
adaptive tree; all unused calendar noise is integrated out explicitly. -/
theorem rawRisk_succ_data {H : ℕ} (a : A) (next : ℝ → Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hc : 0 < c a)
    (hnext : Measurable (fun p : ℝ × RawStream A H => run (next p.1) p.2)) :
    (rawRisk (.sample a next) c m =
      ∫ z, rawRisk (next (m a + ((c a : ℝ) + 1) * z)) (advance c a) (shift m a z)
        ∂gaussianReal 0 (innovation c a)) ∧
    Integrable (fun z => rawRisk (next (m a + ((c a : ℝ) + 1) * z))
      (advance c a) (shift m a z)) (gaussianReal 0 (innovation c a)) := by
  let F := continuationPayoff next
  have hF : Measurable F := continuationPayoff_measurable next hnext
  have hraw : Integrable (fun p : (Means A × ℝ) × RawStream A H =>
      F (ObservationBridge.rawObservation a p.1, p.2))
      (((posteriorLaw c m).prod (gaussianReal 0 1)).prod (arrayLaw H)) :=
    continuationPayoff_raw_integrable next hnext c m a
  have hsplit := (MeasurePreserving.symm MeasurableEquiv.prodAssoc
    (measurePreserving_prodAssoc (posteriorLaw c m) (gaussianReal 0 1) (arrayLaw H))).comp
      ((MeasurePreserving.id (posteriorLaw c m)).prod (selected_noise_split_preserving H a))
  have hfirst : rawRisk (.sample a next) c m =
      ∫ p : (Means A × ℝ) × RawStream A H,
        F (ObservationBridge.rawObservation a p.1, p.2)
          ∂((posteriorLaw c m).prod (gaussianReal 0 1)).prod (arrayLaw H) := by
    have hfraw : Measurable (fun p : (Means A × ℝ) × RawStream A H =>
        F (ObservationBridge.rawObservation a p.1, p.2)) := by
      exact hF.comp ((ObservationBridge.rawObservation_measurable a).prodMap measurable_id)
    rw [← hsplit.map_eq, integral_map hsplit.measurable.aemeasurable hfraw.aestronglyMeasurable]
    rfl
  let G : Means A × ℝ → ℝ := fun p => ∫ e, F (p, e) ∂arrayLaw H
  have hG : Measurable G := (hF.stronglyMeasurable.integral_prod_right').measurable
  have hsecond : rawRisk (.sample a next) c m =
      ∫ p : ℝ × Means A, G (ObservationBridge.reconstructedObservation c m a p)
        ∂(gaussianReal 0 (innovation c a)).prod (centeredLaw (advance c a)) := by
    rw [hfirst, integral_prod _ hraw]
    exact ObservationBridge.observation_integral c m a hc G hG
  have hrawMap : Integrable F
      ((((posteriorLaw c m).prod (gaussianReal 0 1)).map
        (ObservationBridge.rawObservation a)).prod (arrayLaw H)) := by
    have hpres : MeasurePreserving (ObservationBridge.rawObservation a)
        ((posteriorLaw c m).prod (gaussianReal 0 1))
        (((posteriorLaw c m).prod (gaussianReal 0 1)).map
          (ObservationBridge.rawObservation a)) :=
      ⟨by fun_prop, rfl⟩
    exact ((hpres.prod (MeasurePreserving.id (arrayLaw H))).integrable_comp
      hF.aestronglyMeasurable).mp hraw
  rw [ObservationBridge.observation_joint_law c m a hc] at hrawMap
  have hpost : Integrable (fun p : (ℝ × Means A) × RawStream A H =>
      F (ObservationBridge.reconstructedObservation c m a p.1, p.2))
      (((gaussianReal 0 (innovation c a)).prod (centeredLaw (advance c a))).prod
        (arrayLaw H)) := by
    have hpres : MeasurePreserving (ObservationBridge.reconstructedObservation c m a)
        ((gaussianReal 0 (innovation c a)).prod (centeredLaw (advance c a)))
        (((gaussianReal 0 (innovation c a)).prod (centeredLaw (advance c a))).map
          (ObservationBridge.reconstructedObservation c m a)) := ⟨by fun_prop, rfl⟩
    exact (hpres.prod (MeasurePreserving.id (arrayLaw H))).integrable_comp_of_integrable hrawMap
  have hpostG : Integrable (fun p : ℝ × Means A =>
      G (ObservationBridge.reconstructedObservation c m a p))
      ((gaussianReal 0 (innovation c a)).prod (centeredLaw (advance c a))) :=
    hpost.integral_prod_left
  rw [integral_prod _ hpostG] at hsecond
  have hzEq (z : ℝ) :
      (∫ u : Means A, G (ObservationBridge.reconstructedObservation c m a (z, u))
        ∂centeredLaw (advance c a)) =
      rawRisk (next (m a + ((c a : ℝ) + 1) * z)) (advance c a) (shift m a z) := by
    have hr : Measurable (run (next (m a + ((c a : ℝ) + 1) * z))) :=
      hnext.comp (measurable_const.prodMk measurable_id)
    have hc' : Measurable (fun p : (Means A × RawStream A H) × A => p.1.1 p.2) :=
      measurable_from_prod_countable_left (fun b => by
        change Measurable (fun p : Means A × RawStream A H => p.1 b)
        fun_prop)
    have hfz : Measurable (fun p : Means A × RawStream A H =>
        maxAll p.1 - p.1 (run (next (m a + ((c a : ℝ) + 1) * z)) (observations p))) :=
      ((maxAll_lipschitz (A := A)).continuous.measurable.comp measurable_fst).sub
        (hc'.comp (measurable_id.prodMk (hr.comp (observations_measurable H))))
    unfold G F rawRisk worldLaw
    simp only [posteriorLaw]
    conv_rhs => rw [← Measure.map_id (μ := arrayLaw (A := A) H)]
    rw [
      Measure.map_prod_map _ _ (by fun_prop) measurable_id,
      integral_map (by fun_prop) hfz.aestronglyMeasurable]
    have hz : Integrable (fun p : Means A × RawStream A H =>
        continuationPayoff next
          (ObservationBridge.reconstructedObservation c m a (z, p.1), p.2))
        ((centeredLaw (advance c a)).prod (arrayLaw H)) := by
      have hi := rawRisk_integrable (next (m a + ((c a : ℝ) + 1) * z))
        (advance c a) (shift m a z) hr
      simp only [worldLaw, posteriorLaw] at hi
      rw [← Measure.map_id (μ := arrayLaw (A := A) H),
        Measure.map_prod_map _ _ (by fun_prop) measurable_id] at hi
      exact hi.comp_measurable (by fun_prop)
    exact (integral_prod _ hz).symm
  constructor
  · simpa only [hzEq] using hsecond
  · simpa only [hzEq] using hpostG.integral_prod_left

theorem rawRisk_succ {H : ℕ} (a : A) (next : ℝ → Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hc : 0 < c a)
    (hnext : Measurable (fun p : ℝ × RawStream A H => run (next p.1) p.2)) :
    rawRisk (.sample a next) c m =
      ∫ z, rawRisk (next (m a + ((c a : ℝ) + 1) * z)) (advance c a) (shift m a z)
        ∂gaussianReal 0 (innovation c a) :=
  (rawRisk_succ_data a next c m hc hnext).1

/-- Raw-history strategies, including strategies that deliberately discard
some real observations, cannot improve on the actual Gaussian Bellman loss. -/
theorem rawRisk_ge_loss {H : ℕ} (s : Strategy (A := A) H) (c : Precisions A)
    (m : Means A) (hc : ∀ a, 0 < c a) (hs : Regular s) :
    loss H c m ≤ rawRisk s c m := by
  induction H generalizing c m with
  | zero =>
    cases s with
    | stop a =>
      rw [rawRisk_zero, terminalRegret_eq]
      exact sub_le_sub_left (coordinate_le_maxAll m a) (potential c m)
  | succ H ih =>
    cases s with
    | sample a next =>
      have hdata := rawRisk_succ_data a next c m (hc a) hs.2
      have hi : Integrable (fun z => loss H (advance c a) (shift m a z))
          (gaussianReal 0 (innovation c a)) :=
        (shifted_integrable (potential_lipschitz _) m a _).sub (action_integrable H c m a)
      have h := integral_mono hi hdata.2 (fun z => ih (next _)
        (advance c a) (shift m a z) (advance_positive c a hc) (hs.1 _))
      rw [← hdata.1] at h
      change actionLoss H c m a ≤ rawRisk (.sample a next) c m at h
      rw [actionLoss_eq H c m a (hc a)] at h
      have ha := action_le_value H c m a
      unfold loss
      linarith

theorem rawRisk_ge_actionLoss {H : ℕ} (a : A) (next : ℝ → Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hc : ∀ b, 0 < c b)
    (hs : Regular (.sample a next)) :
    actionLoss H c m a ≤ rawRisk (.sample a next) c m := by
  have hdata := rawRisk_succ_data a next c m (hc a) hs.2
  have hi : Integrable (fun z => loss H (advance c a) (shift m a z))
      (gaussianReal 0 (innovation c a)) :=
    (shifted_integrable (potential_lipschitz _) m a _).sub (action_integrable H c m a)
  have h := integral_mono hi hdata.2 (fun z => rawRisk_ge_loss (next _)
    (advance c a) (shift m a z) (advance_positive c a hc) (hs.1 _))
  rw [← hdata.1] at h
  exact h

theorem recovered_innovation (c : ℝ≥0) (m z : ℝ) :
    (m + ((c : ℝ) + 1) * z - m) / ((c : ℝ) + 1) = z := by
  have h : (c : ℝ) + 1 ≠ 0 := by positivity
  field_simp [h]
  ring

/-- Actual latent-parameter/noise-array execution has exactly the same
posterior expected simple regret as the innovation-coordinate strategy. -/
theorem rawOfInnovation_risk {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions A) (m : Means A) (hc : ∀ a, 0 < c a)
    (hs : Regular (rawOfInnovation s c m)) :
    rawRisk (rawOfInnovation s c m) c m = expectedRegret s c m := by
  induction H generalizing c m with
  | zero => cases s with | stop a => exact rawRisk_zero a c m
  | succ H ih =>
    cases s with
    | sample a next =>
      let continuation : ℝ → Strategy (A := A) H := fun y =>
        let z := (y - m a) / ((c a : ℝ) + 1)
        rawOfInnovation (next z) (advance c a) (shift m a z)
      have hs' : Regular (.sample a continuation) := hs
      change rawRisk (.sample a continuation) c m = _
      rw [rawRisk_succ a continuation c m (hc a) hs'.2]
      apply integral_congr_ae
      apply Filter.Eventually.of_forall
      intro z
      have hreg := hs'.1 (m a + ((c a : ℝ) + 1) * z)
      simp only [continuation, recovered_innovation] at hreg ⊢
      exact ih (next z) (advance c a) (shift m a z) (advance_positive c a hc) hreg

/-- A genuine measurable raw-history optimal policy attains Bellman loss;
the result uses the proved observation disintegration, rather than assuming
that the latent Gaussian execution agrees with the Bellman recursion. -/
theorem optimalFirst_rawRisk (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) :
    rawRisk (rawOfInnovation (optimalFirst H c m) c m) c m = loss H c m := by
  rw [rawOfInnovation_risk _ _ _ hc (optimalFirst_raw_regular H c m),
    expectedRegret_eq _ _ _ (optimalFirst_admissible H c m) hc,
    optimalFirst_utility]
  rfl

end Measurable
end
end Junpei.EmbeddingExecution
