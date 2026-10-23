import Junpei.CountModelBridge
import Junpei.PolicyContinuation
import Junpei.PolicyContinuationOptimal
import Junpei.PolicySeeds
import Junpei.EmbeddingRandomized
import Junpei.BayesOptimal
import Junpei.MainExactRate
import Junpei.RegretRatio

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal ENNReal

namespace Junpei.SequentialRisk

open Draft0.BayesianModel
open Junpei.Policy Junpei.PolicyContinuation Junpei.CountModelBridge
open Junpei.EmbeddingRaw Junpei.EmbeddingExecution Junpei.EmbeddingRandomized

noncomputable section

/-- A continuation family sees an independent Gaussian seed for each draw
and for the terminal recommendation. Its raw tree sees only selected rewards. -/
abbrev Family (H : ℕ) := Seeds H → Junpei.FiniteGaussian.Strategy (A := Arm) H

def Feasible {H : ℕ} (s : Family H) : Prop :=
  (∀ u, Regular (s u)) ∧
    Measurable (fun p : Seeds H × RawStream Arm H => run (s p.1) p.2)

/-- Actual posterior expected simple regret, integrated over independent
policy seeds and the latent Gaussian/noise-array world used by rawRisk. -/
def familyRisk {H : ℕ} (s : Family H) (c : Counts) (m : Means) : ℝ :=
  ∫ u, rawRisk (s u) (natPrecisions c) m ∂seedLaw (H + 1)

def continuationRisk (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means) : ℝ :=
  familyRisk (rawTree π t H c m) c m

/-- This is risk minimization, with no Bellman-action premise. Competitors
are regular Borel raw-history trees with independent Gaussian randomization. -/
def MinimizesActualRisk (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means) : Prop :=
  ∀ s : Family H, Feasible s → continuationRisk π t H c m ≤ familyRisk s c m

/-- Sequential actual-risk minimization at all proper integer-count states.
The hypothesis is stated in actual posterior regret, not local action values. -/
def SequentialRiskOptimal (π : StatePolicy) (T : ℕ) : Prop :=
  (∀ (t : ℕ) (c : Counts) (m : Means), 4 ≤ t → t ≤ T → (∀ a, 0 < c a) →
    MinimizesActualRisk π t (T - t + 1) c m) ∧
  (∀ (c : Counts) (m : Means), (∀ a, 0 < c a) →
    MinimizesActualRisk π (T + 1) 0 c m)

theorem rawTree_feasible (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means) :
    Feasible (rawTree π t H c m) := by
  constructor
  · exact rawTree_regular π t H c m
  · exact (rawTree_run_measurable π t H c).comp
      (measurable_const.prodMk measurable_id)

theorem familyRisk_integrable {H : ℕ} (s : Family H) (c : Counts) (m : Means)
    (hs : Feasible s) :
    Integrable (fun u => rawRisk (s u) (natPrecisions c) m) (seedLaw (H + 1)) :=
  randomized_rawRisk_integrable (seedLaw (H + 1)) s (natPrecisions c) m hs.2

theorem familyRisk_eq_joint {H : ℕ} (s : Family H) (c : Counts) (m : Means)
    (hs : Feasible s) :
    familyRisk s c m =
      ∫ p : Seeds H × (Junpei.FiniteGaussian.Means Arm × RawStream Arm H),
        randomizedPayoff s p
        ∂(seedLaw (H + 1)).prod (worldLaw H (natPrecisions c) m) := by
  exact (integral_prod _
    (randomizedPayoff_integrable (seedLaw (H + 1)) s (natPrecisions c) m hs.2)).symm

theorem familyRisk_ge_loss {H : ℕ} (s : Family H) (c : Counts) (m : Means)
    (hs : Feasible s) (hc : ∀ a, 0 < c a) :
    Junpei.FinitePosterior.loss H (natPrecisions c) m ≤ familyRisk s c m :=
  randomized_rawRisk_ge_loss (seedLaw (H + 1)) s (natPrecisions c) m
    (natPrecisions_positive c hc) hs.1 hs.2

def canonicalFamily (H : ℕ) (c : Counts) (m : Means) : Family H :=
  fun _ => rawOfInnovation (Junpei.FiniteModelMeasurable.optimalFirst H (natPrecisions c) m)
    (natPrecisions c) m

theorem canonicalFamily_feasible (H : ℕ) (c : Counts) (m : Means) :
    Feasible (canonicalFamily H c m) := by
  constructor
  · intro _
    exact optimalFirst_raw_regular H (natPrecisions c) m
  · exact (regular_run_measurable _
      (optimalFirst_raw_regular H (natPrecisions c) m)).comp measurable_snd

/-- The comparator risk identity is derived from the actual Gaussian
observation model by the proved raw-observation execution bridge. -/
theorem canonicalFamily_risk (H : ℕ) (c : Counts) (m : Means) (hc : ∀ a, 0 < c a) :
    familyRisk (canonicalFamily H c m) c m =
      Junpei.FinitePosterior.loss H (natPrecisions c) m := by
  unfold familyRisk canonicalFamily
  simp only [optimalFirst_rawRisk H (natPrecisions c) m (natPrecisions_positive c hc)]
  simp

theorem minimizing_risk_le_loss (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t H c m) :
    continuationRisk π t H c m ≤ Junpei.FinitePosterior.loss H (natPrecisions c) m := by
  have h := hπ (canonicalFamily H c m) (canonicalFamily_feasible H c m)
  rwa [canonicalFamily_risk H c m hc] at h

theorem minimizing_risk_eq_loss (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t H c m) :
    continuationRisk π t H c m = Junpei.FinitePosterior.loss H (natPrecisions c) m :=
  le_antisymm (minimizing_risk_le_loss π t H c m hc hπ)
    (familyRisk_ge_loss _ c m (rawTree_feasible π t H c m) hc)

theorem integrable_finite_selector {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] (f : Arm → ℝ) (a : Ω → Arm)
    (ha : Measurable a) : Integrable (fun u => f (a u)) μ := by
  have hm : Measurable (fun u => f (a u)) := (measurable_of_countable f).comp ha
  apply (integrable_const (∑ b : Arm, |f b|)).mono' hm.aestronglyMeasurable
  apply Filter.Eventually.of_forall
  intro u
  rw [Real.norm_eq_abs]
  exact Finset.single_le_sum (fun b _ => abs_nonneg (f b)) (Finset.mem_univ (a u))

theorem ae_eq_const_of_lower_integral {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (f : Ω → ℝ) (a : ℝ)
    (hi : Integrable f μ) (hl : ∀ u, a ≤ f u) (hu : (∫ u, f u ∂μ) ≤ a) :
    ∀ᵐ u ∂μ, f u = a := by
  have hlo : (∫ _ : Ω, a ∂μ) ≤ ∫ u, f u ∂μ :=
    integral_mono (integrable_const a) hi hl
  have he : (∫ _ : Ω, a ∂μ) = ∫ u, f u ∂μ := by
    simp only [integral_const, measureReal_univ_eq_one, one_smul] at hlo ⊢
    exact le_antisymm hlo hu
  exact ((integral_eq_iff_of_ae_le (integrable_const a) hi
    (Filter.Eventually.of_forall hl)).mp he).symm

theorem seed_head_ae (H : ℕ) (p : ℝ → Prop) (hp : MeasurableSet {u | p u})
    (h : ∀ᵐ u : Seeds H ∂seedLaw (H + 1), p (u ⟨0, by omega⟩)) :
    ∀ᵐ u : ℝ ∂gaussianReal 0 1, p u := by
  let head : Seeds H → ℝ := fun u => u ⟨0, by omega⟩
  have hm : Measurable head := measurable_pi_apply (⟨0, by omega⟩ : Fin (H + 1))
  have hmap : (seedLaw (H + 1)).map head = gaussianReal 0 1 :=
    (measurePreserving_eval (fun _ : Fin (H + 1) => gaussianReal 0 1)
      (⟨0, by omega⟩ : Fin (H + 1))).map_eq
  have hh : ∀ᵐ u : ℝ ∂(seedLaw (H + 1)).map head, p u :=
    (ae_map_iff hm.aemeasurable hp).mpr h
  rwa [hmap] at hh

theorem minimizing_draw_ae (π : StatePolicy) (t H : ℕ) (c : Counts) (m : Means)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t (H + 1) c m) :
    ∀ᵐ u : ℝ ∂gaussianReal 0 1,
      action H c m (π.draw t c m u) = value (H + 1) c m := by
  let a : Seeds (H + 1) → Arm := fun u => π.draw t c m (u ⟨0, by omega⟩)
  let f : Seeds (H + 1) → ℝ := fun u =>
    Junpei.FinitePosterior.actionLoss H (natPrecisions c) m (a u)
  have ha : Measurable a := (π.draw_measurable t).comp
    (measurable_const.prodMk (measurable_const.prodMk (measurable_pi_apply _)))
  have hi : Integrable f (seedLaw (H + 2)) :=
    integrable_finite_selector _ _ a ha
  have hpoint (u : Seeds (H + 1)) :
      f u ≤ rawRisk (rawTree π t (H + 1) c m u) (natPrecisions c) m := by
    exact rawRisk_ge_actionLoss (a u) _ (natPrecisions c) m
      (natPrecisions_positive c hc) (rawTree_regular π t (H + 1) c m u)
  have hupper : (∫ u, f u ∂seedLaw (H + 2)) ≤
      Junpei.FinitePosterior.loss (H + 1) (natPrecisions c) m := by
    exact (integral_mono hi
      (familyRisk_integrable _ c m (rawTree_feasible π t (H + 1) c m)) hpoint).trans
      (minimizing_risk_le_loss π t (H + 1) c m hc hπ)
  have he := ae_eq_const_of_lower_integral (seedLaw (H + 2)) f
    (Junpei.FinitePosterior.loss (H + 1) (natPrecisions c) m) hi
    (fun u => Junpei.BayesOptimal.actionLoss_minimal H (natPrecisions c) m
      (natPrecisions_positive c hc) (a u)) hupper
  have hh : ∀ᵐ u : Seeds (H + 1) ∂seedLaw (H + 2),
      action H c m (π.draw t c m (u ⟨0, by omega⟩)) = value (H + 1) c m := by
    filter_upwards [he] with u hu
    exact (CountModelBridge.actionLoss_eq_loss_iff H c m (a u) (hc (a u))).mp hu
  have hp : MeasurableSet {u : ℝ |
      action H c m (π.draw t c m u) = value (H + 1) c m} := by
    have hd : Measurable (π.draw t c m) := (π.draw_measurable t).comp
      (measurable_const.prodMk (measurable_const.prodMk measurable_id))
    exact measurableSet_eq_fun ((measurable_of_countable (action H c m)).comp hd)
      measurable_const
  exact seed_head_ae (H + 1) _ hp hh

theorem minimizing_recommend_ae (π : StatePolicy) (t : ℕ) (c : Counts) (m : Means)
    (hc : ∀ a, 0 < c a) (hπ : MinimizesActualRisk π t 0 c m) :
    ∀ᵐ u : ℝ ∂gaussianReal 0 1, m (π.recommend c m u) = terminal m := by
  let a : Seeds 0 → Arm := fun u => π.recommend c m (u ⟨0, by omega⟩)
  let f : Seeds 0 → ℝ := fun u =>
    Junpei.FinitePosterior.terminalRegret (natPrecisions c) m (a u)
  have ha : Measurable a := π.recommend_measurable.comp
    (measurable_const.prodMk (measurable_const.prodMk (measurable_pi_apply _)))
  have hi : Integrable f (seedLaw 1) := integrable_finite_selector _ _ a ha
  have hf (u : Seeds 0) : f u = rawRisk (rawTree π t 0 c m u) (natPrecisions c) m := by
    exact (rawRisk_zero (a u) (natPrecisions c) m).symm
  have hupper : (∫ u, f u ∂seedLaw 1) ≤
      Junpei.FinitePosterior.loss 0 (natPrecisions c) m := by
    simpa only [hf] using minimizing_risk_le_loss π t 0 c m hc hπ
  have hl (u : Seeds 0) : Junpei.FinitePosterior.loss 0 (natPrecisions c) m ≤ f u := by
    rw [hf]
    exact rawRisk_ge_loss _ _ _ (natPrecisions_positive c hc)
      (rawTree_regular π t 0 c m u)
  have he := ae_eq_const_of_lower_integral (seedLaw 1) f
    (Junpei.FinitePosterior.loss 0 (natPrecisions c) m) hi hl hupper
  have hh : ∀ᵐ u : Seeds 0 ∂seedLaw 1,
      m (π.recommend c m (u ⟨0, by omega⟩)) = terminal m := by
    filter_upwards [he] with u hu
    change Junpei.FinitePosterior.terminalRegret (natPrecisions c) m (a u) = _ at hu
    rw [Junpei.FinitePosterior.terminalRegret_eq] at hu
    unfold Junpei.FinitePosterior.loss at hu
    rw [CountModelBridge.value_eq, Draft0.BayesianModel.value] at hu
    linarith
  have hp : MeasurableSet {u : ℝ | m (π.recommend c m u) = terminal m} := by
    have hd : Measurable (π.recommend c m) := π.recommend_measurable.comp
      (measurable_const.prodMk (measurable_const.prodMk measurable_id))
    exact measurableSet_eq_fun ((measurable_of_countable m).comp hd) measurable_const
  exact seed_head_ae 0 _ hp hh

/-- Actual sequential posterior-regret minimization implies the local
almost-everywhere Bellman optimality needed by the frequentist main proof. -/
theorem sequentialRiskOptimal_implies_aeOptimal (π : StatePolicy) (T : ℕ)
    (hπ : SequentialRiskOptimal π T) : AEOptimal π T := by
  constructor
  · intro t c m ht htT hc
    exact minimizing_draw_ae π t (T - t) c m hc (hπ.1 t c m ht htT hc)
  · intro c m hc
    exact minimizing_recommend_ae π (T + 1) c m hc (hπ.2 c m hc)

/-- The measurable deterministic Bellman state policy attains the minimum
against all feasible randomized raw-history competitors. -/
theorem deterministicBellman_minimizes_actual_risk (T t H : ℕ) (c : Counts) (m : Means)
    (horizon : t + H = T + 1) (hc : ∀ a, 0 < c a) :
    MinimizesActualRisk (deterministicBellman T) t H c m := by
  intro s hs
  have he : continuationRisk (deterministicBellman T) t H c m =
      Junpei.FinitePosterior.loss H (natPrecisions c) m := by
    unfold continuationRisk familyRisk
    simp only [Junpei.PolicyContinuationOptimal.deterministicBellman_rawRisk T t H c m
      _ horizon hc]
    simp
  rw [he]
  exact familyRisk_ge_loss s c m hs hc

/-- In particular the actual-risk policy class used by the main wrapper has
an explicitly constructed measurable policy at every horizon. -/
theorem deterministicBellman_sequentialRiskOptimal (T : ℕ) :
    SequentialRiskOptimal (deterministicBellman T) T := by
  constructor
  · intro t c m _ htT hc
    exact deterministicBellman_minimizes_actual_risk T t (T - t + 1) c m (by omega) hc
  · intro c m hc
    exact deterministicBellman_minimizes_actual_risk T (T + 1) 0 c m (by omega) hc

theorem exists_sequentialRiskOptimal (T : ℕ) :
    ∃ π : StatePolicy, SequentialRiskOptimal π T :=
  ⟨deterministicBellman T, deterministicBellman_sequentialRiskOptimal T⟩

/-- Main polynomial bound with only actual sequential-risk optimality as
the policy hypothesis. Local Bellman optimality is derived inside Lean. -/
theorem theorem2_sequentialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (C T0 : ℕ), 0 < c ∧ 0 < C ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, SequentialRiskOptimal π T →
        c * Draft0.Rates.invPower C T ≤ Junpei.Main.regret π T Δ := by
  obtain ⟨c, C, T0, hc, hC, hmain⟩ := Junpei.Main.theorem2 Δ hΔ
  exact ⟨c, C, T0, hc, hC, fun T hT ho π hπ =>
    hmain T hT ho π (sequentialRiskOptimal_implies_aeOptimal π T hπ)⟩

theorem theorem2_paper_exponent_sequentialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (T0 : ℕ), 0 < c ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : StatePolicy, SequentialRiskOptimal π T →
        c * (T : ℝ)^(-((33+Δ)^2/2+4)) ≤ Junpei.Main.regret π T Δ := by
  obtain ⟨c, T0, hc, hmain⟩ := Junpei.Main.theorem2_paper_exponent Δ hΔ
  exact ⟨c, T0, hc, fun T hT ho π hπ =>
    hmain T hT ho π (sequentialRiskOptimal_implies_aeOptimal π T hπ)⟩

theorem theorem2_exponential_rate_sequentialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ)
    (family : ℕ → StatePolicy)
    (hfamily : ∀ T, 3 ≤ T → SequentialRiskOptimal (family T) T) :
    Filter.liminf (Junpei.RegretExponent.rate
      (fun T => Junpei.Main.regret (family T) T Δ)) atTop = 0 :=
  Junpei.Main.theorem2_exponential_rate Δ hΔ family
    (fun T hT => sequentialRiskOptimal_implies_aeOptimal (family T) T (hfamily T hT))

theorem remark2_sequentialRiskOptimal (Δ : ℝ) (hΔ : 0 < Δ)
    (family : ℕ → StatePolicy)
    (hfamily : ∀ T, 3 ≤ T → SequentialRiskOptimal (family T) T)
    (rFreq : ℕ → ℝ) (hF : ∀ T, 0 ≤ rFreq T)
    (hpositive : 0 < liminf (Junpei.RegretExponent.rate rFreq) atTop) :
    limsup (Junpei.RegretRatio.ratio
      (fun T => Junpei.Main.regret (family T) T Δ) rFreq) atTop = ⊤ :=
  Junpei.RegretRatio.remark2 Δ hΔ family
    (fun T hT => sequentialRiskOptimal_implies_aeOptimal (family T) T (hfamily T hT))
    rFreq hF hpositive

end
end Junpei.SequentialRisk
