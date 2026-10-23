import Junpei.RealPrecisionPolicy
import Junpei.InitialHistoryMain

set_option autoImplicit false
set_option warningAsError true
set_option maxHeartbeats 1000000

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal ENNReal

namespace Junpei.RealPrecisionMain

open Draft0.BayesianModel
open Junpei.CountModelBridge Junpei.PolicyContinuation Junpei.SequentialRisk

noncomputable section

/-- The paper's real-precision Gaussian-state policy, restricted to the
integer-count states reached after flat-prior initialization at horizon T. -/
def restrict (π : RealPrecisionPolicy.Policy Arm) (T : ℕ) : Junpei.Policy.StatePolicy where
  draw := fun t c m u => π.draw (T - t) (natPrecisions c) m u
  recommend := fun c m u => π.recommend (natPrecisions c) m u
  draw_measurable := by
    intro t
    have hc : Measurable (fun p : Counts × Means × ℝ => natPrecisions p.1) :=
      (measurable_of_countable natPrecisions).comp measurable_fst
    have hm : Measurable (fun p : Counts × Means × ℝ => p.2.1) :=
      measurable_fst.comp measurable_snd
    exact (π.draw_measurable (T - t)).comp
      ((hc.prodMk hm).prodMk (measurable_snd.comp measurable_snd))
  recommend_measurable := by
    have hc : Measurable (fun p : Counts × Means × ℝ => natPrecisions p.1) :=
      (measurable_of_countable natPrecisions).comp measurable_fst
    have hm : Measurable (fun p : Counts × Means × ℝ => p.2.1) :=
      measurable_fst.comp measurable_snd
    exact π.recommend_measurable.comp
      ((hc.prodMk hm).prodMk (measurable_snd.comp measurable_snd))

/-- The integer restriction has literally the original real-precision raw
strategy at every consistent remaining horizon. -/
theorem rawTree_restriction (π : RealPrecisionPolicy.Policy Arm) (T t H : ℕ)
    (c : Counts) (m : Means) (u : Seeds H) (horizon : t + H = T + 1) :
    Junpei.PolicyContinuation.rawTree (restrict π T) t H c m u =
      RealPrecisionPolicy.rawTree π H (natPrecisions c) m u := by
  induction H generalizing t c m with
  | zero => rfl
  | succ H ih =>
    have hremaining : T - t = H := by omega
    let a := π.draw H (natPrecisions c) m (u ⟨0, by omega⟩)
    have htree : Junpei.PolicyContinuation.rawTree (restrict π T) t (H + 1) c m u =
        Junpei.FiniteGaussian.Strategy.sample a
          (fun y => Junpei.PolicyContinuation.rawTree (restrict π T) (t + 1) H (advance c a)
            (shift m a ((y - m a) / ((c a : ℝ) + 1))) (seedTail u)) := by
      simp only [Junpei.PolicyContinuation.rawTree, restrict, hremaining]
      rfl
    rw [htree, RealPrecisionPolicy.rawTree]
    apply congrArg (Junpei.FiniteGaussian.Strategy.sample a)
    funext y
    simpa only [advance_natPrecisions, shift_eq, natPrecisions, NNReal.coe_natCast] using
      ih (t + 1) (advance c a) (shift m a ((y - m a) / ((c a : ℝ) + 1)))
        (seedTail u) (by omega)

theorem continuationRisk_restriction (π : RealPrecisionPolicy.Policy Arm) (T t H : ℕ)
    (c : Counts) (m : Means) (horizon : t + H = T + 1) :
    continuationRisk (restrict π T) t H c m =
      RealPrecisionPolicy.actualRisk π H (natPrecisions c) m := by
  unfold continuationRisk familyRisk RealPrecisionPolicy.actualRisk
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro u
  change EmbeddingExecution.rawRisk (Junpei.PolicyContinuation.rawTree (restrict π T) t H c m u)
    (natPrecisions c) m = _
  rw [rawTree_restriction π T t H c m u horizon]

/-- Only the INITIAL risk is minimized, against concrete Borel history
policies. This imposes no optimality at any later or unreachable state, and
no requirement at horizons larger than this fixed T. -/
def InitialHistoryRiskOptimal (π : RealPrecisionPolicy.Policy Arm) (T : ℕ) : Prop :=
  ∀ (m : Means) (ρ : HistoryPolicy.Policy Arm),
    RealPrecisionPolicy.actualRisk π (T - 3) (natPrecisions (fun _ => 1)) m ≤
      HistoryPolicy.actualRisk ρ 0 (T - 3) PosteriorStateHistory.emptyHistory
        (natPrecisions (fun _ => 1)) m

theorem initialHistoryRiskOptimal_iff_restriction (π : RealPrecisionPolicy.Policy Arm)
    (T : ℕ) (hT : 3 ≤ T) :
    InitialHistoryRiskOptimal π T ↔
      PosteriorStateHistory.InitialHistoryRiskOptimal (restrict π T) T := by
  constructor <;> intro hπ m ρ
  · rw [continuationRisk_restriction π T 4 (T - 3) (fun _ => 1) m (by omega)]
    exact hπ m ρ
  · have hp := hπ m ρ
    rw [continuationRisk_restriction π T 4 (T - 3) (fun _ => 1) m (by omega)] at hp
    exact hp

theorem sequentiallyOptimal_implies_initial (π : RealPrecisionPolicy.Policy Arm) (T : ℕ)
    (hπ : RealPrecisionPolicy.SequentiallyOptimal π) : InitialHistoryRiskOptimal π T := by
  intro m ρ
  exact hπ (T - 3) (natPrecisions (fun _ => 1)) m (by intro a; norm_num [natPrecisions])
    ρ 0 PosteriorStateHistory.emptyHistory

theorem sequentiallyOptimal_restriction (π : RealPrecisionPolicy.Policy Arm) (T : ℕ)
    (hπ : RealPrecisionPolicy.SequentiallyOptimal π) : SequentialRiskOptimal (restrict π T) T := by
  constructor
  · intro t c m _ htT hc
    apply (PosteriorStateHistory.minimizesHistoryRisk_iff (restrict π T) T t
      (T - t + 1) c m (by omega) hc).mp
    intro ρ
    rw [continuationRisk_restriction π T t (T - t + 1) c m (by omega)]
    exact hπ _ _ _ (natPrecisions_positive c hc) ρ 0 PosteriorStateHistory.emptyHistory
  · intro c m hc
    apply (PosteriorStateHistory.minimizesHistoryRisk_iff (restrict π T) T (T + 1)
      0 c m (by omega) hc).mp
    intro ρ
    rw [continuationRisk_restriction π T (T + 1) 0 c m (by omega)]
    exact hπ _ _ _ (natPrecisions_positive c hc) ρ 0 PosteriorStateHistory.emptyHistory

theorem regret_repair_eq (π : RealPrecisionPolicy.Policy Arm) (T : ℕ) (hT : 3 ≤ T)
    (Δ : ℝ) (hπ : InitialHistoryRiskOptimal π T) :
    Junpei.Main.regret (restrict π T) T Δ =
      Junpei.Main.regret (Junpei.Policy.repair (restrict π T) T) T Δ :=
  InitialHistoryMain.regret_repair_eq (restrict π T) T Δ hT
    ((initialHistoryRiskOptimal_iff_restriction π T hT).mp hπ)

/-- The paper's full real-precision posterior-state policy class connects
to the actual frequentist Main.regret using INITIAL history-risk optimality
alone. The global sequential policy predicate is not a premise. -/
theorem theorem2 (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (C T0 : ℕ), 0 < c ∧ 0 < C ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : RealPrecisionPolicy.Policy Arm,
        InitialHistoryRiskOptimal π T →
          c * Draft0.Rates.invPower C T ≤ Junpei.Main.regret (restrict π T) T Δ := by
  obtain ⟨c, C, T0, hc, hC, hmain⟩ := InitialHistoryMain.theorem2 Δ hΔ
  refine ⟨c, C, max T0 3, hc, hC, ?_⟩
  intro T hT ho π hπ
  exact hmain T ((le_max_left T0 3).trans hT) ho (restrict π T)
    ((initialHistoryRiskOptimal_iff_restriction π T ((le_max_right T0 3).trans hT)).mp hπ)

theorem theorem2_paper_exponent (Δ : ℝ) (hΔ : 0 < Δ) :
    ∃ (c : ℝ) (T0 : ℕ), 0 < c ∧
      ∀ T : ℕ, T0 ≤ T → Odd T → ∀ π : RealPrecisionPolicy.Policy Arm,
        InitialHistoryRiskOptimal π T →
          c * (T : ℝ)^(-((33+Δ)^2/2+4)) ≤ Junpei.Main.regret (restrict π T) T Δ := by
  obtain ⟨c, T0, hc, hmain⟩ := InitialHistoryMain.theorem2_paper_exponent Δ hΔ
  refine ⟨c, max T0 3, hc, ?_⟩
  intro T hT ho π hπ
  exact hmain T ((le_max_left T0 3).trans hT) ho (restrict π T)
    ((initialHistoryRiskOptimal_iff_restriction π T ((le_max_right T0 3).trans hT)).mp hπ)

theorem theorem2_exponential_rate (Δ : ℝ) (hΔ : 0 < Δ)
    (family : ℕ → RealPrecisionPolicy.Policy Arm)
    (hfamily : ∀ T, 3 ≤ T → InitialHistoryRiskOptimal (family T) T) :
    liminf (Junpei.RegretExponent.rate
      (fun T => Junpei.Main.regret (restrict (family T) T) T Δ)) atTop = 0 := by
  apply InitialHistoryMain.theorem2_exponential_rate Δ hΔ (fun T => restrict (family T) T)
  intro T hT
  exact (initialHistoryRiskOptimal_iff_restriction (family T) T hT).mp (hfamily T hT)

theorem remark2 (Δ : ℝ) (hΔ : 0 < Δ) (family : ℕ → RealPrecisionPolicy.Policy Arm)
    (hfamily : ∀ T, 3 ≤ T → InitialHistoryRiskOptimal (family T) T)
    (rFreq : ℕ → ℝ) (hF : ∀ T, 0 ≤ rFreq T)
    (hpositive : 0 < liminf (Junpei.RegretExponent.rate rFreq) atTop) :
    limsup (Junpei.RegretRatio.ratio
      (fun T => Junpei.Main.regret (restrict (family T) T) T Δ) rFreq) atTop = ⊤ := by
  apply InitialHistoryMain.remark2 Δ hΔ (fun T => restrict (family T) T) _ rFreq hF hpositive
  intro T hT
  exact (initialHistoryRiskOptimal_iff_restriction (family T) T hT).mp (hfamily T hT)

theorem exists_initialHistoryRiskOptimal (T : ℕ) :
    ∃ π : RealPrecisionPolicy.Policy Arm, InitialHistoryRiskOptimal π T :=
  ⟨RealPrecisionPolicy.canonical, sequentiallyOptimal_implies_initial _ T
    RealPrecisionPolicy.canonical_sequentiallyOptimal⟩

end
end Junpei.RealPrecisionMain
