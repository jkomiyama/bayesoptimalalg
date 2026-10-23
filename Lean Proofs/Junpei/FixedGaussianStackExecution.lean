import Junpei.FixedGaussianStackLaws
import Junpei.PolicyContinuation

set_option linter.unusedSectionVars false
set_option maxHeartbeats 800000

open MeasureTheory ProbabilityTheory

namespace Junpei.FixedGaussian

open FiniteGaussian EmbeddingRaw EmbeddingExecution

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

/-- The same raw-observation strategy, now fed rewards indexed by the
number of pulls of each arm rather than by calendar time. -/
def stackRun : {H : ℕ} → Strategy (A := A) H → RawStream A H → A
  | 0, .stop a, _ => a
  | _ + 1, .sample a next, r =>
      stackRun (next (r a ⟨0, by omega⟩)) (stackTail a r)

section Measurable
variable [MeasurableSpace A] [MeasurableSingletonClass A]

def StackRegular : {H : ℕ} → Strategy (A := A) H → Prop
  | 0, .stop _ => True
  | _ + 1, .sample _ next =>
      (∀ y, StackRegular (next y)) ∧
        Measurable (fun p : ℝ × RawStream A _ => stackRun (next p.1) p.2)

theorem stackRegular_run_measurable {H : ℕ} (s : Strategy (A := A) H)
    (hs : StackRegular s) : Measurable (stackRun s) := by
  cases s with
  | stop a => exact measurable_const
  | @sample H a next =>
    exact hs.2.comp ((by fun_prop : Measurable (fun r : RawStream A (H + 1) =>
      r a ⟨0, Nat.zero_lt_succ H⟩)).prodMk (stackTail_measurable H a))

/-- Finite adaptive Gaussian sampling has the same recommendation law under
calendar arrays and per-arm pull stacks. This is a theorem about the actual
execution maps, not an assumed equivalence of sampling conventions. -/
theorem stackRun_law_eq_run {H : ℕ} (s : Strategy (A := A) H)
    (μ : Means A) (hs : Regular s) (hstack : StackRegular s) :
    (fixedLaw H μ).map (stackRun s) = (fixedLaw H μ).map (run s) := by
  induction H with
  | zero => cases s; rfl
  | succ H ih =>
    cases s with
    | sample a next =>
      have hsplit := selected_stack_split_preserving H μ a
      have hcal := selected_calendar_split_preserving H μ a
      have hsm : (fixedLaw (H + 1) μ).map (stackRun (.sample a next)) =
          ((gaussianReal (μ a) 1).prod (fixedLaw H μ)).map
            (fun p : ℝ × RawStream A H => stackRun (next p.1) p.2) := by
        calc
          _ = ((fixedLaw (H + 1) μ).map
              (fun r => (r a ⟨0, by omega⟩, stackTail a r))).map
              (fun p : ℝ × RawStream A H => stackRun (next p.1) p.2) :=
            (Measure.map_map hstack.2 hsplit.measurable).symm
          _ = _ := by rw [hsplit.map_eq]
      have hcm : (fixedLaw (H + 1) μ).map (run (.sample a next)) =
          ((gaussianReal (μ a) 1).prod (fixedLaw H μ)).map
            (fun p : ℝ × RawStream A H => run (next p.1) p.2) := by
        calc
          _ = ((fixedLaw (H + 1) μ).map
              (fun r => (r a ⟨0, by omega⟩, tail r))).map
              (fun p : ℝ × RawStream A H => run (next p.1) p.2) :=
            (Measure.map_map hs.2 hcal.measurable).symm
          _ = _ := by rw [hcal.map_eq]
      rw [hsm, hcm]
      apply Measure.ext
      intro B hB
      rw [Measure.map_apply hstack.2 hB, Measure.map_apply hs.2 hB,
        Measure.prod_apply (hstack.2 hB), Measure.prod_apply (hs.2 hB)]
      apply lintegral_congr
      intro y
      have hy := congrArg (fun ν : Measure A => ν B)
        (ih (next y) (hs.1 y) (hstack.1 y))
      simpa only [Measure.map_apply (stackRegular_run_measurable _ (hstack.1 y)) hB,
        Measure.map_apply (regular_run_measurable _ (hs.1 y)) hB] using hy

end Measurable

open Draft0.BayesianModel Junpei.Policy Junpei.PolicyContinuation

theorem rawTree_stackRun_measurable (π : StatePolicy) (t H : ℕ) (c : Counts) :
    Measurable (fun p : Input H => stackRun (rawTree π t H c p.1 p.2.1) p.2.2) := by
  induction H generalizing t c with
  | zero =>
      exact π.recommend_measurable.comp
        (measurable_const.prodMk (measurable_fst.prodMk
          ((measurable_pi_apply (⟨0, by omega⟩ : Fin 1)).comp
            (measurable_fst.comp measurable_snd))))
  | succ H ih =>
      let g : Input (H + 1) × Arm → Arm := fun p =>
        stackRun (rawTree π (t + 1) H (Draft0.BayesianModel.advance c p.2)
          (Draft0.BayesianModel.shift p.1.1 p.2
            ((p.1.2.2 p.2 ⟨0, by omega⟩ - p.1.1 p.2) / ((c p.2 : ℝ) + 1)))
          (seedTail p.1.2.1)) (stackTail p.2 p.1.2.2)
      have hg : Measurable g := by
        apply measurable_from_prod_countable_left
        intro a
        have hin : Measurable (fun p : Input (H + 1) =>
            (Draft0.BayesianModel.shift p.1 a
              ((p.2.2 a ⟨0, by omega⟩ - p.1 a) / ((c a : ℝ) + 1)),
                (seedTail p.2.1, stackTail a p.2.2))) := by
          apply Measurable.prodMk
          · apply measurable_pi_lambda
            intro b
            simp only [Draft0.BayesianModel.shift]
            split_ifs <;> fun_prop
          · have hseed : Measurable (fun p : Input (H + 1) => seedTail p.2.1) := by
              unfold seedTail
              fun_prop
            exact hseed.prodMk
              ((stackTail_measurable H a).comp (measurable_snd.comp measurable_snd))
        exact (ih (t + 1) (Draft0.BayesianModel.advance c a)).comp hin
      have hdraw : Measurable (fun p : Input (H + 1) =>
          π.draw t c p.1 (p.2.1 ⟨0, by omega⟩)) :=
        (π.draw_measurable t).comp
          (measurable_const.prodMk (measurable_fst.prodMk
            ((measurable_pi_apply (⟨0, by omega⟩ : Fin (H + 2))).comp
              (measurable_fst.comp measurable_snd))))
      exact hg.comp (measurable_id.prodMk hdraw)

theorem rawTree_stackRegular (π : StatePolicy) (t H : ℕ)
    (c : Counts) (m : Draft0.BayesianModel.Means) (u : Seeds H) :
    StackRegular (rawTree π t H c m u) := by
  induction H generalizing t c m with
  | zero => trivial
  | succ H ih =>
    constructor
    · intro y
      exact ih (t + 1) _ _ (seedTail u)
    · let a := π.draw t c m (u ⟨0, by omega⟩)
      have hin : Measurable (fun p : ℝ × RawStream Arm H =>
          (Draft0.BayesianModel.shift m a ((p.1 - m a) / ((c a : ℝ) + 1)),
            (seedTail u, p.2))) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro b
          simp only [Draft0.BayesianModel.shift]
          split_ifs <;> fun_prop
        · exact measurable_const.prodMk measurable_snd
      exact (rawTree_stackRun_measurable π (t + 1) H
        (Draft0.BayesianModel.advance c a)).comp hin

theorem rawTree_stack_law_eq (π : StatePolicy) (t H : ℕ)
    (c : Counts) (m μ : Draft0.BayesianModel.Means) (u : Seeds H) :
    (fixedLaw H μ).map (stackRun (rawTree π t H c m u)) =
      (fixedLaw H μ).map (run (rawTree π t H c m u)) :=
  stackRun_law_eq_run _ μ (rawTree_regular π t H c m u)
    (rawTree_stackRegular π t H c m u)

end
end Junpei.FixedGaussian
