import Junpei.Policy
import Junpei.EmbeddingExecution

open MeasureTheory ProbabilityTheory

namespace Junpei.PolicyContinuation

open Draft0.BayesianModel Junpei.FiniteGaussian Junpei.EmbeddingRaw
open Junpei.EmbeddingExecution

noncomputable section

abbrev Seeds (H : ℕ) := Fin (H + 1) → ℝ

def seedTail {H : ℕ} (u : Seeds (H + 1)) : Seeds H := fun j => u j.succ

/-- The actual raw-observation continuation of a posterior-state policy.
The final seed is used for the terminal recommendation. -/
def rawTree (π : Junpei.Policy.StatePolicy) (t : ℕ) : (H : ℕ) →
    Counts → Draft0.BayesianModel.Means → Seeds H →
    Junpei.FiniteGaussian.Strategy (A := Arm) H
  | 0, c, m, u => .stop (π.recommend c m (u ⟨0, by omega⟩))
  | H + 1, c, m, u =>
      let a := π.draw t c m (u ⟨0, by omega⟩)
      .sample a (fun y =>
        let z := (y - m a) / ((c a : ℝ) + 1)
        rawTree π (t + 1) H (Draft0.BayesianModel.advance c a)
          (Draft0.BayesianModel.shift m a z) (seedTail u))

theorem rawTree_zero (π : Junpei.Policy.StatePolicy) (t : ℕ)
    (c : Counts) (m : Draft0.BayesianModel.Means) (u : Seeds 0) :
    rawTree π t 0 c m u = .stop (π.recommend c m (u ⟨0, by omega⟩)) := rfl

theorem rawTree_succ (π : Junpei.Policy.StatePolicy) (t H : ℕ)
    (c : Counts) (m : Draft0.BayesianModel.Means) (u : Seeds (H + 1)) :
    rawTree π t (H + 1) c m u =
      .sample (π.draw t c m (u ⟨0, by omega⟩)) (fun y =>
        let a := π.draw t c m (u ⟨0, by omega⟩)
        let z := (y - m a) / ((c a : ℝ) + 1)
        rawTree π (t + 1) H (Draft0.BayesianModel.advance c a)
          (Draft0.BayesianModel.shift m a z) (seedTail u)) := rfl

abbrev Input (H : ℕ) := Draft0.BayesianModel.Means × (Seeds H × RawStream Arm H)

/-- Joint Borel measurability includes the starting means, every independent
policy seed, and every raw reward coordinate. -/
theorem rawTree_run_measurable (π : Junpei.Policy.StatePolicy) (t H : ℕ) (c : Counts) :
    Measurable (fun p : Input H => run (rawTree π t H c p.1 p.2.1) p.2.2) := by
  induction H generalizing t c with
  | zero =>
      have hin : Measurable (fun p : Input 0 =>
          (c, p.1, p.2.1 (⟨0, by omega⟩ : Fin 1))) :=
        measurable_const.prodMk (measurable_fst.prodMk
          ((measurable_pi_apply (⟨0, by omega⟩ : Fin 1)).comp
            (measurable_fst.comp measurable_snd)))
      exact π.recommend_measurable.comp hin
  | succ H ih =>
      let g : Input (H + 1) × Arm → Arm := fun p =>
        run (rawTree π (t + 1) H (Draft0.BayesianModel.advance c p.2)
          (Draft0.BayesianModel.shift p.1.1 p.2
            ((p.1.2.2 p.2 ⟨0, by omega⟩ - p.1.1 p.2) / ((c p.2 : ℝ) + 1)))
          (seedTail p.1.2.1)) (tail p.1.2.2)
      have hg : Measurable g := by
        apply measurable_from_prod_countable_left
        intro a
        have hin : Measurable (fun p : Input (H + 1) =>
            (Draft0.BayesianModel.shift p.1 a
              ((p.2.2 a ⟨0, by omega⟩ - p.1 a) / ((c a : ℝ) + 1)),
                (seedTail p.2.1, tail p.2.2))) := by
          apply Measurable.prodMk
          · apply measurable_pi_lambda
            intro b
            simp only [Draft0.BayesianModel.shift]
            split_ifs <;> fun_prop
          · unfold seedTail tail
            fun_prop
        exact (ih (t + 1) (Draft0.BayesianModel.advance c a)).comp hin
      have hdraw : Measurable (fun p : Input (H + 1) =>
          π.draw t c p.1 (p.2.1 ⟨0, by omega⟩)) := by
        have hin : Measurable (fun p : Input (H + 1) =>
            (c, p.1, p.2.1 (⟨0, by omega⟩ : Fin (H + 2)))) :=
          measurable_const.prodMk (measurable_fst.prodMk
            ((measurable_pi_apply (⟨0, by omega⟩ : Fin (H + 2))).comp
              (measurable_fst.comp measurable_snd)))
        exact (π.draw_measurable t).comp hin
      exact hg.comp (measurable_id.prodMk hdraw)

/-- Every fixed-seed continuation is a regular raw-history tree; this is
proved from the policy's Borel decision maps. -/
theorem rawTree_regular (π : Junpei.Policy.StatePolicy) (t H : ℕ)
    (c : Counts) (m : Draft0.BayesianModel.Means) (u : Seeds H) :
    Regular (rawTree π t H c m u) := by
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
        exact (rawTree_run_measurable π (t + 1) H
          (Draft0.BayesianModel.advance c a)).comp hin

theorem rawTree_budget (π : Junpei.Policy.StatePolicy) (t H : ℕ)
    (c : Counts) (m : Draft0.BayesianModel.Means) (u : Seeds H) (r : RawStream Arm H) :
    (trace (rawTree π t H c m u) r).length = H := trace_budget _ _

end
end Junpei.PolicyContinuation
