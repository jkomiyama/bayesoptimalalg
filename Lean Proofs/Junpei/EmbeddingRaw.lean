import Junpei.EmbeddingDirect
import Junpei.FiniteModelMeasurable

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.EmbeddingRaw

open FiniteGaussian FinitePosterior FiniteModelExclusion EmbeddingDirect FiniteModelMeasurable

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

abbrev RawStream (A : Type*) (H : ℕ) := A → Fin H → ℝ

def tail {H : ℕ} (s : RawStream A (H + 1)) : RawStream A H := fun a t => s a t.succ

/-- Execute a finite adaptive observation tree on calendar-indexed reward streams.
Each sample consumes exactly one calendar round. The next tree sees only the
reward of the arm selected at that round. -/
def run : {H : ℕ} → Strategy (A := A) H → RawStream A H → A
  | 0, .stop a, _ => a
  | H + 1, .sample a next, s => run (next (s a ⟨0, Nat.zero_lt_succ H⟩)) (tail s)

def trace : {H : ℕ} → Strategy (A := A) H → RawStream A H → List A
  | 0, .stop _, _ => []
  | H + 1, .sample a next, s => a :: trace (next (s a ⟨0, Nat.zero_lt_succ H⟩)) (tail s)

omit [Fintype A] [Nonempty A] [DecidableEq A] in
theorem trace_budget {H : ℕ} (s : Strategy (A := A) H) (r : RawStream A H) :
    (trace s r).length = H := by
  induction H with
  | zero => cases s; rfl
  | succ H ih => cases s with | sample a next => simp only [trace, List.length_cons, ih]

/-- Convert the innovation-coordinate strategy used by the Bellman recursion
to a strategy whose continuation receives the actual raw reward. -/
def rawOfInnovation : {H : ℕ} → Strategy (A := A) H → Precisions A → Means A →
    Strategy (A := A) H
  | 0, .stop a, _, _ => .stop a
  | _ + 1, .sample a next, c, m =>
      .sample a (fun y =>
        let z := (y - m a) / ((c a : ℝ) + 1)
        rawOfInnovation (next z) (advance c a) (shift m a z))

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

def insert {H : ℕ} (real : RawStream (Retained i) H) (imaginary : Fin H → ℝ) :
    RawStream A H :=
  fun a t => if h : a = i then imaginary t else real ⟨a, h⟩ t

omit [Fintype A] [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem insert_retained {H : ℕ} (real : RawStream (Retained i) H)
    (imaginary : Fin H → ℝ) (a : Retained i) (t : Fin H) :
    insert i real imaginary a t = real a t := by simp [insert, a.property]

omit [Fintype A] [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem insert_removed {H : ℕ} (real : RawStream (Retained i) H)
    (imaginary : Fin H → ℝ) (t : Fin H) :
    insert i real imaginary i t = imaginary t := by simp [insert]

omit [Fintype A] [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem tail_insert {H : ℕ} (real : RawStream (Retained i) (H + 1))
    (imaginary : Fin (H + 1) → ℝ) :
    tail (insert i real imaginary) = insert i (tail real) (fun t => imaginary t.succ) := rfl

theorem embed_run {H : ℕ} (s : Strategy (A := Retained i) H)
    (real : RawStream (Retained i) H) (imaginary : Fin H → ℝ) :
    run (embed i s) (insert i real imaginary) = (run (A := Retained i) s real : A) := by
  induction H with
  | zero => cases s; rfl
  | succ H ih =>
    cases s with
    | sample a next =>
      simp only [embed, run, insert_retained, tail_insert]
      exact ih (next _) _ _

/-- Literal second embedding of Lemma 13. A request for the removed arm
consumes one real observation from `dummy` and discards it. Its continuation
instead receives the independent imaginary reward. Retained observations
update the internal retained posterior. At the terminal node a removed-arm
recommendation is replaced by a posterior-best retained arm. -/
def simulate (dummy : Retained i) : {H : ℕ} → Strategy (A := A) H →
    Precisions (Retained i) → Means (Retained i) → (Fin H → ℝ) → Strategy (A := Retained i) H
  | 0, .stop a, _, m, _ =>
      if h : a = i then .stop (firstMax m) else .stop ⟨a, h⟩
  | H + 1, .sample a next, c, m, imaginary =>
      if h : a = i then
        .sample dummy (fun _ => simulate dummy
          (next (imaginary ⟨0, Nat.zero_lt_succ H⟩)) c m (fun t => imaginary t.succ))
      else
        .sample ⟨a, h⟩ (fun y =>
          let b : Retained i := ⟨a, h⟩
          let z := (y - m b) / ((c b : ℝ) + 1)
          simulate dummy (next y) (advance c b) (shift m b z) (fun t => imaginary t.succ))

theorem simulate_terminal_best (dummy : Retained i) (c : Precisions (Retained i))
    (m : Means (Retained i)) :
    simulate i dummy (.stop i) c m Fin.elim0 = .stop (firstMax m) := by
  simp [simulate]

theorem simulate_terminal_best_mean (m : Means (Retained i)) :
    m (firstMax m) = maxAll m := firstMax_attains m

theorem simulate_forced_retained (dummy : Retained i) {H : ℕ}
    (a : Retained i) (next : ℝ → Strategy (A := A) H)
    (c : Precisions (Retained i)) (m : Means (Retained i))
    (imaginary : Fin (H + 1) → ℝ) :
    simulate i dummy (.sample a next) c m imaginary =
      .sample a (fun y => simulate i dummy (next y) (advance c a)
        (shift m a ((y - m a) / ((c a : ℝ) + 1))) (fun t => imaginary t.succ)) := by
  simp [simulate, a.property]

theorem simulate_removed_discards_real (dummy : Retained i) {H : ℕ}
    (next : ℝ → Strategy (A := A) H)
    (c : Precisions (Retained i)) (m : Means (Retained i))
    (imaginary : Fin (H + 1) → ℝ) :
    simulate i dummy (.sample i next) c m imaginary =
      .sample dummy (fun _ => simulate i dummy
        (next (imaginary ⟨0, Nat.zero_lt_succ H⟩)) c m (fun t => imaginary t.succ)) := by
  simp [simulate]

theorem simulate_budget (dummy : Retained i) {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions (Retained i)) (m : Means (Retained i)) (imaginary : Fin H → ℝ)
    (real : RawStream (Retained i) H) :
    (trace (A := Retained i) (simulate i dummy s c m imaginary) real).length = H :=
  trace_budget _ _

/-- Whenever the internal full policy recommends a retained arm, replay
keeps exactly that recommendation, pathwise and for every reward array. -/
theorem simulate_run_agrees (dummy : Retained i) {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions (Retained i)) (m : Means (Retained i)) (imaginary : Fin H → ℝ)
    (real : RawStream (Retained i) H)
    (hne : run s (insert i real imaginary) ≠ i) :
    (run (A := Retained i) (simulate i dummy s c m imaginary) real : A) =
      run s (insert i real imaginary) := by
  induction H generalizing c m with
  | zero =>
    cases s with
    | stop a => simp only [run] at hne; simp [simulate, hne, run]
  | succ H ih =>
    cases s with
    | sample a next =>
      by_cases ha : a = i
      · subst a
        simp only [run, insert_removed, tail_insert] at hne
        simp only [simulate, dite_true, run, insert_removed, tail_insert]
        exact ih (next _) c m _ _ hne
      · simp only [run, insert, dif_neg ha, tail_insert] at hne
        simp only [simulate, run, insert, dif_neg ha, tail_insert]
        exact ih (next _) _ _ _ _ hne

/-- Pointwise comparison underlying the second embedding. It needs no
distributional assumptions: any replacement by a retained arm has regret
increase bounded by the removed arm's positive advantage over the minimum
retained latent mean. -/
theorem simulate_regret_bound (dummy : Retained i) {H : ℕ} (s : Strategy (A := A) H)
    (c : Precisions (Retained i)) (m : Means (Retained i)) (imaginary : Fin H → ℝ)
    (real : RawStream (Retained i) H) (u : Means A) :
    otherMaximum i u - u (run (A := Retained i) (simulate i dummy s c m imaginary) real) ≤
      maxAll u - u (run s (insert i real imaginary)) + max (u i - otherMinimum i u) 0 := by
  have hmax : otherMaximum i u ≤ maxAll u :=
    maxAll_le _ _ (fun a => coordinate_le_maxAll u a)
  have hpos := le_max_right (u i - otherMinimum i u) 0
  by_cases h : run s (insert i real imaginary) = i
  · rw [h]
    have hmin := otherMinimum_le i u
      (run (A := Retained i) (simulate i dummy s c m imaginary) real)
    have hp := le_max_left (u i - otherMinimum i u) 0
    linarith
  · rw [simulate_run_agrees i dummy s c m imaginary real h]
    linarith

end Removed
end
end Junpei.EmbeddingRaw
