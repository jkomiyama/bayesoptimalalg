import Draft0.FrequentistModel

namespace Draft0.Execution

open BayesianModel PolicyPaths FrequentistModel

noncomputable section

theorem advance_total (c : BayesianModel.Counts) (i : Arm) :
    advance c i 0 + advance c i 1 + advance c i 2 = c 0 + c 1 + c 2 + 1 := by
  fin_cases i <;> simp [advance] <;> omega

theorem afterInitial_total (T r : ℕ) (p : Path) :
    afterInitial T p r 0 + afterInitial T p r 1 + afterInitial T p r 2 = 3 + r := by
  induction r with
  | zero => rfl
  | succ r ih =>
    simp only [afterInitial, advance_total]
    omega

theorem counts_total (T s : ℕ) (p : Path) (hs : 3 ≤ s) :
    counts T p s 0 + counts T p s 1 + counts T p s 2 = s := by
  have h := afterInitial_total T (s - 3) p
  unfold counts
  omega

theorem count_before_draw_lt_budget (T t : ℕ) (p : Path) (ht : 4 ≤ t) (htT : t ≤ T)
    (i : Arm) : counts T p (t - 1) i < T := by
  have hsum := counts_total T (t - 1) p (by omega)
  fin_cases i
  · change counts T p (t - 1) 0 < T
    omega
  · change counts T p (t - 1) 1 < T
    omega
  · change counts T p (t - 1) 2 < T
    omega

theorem initial_mean (T : ℕ) (hT : 0 < T) (ω : BadEvent.RawArrays T) (i : Arm) :
    meansAt (empiricalPath ω) (counts T (empiricalPath ω) 3) i =
      observation ω i ⟨0, hT⟩ := by
  simp only [meansAt, counts_initial, empiricalPath, observation]
  rw [BadEvent.sampleMean_one hT]

theorem meansAt_advance_eq_shift {T : ℕ} (ω : BadEvent.RawArrays T)
    (c : BayesianModel.Counts) (i : Arm) (hn : 0 < c i) (hb : c i < T) :
    meansAt (empiricalPath ω) (advance c i) =
      shift (meansAt (empiricalPath ω) c) i
        (Gaussian.posteriorMeanUpdate (meansAt (empiricalPath ω) c i) (c i)
          (observation ω i ⟨c i, hb⟩) - meansAt (empiricalPath ω) c i) := by
  funext j
  by_cases hj : j = i
  · subst j
    simp only [meansAt, advance, shift, if_true]
    rw [empiricalPath_update ω i (c i) hn hb]
    ring
  · simp only [meansAt, advance, shift, if_neg hj, add_zero]

/-- Each actual posterior state follows the weighted sample update (Eq.3),
using an unused raw Gaussian reward of the arm selected by the Bayes policy. -/
theorem actual_mean_update (T t : ℕ) (ht : 4 ≤ t) (htT : t ≤ T)
    (ω : BadEvent.RawArrays T) :
    let p := empiricalPath ω
    let c := counts T p (t - 1)
    let i := selected T p t
    let oldMean := meansAt p c i
    let reward := observation ω i ⟨c i, count_before_draw_lt_budget T t p ht htT i⟩
    meansAt p (counts T p t) =
      shift (meansAt p c) i (Gaussian.posteriorMeanUpdate oldMean (c i) reward - oldMean) := by
  dsimp only
  have he : t - 1 + 1 = t := by omega
  have hu := counts_update T (empiricalPath ω) (t - 1) (by omega)
  rw [he] at hu
  rw [hu]
  exact meansAt_advance_eq_shift ω (counts T (empiricalPath ω) (t - 1))
    (selected T (empiricalPath ω) t)
    (counts_positive T (empiricalPath ω) (t - 1) (selected T (empiricalPath ω) t))
    (count_before_draw_lt_budget T t (empiricalPath ω) ht htT (selected T (empiricalPath ω) t))

end

end Draft0.Execution
