import Junpei.FiniteModelExclusion

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.EmbeddingDirect

open FiniteGaussian FinitePosterior FiniteModelExclusion

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
variable (i : A) [Nonempty {a : A // a ≠ i}]

noncomputable section

abbrev Retained := {a : A // a ≠ i}

def restrictMeans (m : Means A) : Means (Retained i) := fun a => m a
def restrictPrecisions (c : Precisions A) : Precisions (Retained i) := fun a => c a

omit [Fintype A] [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem restrict_advance (c : Precisions A) (a : Retained i) :
    restrictPrecisions i (advance c a) = advance (restrictPrecisions i c) a := by
  ext b
  simp only [restrictPrecisions, advance, Subtype.ext_iff]

omit [Fintype A] [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem restrict_shift (m : Means A) (a : Retained i) (z : ℝ) :
    restrictMeans i (shift m a z) = shift (restrictMeans i m) a z := by
  ext b
  simp only [restrictMeans, shift, Subtype.ext_iff]

/-- The first embedding in Lemma 13: neither sample nor recommend the removed arm. -/
def embed : {k : ℕ} → Strategy (A := Retained i) k → Strategy (A := A) k
  | 0, .stop a => .stop a
  | _ + 1, .sample a next => .sample a (fun z => embed (next z))

theorem embed_utility {k : ℕ} (s : Strategy (A := Retained i) k)
    (c : Precisions A) (m : Means A) :
    utility (embed i s) c m = utility s (restrictPrecisions i c) (restrictMeans i m) := by
  induction k generalizing c m with
  | zero => cases s; rfl
  | succ k ih =>
    cases s with
    | sample a next =>
      simp only [embed, utility, ih, restrict_advance, restrict_shift]
      rfl

theorem embed_admissible {k : ℕ} (s : Strategy (A := Retained i) k)
    (c : Precisions A) (m : Means A)
    (hs : Admissible s (restrictPrecisions i c) (restrictMeans i m)) :
    Admissible (embed i s) c m := by
  induction k generalizing c m with
  | zero => cases s; trivial
  | succ k ih =>
    cases s with
    | sample a next =>
      constructor
      · intro z
        apply ih
        simpa only [restrict_advance, restrict_shift] using hs.1 z
      · simp only [embed_utility, restrict_advance, restrict_shift]
        exact hs.2

def reducedPotential (c : Precisions A) (m : Means A) : ℝ :=
  potential (restrictPrecisions i c) (restrictMeans i m)

theorem reducedPotential_eq_posterior (c : Precisions A) (m : Means A) :
    reducedPotential i c m = ∫ u, otherMaximum i u ∂posteriorLaw c m := by
  rw [posterior_integral (otherMaximum_lipschitz i), ← expectation]
  rw [expectation_split (otherMaximum_lipschitz i) c m i]
  have hj (u : Retained i → ℝ) (x : ℝ) :
      otherMaximum i (m + join i (x, u)) = maxAll (fun a : Retained i => m a + u a) := by
    unfold otherMaximum
    congr 1
    ext a
    simp [Pi.add_apply, join, a.property]
  simp_rw [hj]
  unfold reducedPotential potential expectation centeredLaw otherLaw
  simp only [restrictPrecisions, integral_const, measureReal_univ_eq_one, one_smul]
  rfl

omit [Nonempty A] in
theorem otherMinimum_le (m : Means A) (a : Retained i) : otherMinimum i m ≤ m a := by
  have h := coordinate_le_maxAll (fun b : Retained i => -m b) a
  unfold otherMinimum
  linarith

theorem removed_excess_identity (u : Means A) :
    maxAll u - otherMaximum i u = max (u i - otherMaximum i u) 0 := by
  have he : maxAll u = max (otherMaximum i u) (u i) := by
    apply le_antisymm
    · apply maxAll_le
      intro a
      by_cases ha : a = i
      · subst a; exact le_max_right _ _
      · exact (coordinate_le_maxAll (fun b : Retained i => u b) ⟨a, ha⟩).trans
          (le_max_left _ _)
    · exact max_le (maxAll_le _ _ (fun a => coordinate_le_maxAll u a))
        (coordinate_le_maxAll u i)
  rw [he]
  by_cases h : u i ≤ otherMaximum i u
  · rw [max_eq_left h, max_eq_right (by linarith : u i - otherMaximum i u ≤ 0)]
    ring
  · rw [max_eq_right (le_of_not_ge h), max_eq_left (by linarith : 0 ≤ u i - otherMaximum i u)]

theorem potential_excess_eq (c : Precisions A) (m : Means A) :
    potential c m - reducedPotential i c m =
      ∫ u, max (u i - otherMaximum i u) 0 ∂posteriorLaw c m := by
  rw [potential_eq_posterior, reducedPotential_eq_posterior,
    ← integral_sub (posterior_integrable maxAll_lipschitz c m)
      (posterior_integrable (otherMaximum_lipschitz i) c m)]
  apply integral_congr_ae
  exact Filter.Eventually.of_forall (removed_excess_identity i)

theorem potential_excess_bounds (c : Precisions A) (m : Means A) :
    0 ≤ potential c m - reducedPotential i c m ∧
      potential c m - reducedPotential i c m ≤ exclusion i c m := by
  rw [potential_excess_eq]
  constructor
  · exact integral_nonneg (fun _ => le_max_right _ _)
  · apply integral_mono
      ((posterior_coordinate_integrable c m i).sub
        (posterior_integrable (otherMaximum_lipschitz i) c m)).pos_part
      (exclusion_integrable i c m)
    intro u
    obtain ⟨a⟩ := (inferInstance : Nonempty (Retained i))
    have hmin := otherMinimum_le i u a
    have hmax := coordinate_le_maxAll (fun b : Retained i => u b) a
    exact max_le_max (by dsimp [otherMaximum]; linarith) le_rfl

/-- Equality of actual posterior expected simple regrets in the first embedding.
The excess is the expectation of the removed arm's positive advantage over
the best retained arm, independent of which retained policy is embedded. -/
theorem embed_expectedRegret {k : ℕ} (s : Strategy (A := Retained i) k)
    (c : Precisions A) (m : Means A)
    (hs : Admissible s (restrictPrecisions i c) (restrictMeans i m))
    (hc : ∀ a, 0 < c a) :
    expectedRegret (embed i s) c m =
      expectedRegret s (restrictPrecisions i c) (restrictMeans i m) +
      ∫ u, max (u i - otherMaximum i u) 0 ∂posteriorLaw c m := by
  rw [expectedRegret_eq _ _ _ (embed_admissible i s c m hs) hc,
    expectedRegret_eq _ _ _ hs (fun a => hc a), embed_utility,
    ← potential_excess_eq]
  unfold reducedPotential
  ring

/-- The upper half of Lemma 13 follows by applying global optimality to
the explicitly embedded optimal retained policy. -/
theorem lemma13_first_embedding (k : ℕ) (c : Precisions A) (m : Means A)
    (_hc : ∀ a, 0 < c a) :
    loss k c m - loss k (restrictPrecisions i c) (restrictMeans i m) ≤ exclusion i c m := by
  let s := optimal k (restrictPrecisions i c) (restrictMeans i m)
  have hs : Admissible s (restrictPrecisions i c) (restrictMeans i m) := optimal_admissible _ _ _
  have ho := utility_le_value (embed i s) c m (embed_admissible i s c m hs)
  rw [embed_utility, optimal_utility] at ho
  have hp := (potential_excess_bounds i c m).2
  unfold loss reducedPotential at *
  linarith

theorem lemma13_first_embedding_forced (k : ℕ) (c : Precisions A) (m : Means A)
    (a : Retained i) (hc : ∀ b, 0 < c b) :
    actionLoss k c m a -
      actionLoss k (restrictPrecisions i c) (restrictMeans i m) a ≤ exclusion i c m := by
  rw [actionLoss_eq k c m a (hc a),
    actionLoss_eq k (restrictPrecisions i c) (restrictMeans i m) a (hc a)]
  have hu : action k (restrictPrecisions i c) (restrictMeans i m) a ≤ action k c m a := by
    unfold action
    change (∫ z, value k (advance (restrictPrecisions i c) a)
      (shift (restrictMeans i m) a z) ∂gaussianReal 0 (innovation c a)) ≤ _
    apply integral_mono (action_integrable k (restrictPrecisions i c) (restrictMeans i m) a)
      (action_integrable k c m a)
    intro z
    let s := optimal k (restrictPrecisions i (advance c a)) (restrictMeans i (shift m a z))
    have hs := embed_admissible i s (advance c a) (shift m a z) (optimal_admissible _ _ _)
    have h := utility_le_value (embed i s) (advance c a) (shift m a z) hs
    rw [embed_utility, optimal_utility, restrict_advance, restrict_shift] at h
    exact h
  have hp := (potential_excess_bounds i c m).2
  unfold reducedPotential at hp
  linarith

end
end Junpei.EmbeddingDirect
