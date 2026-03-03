# Quick Citation Guide for Thesis Methods Section

## Template for Your Methods Section

Copy and adapt this text for your thesis:

---

### 4.X Convergence Analysis

**Statistical Approach**

We assessed convergence using a paired t-test approach \cite{box2005statistics} to compare model performance between consecutive communication rounds within each independent experimental run. This paired design accounts for within-run correlation \cite{bland1995calculating} and provides greater statistical power than independent tests. With six independent runs (N=6), we used the t-distribution with five degrees of freedom \cite{student1908probable}, which is appropriate for small sample sizes in machine learning experiments \cite{henderson2018deep, bouthillier2019unreproducible}.

**Practical Significance Threshold**

Following established principles in applied statistics \cite{sullivan2012effect} and machine learning research \cite{kirk1996practical}, we defined convergence based on *practical significance* rather than purely statistical significance. Specifically, we considered the model converged when round-to-round improvements fell below 0.005 (0.5% change in C-index). This threshold is consistent with clinical and epidemiological standards for survival models, where C-index improvements below 0.01 are generally considered negligible \cite{pencina2011extensions, harrington1982class}.

The use of a practical significance threshold is particularly important in federated learning, where large numbers of data points across multiple rounds can lead to detection of statistically significant but practically meaningless improvements \cite{li2020federated}.

**Sustained Stability Criterion**

To avoid declaring convergence prematurely due to temporary plateaus—a known challenge in federated optimization \cite{reddi2020adaptive}—we required three consecutive rounds of stable behavior before declaring convergence. This "patience" parameter of k=3 is consistent with established early stopping practices in neural network training \cite{prechelt1998early} and accounts for the stochastic nature of federated learning algorithms \cite{khaled2020tighter}.

**Federated-Specific Metrics**

Unlike centralized training, federated learning requires monitoring multiple convergence indicators due to data heterogeneity across clients \cite{kairouz2021advances}. Following best practices in federated learning research, we tracked three distinct metrics:

1. **Global Average Performance**: The mean model performance averaged across all clients, representing overall system effectiveness.

2. **Worst-Case Client Performance**: The minimum performance across all clients in each round. This metric ensures fair treatment of all participants \cite{li2019fair} and aligns with the min-max fairness objective in federated systems \cite{mohri2019agnostic}.

3. **Inter-Client Heterogeneity**: The standard deviation of performance metrics across clients, quantifying the degree of data heterogeneity \cite{zhao2018federated} and model consensus \cite{kairouz2021advances}.

**Oscillation Patterns as Convergence Signal**

In addition to monotonic convergence, we identified stable oscillation—small fluctuations around zero improvement—as an alternative convergence signal. This behavior is theoretically predicted in federated algorithms due to client sampling variance and heterogeneous data distributions \cite{khaled2020tighter, haddadpour2019convergence, woodworth2020local}. We defined stable oscillation as occurring when the 95% confidence interval for round-to-round improvement includes zero, the absolute mean change is below the practical threshold (0.005), and this pattern persists for three consecutive rounds.

**Implementation Details**

For each metric (global average, worst-case, and heterogeneity), we computed:
- Paired differences between consecutive rounds within each run
- Mean difference and standard error across the six independent runs
- 95% confidence intervals using the t-distribution (df=5)
- Convergence flags based on the criteria described above

This comprehensive approach provides robust convergence detection while accounting for the unique challenges of federated learning: communication constraints, client heterogeneity, and non-IID data distributions \cite{li2020federated, yang2019federated}.

---

## Inline Citation Examples

Here are specific sentences you can use:

### For Paired t-tests:
> "We used paired t-tests to account for within-run correlation between consecutive rounds \cite{bland1995calculating}."

### For Practical Significance:
> "Statistical significance does not imply practical importance \cite{sullivan2012effect}, particularly with large sample sizes across many rounds."

> "In survival analysis, C-index improvements below 0.01 are considered clinically negligible \cite{pencina2011extensions}."

### For k=3 Consecutive Rounds:
> "Following early stopping best practices \cite{prechelt1998early}, we required three consecutive stable rounds to avoid false convergence due to temporary plateaus \cite{reddi2020adaptive}."

### For Worst-Case Metric:
> "Fairness in federated learning requires ensuring no client is left behind \cite{li2019fair}, necessitating tracking of worst-case performance \cite{mohri2019agnostic}."

### For Heterogeneity:
> "Variance across clients quantifies data heterogeneity, a key challenge in federated settings \cite{zhao2018federated}."

### For Oscillation:
> "Federated algorithms naturally exhibit oscillatory convergence due to client sampling and local updates \cite{khaled2020tighter}."

### For Sample Size (N=6):
> "Recent ML reproducibility studies recommend 5-10 independent runs for robust statistical inference \cite{henderson2018deep, bouthillier2019unreproducible}."

---

## Key Points to Emphasize

1. **Why paired tests?** Because same experimental setup (same seed) across rounds → correlated measurements

2. **Why 0.5% threshold?** Because:
   - Survival analysis standard (cite Pencina)
   - Practical vs statistical significance (cite Sullivan)
   - FL has inherent noise (cite Li)

3. **Why k=3 rounds?** Because:
   - Early stopping best practice (cite Prechelt)
   - FL has temporary plateaus (cite Reddi)
   - Reduces false positives

4. **Why three metrics?** Because:
   - Fairness concerns (cite Li, Mohri)
   - Heterogeneity matters (cite Zhao)
   - Triangulation strengthens conclusions

5. **Why accept oscillation?** Because:
   - Theoretically predicted (cite Khaled)
   - Natural FL behavior (cite Haddadpour)
   - Still indicates convergence region

---

## Response to Potential Reviewer Questions

**Q: "Why not just use p < 0.05?"**
A: Statistical significance ≠ practical significance (Sullivan & Feinn, 2012). With 240 data points (6 runs × 40 rounds), we could detect tiny, meaningless improvements. Our 0.5% threshold is based on clinical standards in survival analysis (Pencina et al., 2011).

**Q: "Why only 6 runs? Shouldn't you have more?"**
A: Recent ML reproducibility research shows 5-10 runs is sufficient and practical (Henderson et al., 2018; Bouthillier et al., 2019). We chose 6 as a middle ground. Our use of the t-distribution appropriately accounts for the small sample size.

**Q: "Why track worst-case performance?"**
A: Fairness is critical in FL (Li et al., 2019). Optimizing only average performance can leave some clients with poor models (Mohri et al., 2019). Worst-case tracking ensures equitable treatment.

**Q: "Your model oscillates—doesn't that mean it hasn't converged?"**
A: No. Oscillation near the optimum is theoretically expected in FL due to client heterogeneity and local updates (Khaled et al., 2020). Stable oscillation with small magnitude indicates convergence to a neighborhood around the optimum (Haddadpour & Mahdavi, 2019).

**Q: "Why 3 consecutive rounds specifically?"**
A: This "patience" parameter is standard in early stopping (Prechelt, 1998) and prevents false convergence from temporary plateaus (Reddi et al., 2020). We chose 3 as a balance between speed and confidence.

---

## Where to Put These Citations in Your LaTeX

```latex
\section{Convergence Analysis}

We assessed convergence using a paired t-test approach~\cite{box2005statistics, bland1995calculating}
to compare consecutive rounds...

[Continue with the template text above, inserting \cite{} commands as shown]

\bibliography{references}
```

Make sure to copy the `references.bib` file to your LaTeX project directory!

---

## Additional Tips

1. **Don't over-cite**: Use 1-2 key citations per claim, not all available references

2. **Primary sources**: Use original papers (e.g., McMahan 2017 for FedAvg) when possible

3. **Recent reviews**: Kairouz et al. (2021) is a comprehensive FL survey—cite it for general FL claims

4. **Balance**: Mix statistical theory (Box, Student) with ML-specific papers (Henderson, Prechelt)

5. **Domain relevance**: Include survival analysis papers (Pencina) to show domain awareness

---

## Final Checklist

Before submitting your thesis, verify you've cited:

- [ ] Why paired tests (Box or Bland)
- [ ] Practical vs statistical significance (Sullivan)
- [ ] Early stopping / patience (Prechelt)
- [ ] FL convergence theory (McMahan or Kairouz)
- [ ] Fairness / worst-case (Li 2019 or Mohri)
- [ ] Heterogeneity (Zhao)
- [ ] Oscillation behavior (Khaled or Haddadpour)
- [ ] Sample size justification (Henderson or Bouthillier)
- [ ] Domain-specific threshold (Pencina for survival)

All citations are in the `references.bib` file!








