
**Source: an original simulation, not real fitness application records.**
The observation unit is one unique user at the end of a simulated week.
All inputs precede the following week. Each user appears once.
Three overlapping engagement regimes are used only inside the generator;
their IDs are never included in the dataset, models, or cluster evaluation.
Workouts follow a Poisson distribution, streak follows a Gamma distribution,
and notifications follow a Poisson distribution.
Unusual combinations are introduced into 3% of rows: few workouts with a long streak,
many notifications, or high activity. These are plausible unusual observations,
not verified or labeled real-world anomalies.
Next-week activity probability is defined by a nonlinear logistic function:
`z = -2 + 0.55*workouts + 0.9*log1p(streak) + 0.07*min(notifications,10)`
` - 0.06*max(notifications-10,0) - 0.10*max(workouts-9,0)^2 + Normal(0,0.75)`.
The label is sampled from Bernoulli(sigmoid(z)): it is related to the inputs
but is not perfectly determined by them. After generating the label, feature values
are independently masked with NaN at the selected probability (MCAR, a simplification).
All distributions and relationships are educational assumptions, not empirical findings.
The OpenML link in the worksheet is a catalog, not the source of these observations.


- Metrics describe performance on simulated data and do not establish real-world performance.
- A random split suits the independent simulated users. Real logs require a temporal split and separation by user_id.
- MCAR missingness is simpler than real missingness; seasonality, drift, and social factors are absent.
- Cluster counts are exploratory; silhouette is not accuracy. Seed stability does not establish stability under resampling.
- PCA is linear; t-SNE/UMAP distort distances, group sizes, and positions. Visual separation does not prove segments exist.
- DBSCAN noise and Isolation Forest flags do not establish fraud or measurement error.
- Notification relationships reflect generator assumptions. Causal campaign effects require an A/B test.
Improvement: collect consented, anonymized logs, validate metric definitions,
use external temporal evaluation, assess probability calibration, and test retention campaigns experimentally.
