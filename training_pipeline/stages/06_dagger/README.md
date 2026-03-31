# 06 DAgger

Goal:

- Fix covariate shift by labeling the states the student actually visits.

Build here:

- rollout loop with the student in control
- expert relabeling on student-visited states
- dataset merge policy
- retraining schedule

Done when:

- the student recovers from its own drift better than plain behavior cloning
- validation improves on cluttered and moving-goal maps
