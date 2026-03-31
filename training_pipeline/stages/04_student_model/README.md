# 04 Student Model

Goal:

- Build the deployable policy architecture.

Build here:

- depth encoder
- state MLP
- recurrent fusion block
- action head
- optional auxiliary perception heads

Recommended first version:

- CNN over depth
- MLP over the 141-d state
- GRU over fused features

Done when:

- the model can consume only deployable observations
- the model can optionally emit auxiliary perception predictions during training
