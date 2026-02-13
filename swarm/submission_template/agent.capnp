@0xa5d4b3c2e1f09876;

interface Agent {
  ping @0 (message :Text) -> (response :Text);
  act  @1 (obs :Observation) -> (action :Tensor);
  reset @2 ();
  calibrate @3 (obs :Observation) -> (action :Tensor, benchmarkNs :Int64);
}

struct Tensor {
  data  @0 :Data;
  shape @1 :List(Int32);
  dtype @2 :Text;
}

struct ObservationEntry {
  key @0 :Text;
  tensor @1 :Tensor;
}

struct Observation {
  entries @0 :List(ObservationEntry);
}
