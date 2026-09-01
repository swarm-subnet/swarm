# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
