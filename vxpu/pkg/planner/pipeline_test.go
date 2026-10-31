// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package planner

import (
	"math"
	"os"
	"testing"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
)

func deepseekLayers(t *testing.T) LayerBytes {
	t.Helper()
	data, err := os.ReadFile("testdata/deepseek-v41-flash-layers.json")
	if err != nil {
		t.Fatal(err)
	}
	lb, err := ReadLayerBytesJSON(data)
	if err != nil {
		t.Fatal(err)
	}
	return lb
}

func hero8Plan(t *testing.T, batches int) *kernelplan.PipelinePlan {
	t.Helper()
	ranges, err := kernelplan.ParseRanges("0:8,8:20,20:32,32:43")
	if err != nil {
		t.Fatal(err)
	}
	return &kernelplan.PipelinePlan{Format: kernelplan.PlanFormat, Ranges: ranges, Experts: "with-layers",
		Pipeline: kernelplan.Pipeline{Batches: batches, Schedule: kernelplan.ScheduleRangeOrder}}
}

func within(t *testing.T, what string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want)/want > tolerance {
		t.Errorf("%s: predicted %.1f, measured %.1f: off by %.0f%%, tolerance %.0f%%", what, got, want, math.Abs(got-want)/want*100, tolerance*100)
	}
}

// Hero run 8's single-batch table (four RTX PRO 6000, one GPU at a time,
// Oct 31 2026): the calibration and its error.
func TestPredictPipeline_CalibrationOnHero8(t *testing.T) {
	lb := deepseekLayers(t)
	p := hero8Plan(t, 1)
	cases := []struct {
		name     string
		rows     int
		distinct float64
		measured float64
		tol      float64
	}{
		{"one row", 1, 6, 13.6, 0.03},
		{"9 rows, same prompt", 9, 6, 18.5, 0.03},
		{"33 rows, same prompt", 33, 6, 32.8, 0.03},
		{"9 rows, a prompt per row (37.3 distinct experts)", 9, 37.3, 35.6, 0.06},
		{"33 rows, a prompt per row (100 distinct experts)", 33, 100, 99.2, 0.12},
		{"33 rows, a prompt per row, the profiled run (112 distinct)", 33, 112, 99.2, 0.06},
	}
	for _, c := range cases {
		pr, err := PredictPipeline(lb, p, Workload{Rows: c.rows, DistinctExperts: c.distinct, Batches: 1}, RTXPro6000)
		if err != nil {
			t.Fatal(err)
		}
		within(t, c.name, pr.SingleBatchMs, c.measured, c.tol)
		t.Logf("%-58s predicted %.1f ms, measured %.1f (ranges %.1f %.1f %.1f %.1f)", c.name, pr.SingleBatchMs, c.measured,
			pr.Ranges[0].Ms, pr.Ranges[1].Ms, pr.Ranges[2].Ms, pr.Ranges[3].Ms)
	}
}

// Hero run 9's acceptance cases, predicted before the pipeline exists: the
// numbers the run is judged against (design.md, "Hero run 9").
func TestPredictPipeline_Hero9Predictions(t *testing.T) {
	lb := deepseekLayers(t)
	p := hero8Plan(t, 4)
	cases := []struct {
		name     string
		w        Workload
		minTokS  float64 // what we would call a failure to understand
		maxLatMs float64
	}{
		{"four batches of one row", Workload{Rows: 1, DistinctExperts: 6, Batches: 4}, 200, 17},
		{"four batches of 33 rows, same prompt", Workload{Rows: 33, DistinctExperts: 6, Batches: 4}, 2500, 50},
		{"four batches of 33 rows, a prompt per row", Workload{Rows: 33, DistinctExperts: 100, Batches: 4}, 1000, 130},
	}
	for _, c := range cases {
		pr, err := PredictPipeline(lb, p, c.w, RTXPro6000)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%-44s %.0f tok/s aggregate, period %.1f ms, latency %.1f ms (single %.1f), floor %.0f tok/s at the slowest range's bytes; bottleneck %s; %s",
			c.name, pr.AggregateTokS, pr.PeriodMs, pr.LatencyMs, pr.SingleBatchMs, pr.FloorTokS, pr.Bottleneck, pr.BalanceWarning)
		if pr.AggregateTokS < c.minTokS {
			t.Errorf("%s: %.0f tok/s predicted, the model itself says the run cannot reach %.0f", c.name, pr.AggregateTokS, c.minTokS)
		}
		if pr.LatencyMs > c.maxLatMs {
			t.Errorf("%s: latency %.1f ms predicted, over the %.0f ms bar", c.name, pr.LatencyMs, c.maxLatMs)
		}
		if pr.AggregateTokS > pr.FloorTokS*1.001 {
			t.Errorf("%s: predicted %.0f tok/s above the floor %.0f", c.name, pr.AggregateTokS, pr.FloorTokS)
		}
	}
}

func TestBalance_CutsWhereTheStepMayBeCut(t *testing.T) {
	lb := deepseekLayers(t)
	p := hero8Plan(t, 4)
	// the layers a range may start at: the KV and index source layers (design.md, hero 7)
	cuts := []int{2, 8, 14, 20, 24, 28, 32, 36}
	for _, w := range []Workload{{Rows: 1, DistinctExperts: 6, Batches: 4}, {Rows: 33, DistinctExperts: 100, Batches: 4}} {
		best, pred, err := Balance(lb, p, w, RTXPro6000, cuts, 92) // a 96 GB card: the 12-layer ranges hold 89 GB today
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("rows %d: best ranges %s, period %.1f ms, %.0f tok/s; held GB per range %v", w.Rows, rangesText(best), pred.PeriodMs, pred.AggregateTokS,
			func() []float64 {
				var h []float64
				for _, rc := range pred.Ranges {
					h = append(h, math.Round(rc.HeldGB))
				}
				return h
			}())
		for _, rc := range pred.Ranges {
			if rc.HeldGB > 92 {
				t.Errorf("rows %d: range %s holds %.0f GB, over the 92 GB budget", w.Rows, rc.Range.Layers, rc.HeldGB)
			}
		}
		// the cut points (every range must start at a KV-source layer) leave no 4-GPU split with a
		// shorter slowest range than the current 8/12/12/11 within memory: the balancer keeps the plan
		if rangesText(best) != rangesText(p) {
			t.Errorf("rows %d: the balancer moved the ranges to %s; expected the current ones to stand under the cut constraint", w.Rows, rangesText(best))
		}
	}
	// without the cut constraint the balancer finds something at least as good
	freePlan, free, err := Balance(lb, p, Workload{Rows: 33, DistinctExperts: 100, Batches: 4}, RTXPro6000, nil, 92)
	if err != nil {
		t.Fatal(err)
	}
	_, constrained, _ := Balance(lb, p, Workload{Rows: 33, DistinctExperts: 100, Batches: 4}, RTXPro6000, cuts, 92)
	t.Logf("free cuts: %s, period %.1f ms, %.0f tok/s (constrained %.1f ms)", rangesText(freePlan), free.PeriodMs, free.AggregateTokS, constrained.PeriodMs)
	if free.PeriodMs > constrained.PeriodMs*1.001 {
		t.Errorf("free cuts gave a period of %.1f ms, worse than the constrained %.1f", free.PeriodMs, constrained.PeriodMs)
	}
}

func TestReadLayerBytes_FromABinding(t *testing.T) {
	manifest := `{"tensors": {"e0.w1": {"length": 100}, "e1.w1": {"length": 100}, "e0.w2": {"length": 50}, "e1.w2": {"length": 50}},
	              "config": {"num_experts_per_tok": 1}}`
	binding := `{"bound": {"model.layers.0.attn.w": {"length": 1000}, "model.layers.1.attn.w": {"length": 2000}, "model.layers.1.attn.cache": {"length": 999},
	                        "model.mtp.0.head.weight": {"length": 5000}, "model.mtp.0.embed.weight": {"length": 7000}, "model.mtp.0.attn.w": {"length": 300}},
	              "state": ["model.layers.1.attn.cache"],
	              "external": {"experts": {"1": {"w1": {"parts": [["e0.w1"], ["e1.w1"]]}, "w2": {"parts": [["e0.w2"], ["e1.w2"]]}}}}}`
	lb, err := ReadLayerBytes([]byte(manifest), []byte(binding))
	if err != nil {
		t.Fatal(err)
	}
	if lb.HeadBytes != 5000 || lb.DrafterDenseBytes != 300 || lb.TopK != 1 || len(lb.Layers) != 2 {
		t.Fatalf("inventory: %+v", lb)
	}
	if lb.Layers[0].DenseBytes != 1000 || lb.Layers[1].DenseBytes != 2000 || lb.Layers[1].Experts != 2 || lb.Layers[1].BytesPerExpert != 150 {
		t.Fatalf("layers: %+v", lb.Layers)
	}
}
