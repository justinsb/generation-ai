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

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"github.com/gke-labs/generation-ai/vxpu/pkg/planner"
	"k8s.io/klog/v2"
)

// vxpu plan pipeline: the physical plan of a step across GPUs (vxpu-plan/1)
// and what the cost model predicts for it, from an artifact's inventory
// (hero run 9).
func cmdPlan(ctx context.Context, args []string) error {
	if len(args) == 0 || args[0] != "pipeline" {
		return fmt.Errorf("usage: vxpu plan pipeline [flags]")
	}
	return runPlanPipeline(ctx, args[1:])
}

func runPlanPipeline(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("plan pipeline", flag.ContinueOnError)
	artifact := flags.String("artifact", "", "an artifact directory (manifest.json and binding.json): the model's per-layer bytes")
	layersPath := flags.String("layers", "", "the per-layer inventory as JSON instead of an artifact (vxpu plan pipeline --layers-out writes one)")
	layersOut := flags.String("layers-out", "", "write the artifact's per-layer inventory here")
	rangesSpec := flags.String("ranges", "0:8,8:20,20:32,32:43", "the layer ranges, one per GPU in step order")
	batches := flags.Int("batches", 4, "batches in flight; 0: the planner lays --rows out as one to as many batches as ranges and picks the shape (hero run 10)")
	rows := flags.Int("rows", 1, "rows a batch; with --batches 0, the rows to serve at once")
	distinct := flags.Float64("distinct", 0, "distinct experts a layer the batch selects (default: top-k, the same prompt in every row)")
	distinctAt := flags.String("distinct-at", "", "measured distinct experts a layer by batch width, ROWS=COUNT,...: the curve --batches 0 estimates each candidate's count from (default: --distinct at every width)")
	latency := flags.Float64("latency-ms", 0, "with --batches 0, the per-batch step the chosen shape must stay under (0: no budget)")
	maxRows := flags.Int("max-batch-rows", planner.MaxBatchRows, "with --batches 0, the widest batch the kernels run as planned (the one-pass kernels' row limit)")
	drafter := flags.Bool("drafter", false, "the step verifies the drafter's tokens (its stages and dense weights run on the last range)")
	cutsSpec := flags.String("cuts", "", "layers a range may start at, comma-separated (default: any), for --balance")
	balance := flags.Bool("balance", false, "look for ranges with a shorter slowest range within --budget-gb a GPU")
	budget := flags.Float64("budget-gb", 92, "GPU memory a range's weights may take, for --balance")
	bandwidth := flags.Float64("bandwidth-gbs", planner.RTXPro6000.BandwidthGBs, "the GPU's copy bandwidth")
	gpu := flags.String("gpu", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "the card the plan is for")
	kernels := flags.String("kernels", "", "the kernel table to name in the plan (a path)")
	out := flags.String("out", "", "where to write the plan (default: stdout, after the prediction)")
	asJSON := flags.Bool("json", false, "print the prediction as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	var lb planner.LayerBytes
	switch {
	case *layersPath != "":
		data, err := os.ReadFile(*layersPath)
		if err != nil {
			return err
		}
		if lb, err = planner.ReadLayerBytesJSON(data); err != nil {
			return err
		}
	case *artifact != "":
		manifest, err := os.ReadFile(filepath.Join(*artifact, "manifest.json"))
		if err != nil {
			return err
		}
		binding, err := os.ReadFile(filepath.Join(*artifact, "binding.json"))
		if err != nil {
			return err
		}
		if lb, err = planner.ReadLayerBytes(manifest, binding); err != nil {
			return err
		}
	default:
		return fmt.Errorf("vxpu plan pipeline: --artifact DIR or --layers FILE")
	}
	log.Info("read the model's inventory", "layers", len(lb.Layers), "head_gb", float64(lb.HeadBytes)/1e9, "top_k", lb.TopK)
	if *layersOut != "" {
		data, err := json.MarshalIndent(lb, "", " ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(*layersOut, data, 0o644); err != nil {
			return err
		}
	}
	ranges, err := kernelplan.ParseRanges(*rangesSpec)
	if err != nil {
		return err
	}
	plan := &kernelplan.PipelinePlan{Format: kernelplan.PlanFormat, GPU: *gpu, Ranges: ranges, Experts: "with-layers",
		Pipeline: kernelplan.Pipeline{Batches: *batches, Schedule: kernelplan.ScheduleRangeOrder}}
	if *kernels != "" {
		plan.Kernels = json.RawMessage(strconv.Quote(*kernels))
	}
	cost := planner.RTXPro6000
	cost.BandwidthGBs = *bandwidth
	var choice *planner.ShapeChoice
	if *batches == 0 {
		curve, err := distinctCurve(lb, *distinct, *distinctAt)
		if err != nil {
			return err
		}
		c, err := planner.ChooseShape(lb, plan, *rows, len(plan.Ranges), *latency, curve, *drafter, cost, *maxRows)
		if err != nil {
			return err
		}
		choice = &c
		*batches, *rows, *distinct = c.Chosen.Batches, c.Chosen.RowsPerBatch, c.Chosen.Distinct
		plan.Pipeline.Batches = *batches
		plan.Explain = append(plan.Explain, "shape: "+c.Why)
	}
	w := planner.Workload{Rows: *rows, DistinctExperts: *distinct, Batches: *batches, Drafter: *drafter}
	pred, err := planner.PredictPipeline(lb, plan, w, cost)
	if err != nil {
		return err
	}
	if *balance {
		var cuts []int
		if *cutsSpec != "" {
			for _, s := range strings.Split(*cutsSpec, ",") {
				l, err := strconv.Atoi(strings.TrimSpace(s))
				if err != nil {
					return fmt.Errorf("--cuts: %w", err)
				}
				cuts = append(cuts, l)
			}
		}
		if plan, pred, err = planner.Balance(lb, plan, w, cost, cuts, *budget); err != nil {
			return err
		}
	}
	plan.Explain = append(plan.Explain, pred.Basis...)
	if *asJSON {
		var v any = pred
		if choice != nil {
			v = choice
		}
		data, err := json.MarshalIndent(v, "", " ")
		if err != nil {
			return err
		}
		fmt.Println(string(data))
	} else {
		if choice != nil {
			fmt.Println("shape: " + choice.Why)
			for _, cand := range choice.Candidates {
				fits := "fits"
				if !cand.Runnable {
					fits = "wider than the kernels' row limit"
				} else if !cand.Fits {
					fits = "over the budget"
				}
				fmt.Printf("  %d batches of %3d rows: %5.0f tok/s, a batch step every %5.1f ms, latency %5.1f ms, %3.0f distinct experts a layer (%s)\n",
					cand.Batches, cand.RowsPerBatch, cand.AggregateTS, cand.Prediction.PeriodMs, cand.LatencyMs, cand.Distinct, fits)
			}
		}
		fmt.Printf("%d batches of %d rows over %d ranges: %.0f tok/s aggregate, a batch step every %.1f ms, latency %.1f ms (one batch alone %.1f); floor %.0f tok/s at the slowest range's bytes\n",
			*batches, *rows, len(plan.Ranges), pred.AggregateTokS, pred.PeriodMs, pred.LatencyMs, pred.SingleBatchMs, pred.FloorTokS)
		for _, rc := range pred.Ranges {
			fmt.Printf("  gpu %d layers %-6s %2d layers  dense %5.2f GB  experts %6.2f GB  holds %4.0f GB  bytes %5.1f ms + fixed %4.1f = %5.1f ms\n",
				rc.Range.GPU, rc.Range.Layers, rc.Layers, rc.DenseGB, rc.ExpertGB, rc.HeldGB, rc.BytesMs, rc.FixedMs, rc.Ms)
		}
		if pred.BalanceWarning != "" {
			fmt.Println("  " + pred.BalanceWarning)
		}
		for _, b := range pred.Basis {
			fmt.Println("  " + b)
		}
	}
	data, err := plan.Write()
	if err != nil {
		return err
	}
	if *out == "" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, data, 0o644)
}

// distinctCurve is what --batches 0 estimates each candidate width's
// distinct experts from: the measured points of --distinct-at, else a
// fixed --distinct at every width (0: top-k, the same prompt in every row).
func distinctCurve(lb planner.LayerBytes, fixed float64, at string) (planner.DistinctCurve, error) {
	experts := 0
	for _, l := range lb.Layers {
		if l.Experts > experts {
			experts = l.Experts
		}
	}
	curve := planner.DistinctCurve{TopK: lb.TopK, Experts: experts}
	if at != "" {
		pts, err := planner.ParseDistinctPoints(at)
		if err != nil {
			return curve, err
		}
		curve.Points = pts
		return curve, nil
	}
	if fixed > 0 {
		// the same count at every width: two points on a flat line
		curve.Points = []planner.DistinctPoint{{Rows: 2, Distinct: fixed}, {Rows: 1 << 20, Distinct: fixed}}
	}
	return curve, nil
}
