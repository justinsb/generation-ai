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
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"k8s.io/klog/v2"
)

// vxpu kernels plan --record R [--record R2] [--out table.json]
// vxpu kernels check --table T --record R
// vxpu kernels measure --record R [--table T] --out table.json [--python P] [--sms 188]
//
// The kernel planner (pkg/kernelplan): from the records engines wrote
// (VXPU_PLAN_RECORD), the table an engine applies (VXPU_PLAN_TABLE); the
// check that a run applied its table; and the measuring step, which
// sweeps the GEMV templates' blocks per shape class on the card this runs
// on (through python -m vxpu.measure) and writes the fastest into the
// table (hero run 11).
func cmdKernels(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vxpu kernels plan|check|measure ...")
	}
	switch args[0] {
	case "plan":
		return runKernelsPlan(ctx, args[1:])
	case "check":
		return runKernelsCheck(ctx, args[1:])
	case "measure":
		return runKernelsMeasure(ctx, args[1:])
	}
	return fmt.Errorf("vxpu kernels: unknown subcommand %q (plan, check, measure)", args[0])
}

type stringList []string

func (l *stringList) String() string     { return strings.Join(*l, ",") }
func (l *stringList) Set(s string) error { *l = append(*l, s); return nil }

func runKernelsPlan(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("kernels plan", flag.ContinueOnError)
	var paths stringList
	flags.Var(&paths, "record", "a kernel record an engine wrote (repeatable; one GPU)")
	out := flags.String("out", "", "where to write the table (default: stdout)")
	launch := flags.Float64("launch-us", 0, "one kernel launch in a CUDA graph, credited to a fused body (default: the record's launch_us)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("vxpu kernels plan: at least one --record")
	}
	var records []*kernelplan.Record
	for _, p := range paths {
		r, err := kernelplan.ReadRecordFile(p)
		if err != nil {
			return err
		}
		log.Info("read a kernel record", "path", p, "gpu", r.Hardware.GPU, "sites", len(r.Sites))
		records = append(records, r)
	}
	table, err := kernelplan.Plan(ctx, records, kernelplan.Options{LaunchUs: *launch})
	if err != nil {
		return err
	}
	data, err := table.Write()
	if err != nil {
		return err
	}
	for _, line := range table.Explain {
		fmt.Fprintln(os.Stderr, line)
	}
	if *out == "" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, data, 0o644)
}

func runKernelsCheck(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("kernels check", flag.ContinueOnError)
	tablePath := flags.String("table", "", "the kernel table the engine was given (optional: without it only the ledger is printed)")
	recordPath := flags.String("record", "", "the kernel record the engine wrote")
	stepsPath := flags.String("step-times", "", "per-site microseconds from the step itself, {node: us} (dev/hero/ledger.py), in place of the lowering's microbenchmarks")
	bandwidth := flags.Float64("bandwidth-gb-s", 0, "the card's memory bandwidth for the floors (default: the record's measurement, else the card's known figure)")
	threshold := flags.Float64("threshold", 0, "fail a measured site under this efficiency (0.85) unless the table records an accepted reason for it; 0: report only")
	worst := flags.Int("worst", 20, "how many of the worst sites to print")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *recordPath == "" {
		return fmt.Errorf("vxpu kernels check: --record (and --table to check the table was applied)")
	}
	record, err := kernelplan.ReadRecordFile(*recordPath)
	if err != nil {
		return err
	}
	var table *kernelplan.Table
	if *tablePath != "" {
		if table, err = kernelplan.ReadTableFile(*tablePath); err != nil {
			return err
		}
	}
	var steps kernelplan.StepTimes
	if *stepsPath != "" {
		data, err := os.ReadFile(*stepsPath)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(data, &steps); err != nil {
			return fmt.Errorf("parsing %s: %w", *stepsPath, err)
		}
	}
	ledger, err := kernelplan.NewLedger(record, kernelplan.Bandwidth(record, *bandwidth), steps, table)
	if err != nil {
		return err
	}
	fmt.Print(ledger.String(*worst))
	var failed []string
	if table != nil {
		bad := kernelplan.Check(table, record)
		fmt.Printf("%d sites, %d did not run the table's body\n", len(record.Sites), len(bad))
		for _, m := range bad {
			fmt.Println("  " + m.String())
		}
		if len(bad) > 0 {
			failed = append(failed, fmt.Sprintf("the table was not applied at %d sites", len(bad)))
		}
		badBlocks, err := kernelplan.CheckBlocks(table, record)
		if err != nil && len(table.Blocks) > 0 {
			return err
		}
		if len(table.Blocks) > 0 {
			fmt.Printf("%d block classes in the table, %d ran other blocks\n", len(table.Blocks), len(badBlocks))
			for _, m := range badBlocks {
				fmt.Println("  " + m.String())
			}
			if len(badBlocks) > 0 {
				failed = append(failed, fmt.Sprintf("%d block classes ran other blocks", len(badBlocks)))
			}
		}
	}
	if *threshold > 0 {
		under := ledger.Under(*threshold)
		fmt.Printf("%d measured sites under %.0f%% of their floor without a recorded reason\n", len(under), 100**threshold)
		for _, line := range under {
			fmt.Printf("  %s at %s (%s): %.0f%% (%.1f of %.1f us)\n", line.Site.Op, line.Site.Site, line.Site.Shapes,
				100*line.Efficiency, line.FloorUs, line.Us)
		}
		if len(under) > 0 {
			failed = append(failed, fmt.Sprintf("%d sites under the threshold", len(under)))
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("%s", strings.Join(failed, "; "))
	}
	return nil
}

func runKernelsMeasure(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("kernels measure", flag.ContinueOnError)
	recordPath := flags.String("record", "", "the kernel record whose GEMV shape classes to sweep (stats.gemv_classes)")
	tablePath := flags.String("table", "", "a table to extend: its sites and knobs are kept, its blocks replaced (default: a new table)")
	out := flags.String("out", "", "where to write the table (default: stdout)")
	python := flags.String("python", "python", "the interpreter that runs python -m vxpu.measure (with vxpu on its path and the GPU visible)")
	device := flags.String("device", "cuda:0", "the CUDA device the sweep runs on")
	sms := flags.Int("sms", 0, "the card's streaming multiprocessors, the K split's bound (0: no bound)")
	var templates stringList
	flags.Var(&templates, "template", "only this template (repeatable: gemv_rows, dual_gemv_swiglu, gemv_pe)")
	maxRows := flags.Int("max-rows", 0, "only classes of at most this many rows (0: all)")
	keepPath := flags.String("keep", "", "keep the runner's result lines here (default: a temporary file)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *recordPath == "" {
		return fmt.Errorf("vxpu kernels measure: --record")
	}
	record, err := kernelplan.ReadRecordFile(*recordPath)
	if err != nil {
		return err
	}
	classes, err := kernelplan.Classes(record)
	if err != nil {
		return err
	}
	table := &kernelplan.Table{GPU: record.Hardware.GPU, Sites: []kernelplan.Entry{}}
	if *tablePath != "" {
		if table, err = kernelplan.ReadTableFile(*tablePath); err != nil {
			return err
		}
	}
	runner := kernelplan.PythonRunner{Python: *python, Device: *device, Keep: *keepPath, Log: func(format string, a ...any) { log.Info(fmt.Sprintf(format, a...)) }}
	log.Info("measuring the GEMV shape classes", "classes", len(classes), "gpu", record.Hardware.GPU, "device", *device)
	entries, explain, err := kernelplan.Measure(ctx, classes, runner, kernelplan.MeasureOptions{SMs: *sms, Templates: templates, MaxRows: *maxRows})
	if err != nil {
		return err
	}
	// a table being extended keeps its blocks for the classes this record did not run
	merged := map[string]kernelplan.BlockEntry{}
	for _, e := range table.Blocks {
		merged[e.Shapes] = e
	}
	for _, e := range entries {
		merged[e.Shapes] = e
	}
	table.Blocks = table.Blocks[:0]
	for _, e := range merged {
		table.Blocks = append(table.Blocks, e)
	}
	bytesAStep, usAStep := kernelplan.StepFloor(classes, entries)
	explain = append(explain, fmt.Sprintf("the GEMV classes measured: %.2f GB a step at the record's counts, %.2f ms a step under the chosen blocks", float64(bytesAStep)/1e9, usAStep/1e3))
	table.Explain = append(table.Explain, explain...)
	data, err := table.Write()
	if err != nil {
		return err
	}
	for _, line := range explain {
		fmt.Fprintln(os.Stderr, line)
	}
	if *out == "" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, data, 0o644)
}
