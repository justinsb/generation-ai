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
	"testing"
)

// Hero run 9's measured distinct-expert counts with a prompt per row: 37.3
// at 9 rows, 100 at 33 (design.md); the curve passes through them.
func hero9Curve() DistinctCurve {
	return DistinctCurve{Points: []DistinctPoint{{9, 37.3}, {33, 100}}, TopK: 6, Experts: 384}
}

func TestDistinctCurve(t *testing.T) {
	dc := hero9Curve()
	for _, c := range []struct {
		rows int
		want float64
	}{{1, 6}, {9, 37.3}, {33, 100}, {66, 169}, {132, 287}} {
		if got := dc.At(c.rows); math.Abs(got-c.want)/c.want > 0.01 {
			t.Errorf("distinct at %d rows: %.1f, want %.1f", c.rows, got, c.want)
		}
	}
	if got := dc.At(1000); got != 384 {
		t.Errorf("distinct at 1000 rows: %.1f, want the 384 a layer has", got)
	}
	if got := (DistinctCurve{TopK: 6, Experts: 384}).At(33); got != 6 {
		t.Errorf("no points: %.1f, want top-k 6", got)
	}
	if got := (DistinctCurve{Points: []DistinctPoint{{33, 100}}, TopK: 6, Experts: 384}).At(2); got > 12 {
		t.Errorf("two rows select at most 12: %.1f", got)
	}
	pts, err := ParseDistinctPoints("33=100, 9=37.3")
	if err != nil || len(pts) != 2 || pts[0].Rows != 9 {
		t.Fatalf("parse: %v %v", pts, err)
	}
	if _, err := ParseDistinctPoints("9"); err == nil {
		t.Error("parse accepted a point without a count")
	}
}

// Hero run 10's wide-batch question, as the planner answers it before the
// measurement (design.md, "one wide batch against several in flight"):
// with the 33-prompt list cycled every width selects about 100 distinct
// experts and the batches in flight beat the wide batch by more than
// 20% at every budget they fit (the model's own pick is three batches of
// 44, 1,375 tok/s, over four of 33 at 1,243: at the same bytes a step,
// more rows a batch is more tokens a period); with a prompt per row the
// wide batch falls further behind.
func TestChooseShape_WideBatchAgainstBatchesInFlight(t *testing.T) {
	lb := deepseekLayers(t)
	p := hero8Plan(t, 4)
	cycled := DistinctCurve{Points: []DistinctPoint{{33, 100}, {132, 100}}, TopK: 6, Experts: 384}
	choice, err := ChooseShape(lb, p, 132, 4, 0, cycled, false, RTXPro6000)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(choice.Why)
	if choice.Chosen.Batches < 3 || len(choice.Candidates) != 4 {
		t.Errorf("132 rows, no budget: chose %d×%d from %d candidates, want three or four batches in flight", choice.Chosen.Batches, choice.Chosen.RowsPerBatch, len(choice.Candidates))
	}
	wide := choice.Candidates[0]
	if wide.Batches != 1 || wide.AggregateTS > choice.Chosen.AggregateTS*0.8 || wide.LatencyMs < choice.Chosen.LatencyMs {
		t.Errorf("the wide batch: %.0f tok/s at %.1f ms against %.0f at %.1f", wide.AggregateTS, wide.LatencyMs, choice.Chosen.AggregateTS, choice.Chosen.LatencyMs)
	}
	// the budget: the batches in flight take 96 to 110 ms a batch step in the model; under 80 ms nothing with 132 rows fits
	if c, _ := ChooseShape(lb, p, 132, 4, 110, cycled, false, RTXPro6000); c.Chosen.Batches < 3 || !c.Chosen.Fits {
		t.Errorf("110 ms budget: chose %d×%d fits=%v", c.Chosen.Batches, c.Chosen.RowsPerBatch, c.Chosen.Fits)
	}
	if c, _ := ChooseShape(lb, p, 132, 4, 80, cycled, false, RTXPro6000); c.Chosen.Fits {
		t.Errorf("80 ms budget: %s", c.Why)
	} else {
		t.Log(c.Why)
	}
	// a prompt per row: more distinct experts at every width, the wide batch loses by more
	perRow, _ := ChooseShape(lb, p, 132, 4, 0, hero9Curve(), false, RTXPro6000)
	t.Log(perRow.Why)
	if perRow.Chosen.Batches != 4 || perRow.Candidates[0].AggregateTS > wide.AggregateTS {
		t.Errorf("a prompt per row: chose %d batches; the wide batch %.0f tok/s against %.0f cycled", perRow.Chosen.Batches, perRow.Candidates[0].AggregateTS, wide.AggregateTS)
	}
	// one row is one batch
	if c, _ := ChooseShape(lb, p, 1, 4, 0, hero9Curve(), false, RTXPro6000); c.Chosen.Batches != 1 || len(c.Candidates) != 1 {
		t.Errorf("one row: %s", c.Why)
	}
	// nine rows under a 20 ms budget: nothing fits with a prompt per row; with the same prompt in every row, three batches of 3 do
	if c, _ := ChooseShape(lb, p, 9, 4, 20, hero9Curve(), false, RTXPro6000); c.Chosen.Fits {
		t.Errorf("9 different rows in 20 ms: %s", c.Why)
	}
	if c, _ := ChooseShape(lb, p, 9, 4, 20, DistinctCurve{TopK: 6, Experts: 384}, false, RTXPro6000); !c.Chosen.Fits {
		t.Errorf("9 same rows in 20 ms: %s", c.Why)
	} else {
		t.Log(c.Why)
	}
}
