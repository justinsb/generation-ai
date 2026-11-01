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

package kernelplan

import (
	"strings"
	"testing"
)

// A synthetic record: two GEMV sites of 20 MB (one at 15 µs, one at 25),
// an attention site of 1 MB at 4 µs, a site without bytes, a site
// without a measurement. At 1,000 GB/s the 20 MB floor is 20 µs.
func ledgerRecord() *Record {
	layer := 0
	return &Record{
		Format:   RecordFormat,
		Hardware: Hardware{GPU: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", BandwidthGBs: 1000},
		Sites: []Site{
			{Op: "vxpu::attn_q", Site: "model.layers.0.attn", Node: "q0", Shapes: "rows=1", Layer: &layer, Chosen: "gpu-triton",
				MeasuredUs: map[string]float64{"gpu-triton": 25, "gpu-fused": 40}, Bytes: &SiteBytes{Weights: 20e6, Activations: 0}},
			{Op: "vxpu::attn_q", Site: "model.layers.1.attn", Node: "q1", Shapes: "rows=1", Chosen: "gpu-triton",
				MeasuredUs: map[string]float64{"gpu-triton": 15}, Bytes: &SiteBytes{Weights: 20e6 - 1e6, Activations: 1e6, State: 5e9}},
			{Op: "vxpu::attn_select", Site: "model.layers.0.attn", Node: "s0", Shapes: "t=128", Chosen: "gpu-triton",
				MeasuredUs: map[string]float64{"gpu-triton": 4}, Bytes: &SiteBytes{Activations: 1e6}},
			{Op: "vxpu::rotary", Site: "model.layers.0.attn", Node: "r0", Chosen: "gpu-fused"},
			{Op: "vxpu::window_kv", Site: "model.layers.0.attn", Node: "w0", Chosen: "gpu-triton", Source: "table",
				Bytes: &SiteBytes{Weights: 2e6}},
		},
	}
}

func TestLedgerRanksSitesByLostMicroseconds(t *testing.T) {
	l, err := NewLedger(ledgerRecord(), Bandwidth(ledgerRecord(), 0), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.BandwidthGBs != 1000 {
		t.Fatalf("bandwidth %v, want the record's 1000", l.BandwidthGBs)
	}
	if l.NoBytes != 1 || l.Unmeasured != 1 || len(l.Lines) != 4 {
		t.Fatalf("no bytes %d, unmeasured %d, lines %d; want 1, 1, 4", l.NoBytes, l.Unmeasured, len(l.Lines))
	}
	// the state buffer is not in the floor: q1's floor is 20 µs like q0's
	if l.Lines[0].Site.Node != "q0" || l.Lines[0].FloorUs != 20 || l.Lines[0].Efficiency != 0.8 {
		t.Fatalf("worst line %+v, want q0 at 20 µs floor, 80%%", l.Lines[0])
	}
	if l.Lines[1].Site.Node != "s0" || l.Lines[1].LostUs != 3 {
		t.Fatalf("second line %+v, want s0 losing 3 µs", l.Lines[1])
	}
	if l.Lines[2].Site.Node != "q1" || l.Lines[2].Efficiency < 1.33 {
		t.Fatalf("third line %+v, want q1 over its floor (a measurement under the floor is reported, not clipped)", l.Lines[2])
	}
	// attn_select lost 3 µs in all, attn_q 0 (q0's 5 against q1's -5): attn_select leads the totals
	if l.Totals[0].Op != "vxpu::attn_select" || l.Totals[0].LostUs != 3 {
		t.Fatalf("top total %+v, want attn_select losing 3 µs", l.Totals[0])
	}
	var q *Total
	for i := range l.Totals {
		if l.Totals[i].Op == "vxpu::attn_q" {
			q = &l.Totals[i]
		}
	}
	if q == nil || q.Sites != 2 || q.Us != 40 || q.FloorUs != 40 || q.Efficiency != 1 {
		t.Fatalf("attn_q total %+v, want 2 sites, 40 µs measured, 40 µs floor, 100%%", q)
	}
	if l.StepFloorUs != 43 || l.StepUs != 44 {
		t.Fatalf("step floor %v us, measured %v; want 43 and 44 (the unmeasured site counts in the floor only)", l.StepFloorUs, l.StepUs)
	}
	out := l.String(10)
	for _, want := range []string{"ledger at 1000 GB/s", "vxpu::attn_q", "model.layers.0.attn", "80%"} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendering lacks %q:\n%s", want, out)
		}
	}
}

func TestLedgerThresholdHonoursTheTablesAcceptedReason(t *testing.T) {
	r := ledgerRecord()
	l, _ := NewLedger(r, 1000, nil, nil)
	under := l.Under(0.85)
	if len(under) != 2 || under[0].Site.Node != "q0" || under[1].Site.Node != "s0" {
		t.Fatalf("under 85%%: %d sites, want q0 and s0", len(under))
	}
	table := &Table{Sites: []Entry{{Op: "vxpu::attn_q", Site: "model.layers.0.attn", Impl: "gpu-triton",
		Accepted: "long K at one row: the card cannot be filled"}}}
	l, _ = NewLedger(r, 1000, nil, table)
	under = l.Under(0.85)
	if len(under) != 1 || under[0].Site.Node != "s0" {
		t.Fatalf("with the reason recorded: %d sites under, want only s0", len(under))
	}
}

func TestLedgerTakesTheStepsOwnTimes(t *testing.T) {
	l, _ := NewLedger(ledgerRecord(), 1000, StepTimes{"q0": 50, "w0": 4}, nil)
	if l.Lines[0].Site.Node != "q0" || l.Lines[0].Us != 50 {
		t.Fatalf("q0 should take the step's 50 µs over the lowering's 25: %+v", l.Lines[0])
	}
	if l.Unmeasured != 0 {
		t.Fatalf("w0 got a step time, so nothing is unmeasured: %d", l.Unmeasured)
	}
}

func TestBandwidthFallsBackToTheKnownCard(t *testing.T) {
	r := ledgerRecord()
	r.Hardware.ReadBandwidthGBs = 1550
	if bw := Bandwidth(r, 0); bw != 1550 {
		t.Fatalf("the read measurement should win over the copy: %v", bw)
	}
	r.Hardware.ReadBandwidthGBs = 0
	r.Hardware.BandwidthGBs = 0
	if bw := Bandwidth(r, 0); bw != 1460 {
		t.Fatalf("RTX PRO 6000 without a measurement: %v, want 1460", bw)
	}
	if bw := Bandwidth(r, 900); bw != 900 {
		t.Fatalf("override: %v, want 900", bw)
	}
	r.Hardware.GPU = "Some Other Card"
	if _, err := NewLedger(r, Bandwidth(r, 0), nil, nil); err == nil {
		t.Fatal("an unknown card without a measurement should refuse, not divide by zero")
	}
}
