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
	"fmt"
	"sort"
	"strings"
)

// The roofline ledger (hero run 11): every call site of a record against
// its own floor. A site's floor is the bytes it must move (the graph
// knows them: lower.site_bytes) over the card's memory bandwidth (the
// record's measured copy, or a known figure for the card); its efficiency
// is floor over the microseconds its chosen body took. The ledger is a
// function of the record alone, so it reads any model's record the same
// way, and it is what the measuring step and the planner's threshold check
// work from: a site under the threshold without a recorded reason is the
// kernel library's debt, named.

// knownBandwidth is the measured copy bandwidth per card, for records
// written before the engine measured it (GB/s, read and write counted).
var knownBandwidth = map[string]float64{
	"NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 1460,
	"NVIDIA RTX PRO 6000 Blackwell Server Edition":      1460,
	"NVIDIA A100-SXM4-80GB":                             1800,
	"NVIDIA A100-SXM4-40GB":                             1400,
	"NVIDIA L4":                                         260,
}

// Bandwidth returns the bandwidth the ledger should use for a record:
// the override when given, else the record's own measurements (the read
// first, the copy after it), else the card's known figure; zero when none
// applies.
func Bandwidth(r *Record, override float64) float64 {
	switch {
	case override > 0:
		return override
	case r.Hardware.ReadBandwidthGBs > 0:
		return r.Hardware.ReadBandwidthGBs
	case r.Hardware.BandwidthGBs > 0:
		return r.Hardware.BandwidthGBs
	}
	for name, bw := range knownBandwidth {
		if strings.Contains(r.Hardware.GPU, strings.TrimPrefix(name, "NVIDIA ")) {
			return bw
		}
	}
	return 0
}

// Line is one site's line of the ledger.
type Line struct {
	Site       Site
	Bytes      int64   // what the floor counts
	Us         float64 // the chosen body's measured microseconds (the lowering's, or the step's when joined)
	FloorUs    float64
	Efficiency float64 // FloorUs / Us; 0 when Us is unknown
	LostUs     float64 // Us - FloorUs
	Accepted   string  // the table's recorded reason, when under the threshold
}

// Total is a template's (an op's) totals across its sites.
type Total struct {
	Op         string
	Sites      int
	Measured   int // sites with a measured time
	Bytes      int64
	Us         float64
	FloorUs    float64
	Efficiency float64
	LostUs     float64
}

// Ledger is a record's sites against their floors.
type Ledger struct {
	BandwidthGBs float64
	Lines        []Line  // every site with bytes, worst (most lost microseconds) first
	Totals       []Total // per op, most lost first
	Unmeasured   int     // sites with bytes but no measured time
	NoBytes      int     // sites without bytes (an older record, or no shapes)
	StepFloorUs  float64 // every site's bytes over the bandwidth
	StepUs       float64 // every measured site's microseconds
}

// StepTimes gives a site's microseconds from the step itself (a profile
// joined to the record's sites, dev/hero/ledger.py) in place of the
// lowering's microbenchmark; keyed by the site's node name.
type StepTimes map[string]float64

// NewLedger builds the ledger of r at bandwidth bandwidthGBs. A site's
// time is steps[node] when given, else the chosen body's measured_us.
// Sites under threshold (0: none) are failures unless the table (nil:
// no table) records an accepted reason for them.
func NewLedger(r *Record, bandwidthGBs float64, steps StepTimes, t *Table) (*Ledger, error) {
	if bandwidthGBs <= 0 {
		return nil, fmt.Errorf("ledger: the bandwidth is unknown: the record has no gpu_bandwidth_gb_s and %q is not a card the planner knows; pass --bandwidth-gb-s", r.Hardware.GPU)
	}
	l := &Ledger{BandwidthGBs: bandwidthGBs}
	totals := map[string]*Total{}
	for _, s := range r.Sites {
		if s.Bytes == nil {
			l.NoBytes++
			continue
		}
		line := Line{Site: s, Bytes: s.Bytes.Floor()}
		line.FloorUs = float64(line.Bytes) / (bandwidthGBs * 1e3) // bytes / (GB/s) = ns; /1e3 = µs
		if us, ok := steps[s.Node]; ok && us > 0 {
			line.Us = us
		} else if us, ok := s.MeasuredUs[s.Chosen]; ok && us > 0 {
			line.Us = us
		}
		if t != nil {
			if e := t.Entry(s.Op, s.Site, s.Shapes); e != nil {
				line.Accepted = e.Accepted
			}
		}
		tot := totals[s.Op]
		if tot == nil {
			tot = &Total{Op: s.Op}
			totals[s.Op] = tot
		}
		tot.Sites++
		tot.Bytes += line.Bytes
		tot.FloorUs += line.FloorUs
		l.StepFloorUs += line.FloorUs
		if line.Us > 0 {
			line.Efficiency = line.FloorUs / line.Us
			line.LostUs = line.Us - line.FloorUs
			tot.Measured++
			tot.Us += line.Us
			tot.LostUs += line.LostUs
			l.StepUs += line.Us
		} else {
			l.Unmeasured++
		}
		l.Lines = append(l.Lines, line)
	}
	for _, tot := range totals {
		if tot.Us > 0 {
			tot.Efficiency = tot.FloorUs / tot.Us
		}
		l.Totals = append(l.Totals, *tot)
	}
	sort.SliceStable(l.Lines, func(i, j int) bool { // measured lines first, most lost first; unmeasured after
		a, b := l.Lines[i], l.Lines[j]
		if (a.Us > 0) != (b.Us > 0) {
			return a.Us > 0
		}
		return a.LostUs > b.LostUs
	})
	sort.SliceStable(l.Totals, func(i, j int) bool { return l.Totals[i].LostUs > l.Totals[j].LostUs })
	return l, nil
}

// Under returns the measured lines whose efficiency is under threshold
// and that no table entry accepts: the check's failures.
func (l *Ledger) Under(threshold float64) []Line {
	var out []Line
	for _, line := range l.Lines {
		if line.Us > 0 && line.Efficiency < threshold && line.Accepted == "" {
			out = append(out, line)
		}
	}
	return out
}

// String renders the ledger: the step's totals, the per-op totals most
// lost first, and the worst sites (at most worst of them).
func (l *Ledger) String(worst int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "ledger at %.0f GB/s: %d sites with bytes (%d measured, %d unmeasured), %d without bytes\n",
		l.BandwidthGBs, len(l.Lines), len(l.Lines)-l.Unmeasured, l.Unmeasured, l.NoBytes)
	if l.StepUs > 0 {
		fmt.Fprintf(&b, "the sites' floor %.0f us; measured %.0f us; efficiency %.0f%%; lost %.0f us\n",
			l.StepFloorUs, l.StepUs, 100*l.StepFloorUs/l.StepUs, l.StepUs-l.StepFloorUs)
	}
	fmt.Fprintf(&b, "per op (most lost first): sites, bytes, floor us, measured us, efficiency, lost us\n")
	for _, t := range l.Totals {
		eff := "-"
		if t.Us > 0 {
			eff = fmt.Sprintf("%.0f%%", 100*t.Efficiency)
		}
		fmt.Fprintf(&b, "  %-22s %4d %10.1f MB %8.1f %8.1f %5s %8.1f\n", t.Op, t.Sites, float64(t.Bytes)/1e6, t.FloorUs, t.Us, eff, t.LostUs)
	}
	fmt.Fprintf(&b, "worst sites by lost us: op, site, shapes, chosen, bytes, floor us, measured us, efficiency\n")
	n := 0
	for _, line := range l.Lines {
		if line.Us == 0 || n >= worst {
			continue
		}
		n++
		fmt.Fprintf(&b, "  %-18s %-32s %-34s %-16s %8.1f MB %7.1f %7.1f %4.0f%%%s\n", line.Site.Op, line.Site.Site,
			line.Site.Shapes, line.Site.Chosen, float64(line.Bytes)/1e6, line.FloorUs, line.Us, 100*line.Efficiency,
			accepted(line.Accepted))
	}
	return b.String()
}

func accepted(why string) string {
	if why == "" {
		return ""
	}
	return " (accepted: " + why + ")"
}
