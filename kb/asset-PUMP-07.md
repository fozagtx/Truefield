# Asset sheet: PUMP-07 / BRG-1 (demo rig modelled on the NASA IMS bearing test rig)

> Demo asset sheet written for Truefield. Rig data from the IMS Bearing Data Set readme (Lee et al., 2007, NASA PCoE).
> Plant-specific fields (tag numbers, MCC, part numbers) are SYNTHETIC for the demo.

## 1. Description
- Asset: PUMP-07 process water pump, drive-end shaft bearing BRG-1.
- Bearing: Rexnord ZA-2115 double-row roller bearing, 16 rollers per row, pitch diameter 2.815 in, roller diameter 0.331 in, contact angle 15.17°.
- Shaft speed: 2000 rpm (33.3 Hz), constant. Radial load 6000 lb via spring mechanism.
- Lubrication: forced oil circulation loop (oil pump OP-07), with magnetic plug in the return line.
- Criticality: B. Location: Hall 2, line 3.

## 2. Fault frequencies (characteristic defect frequencies at 2000 rpm)
| Frequency | Hz | Indicates |
|---|---|---|
| Shaft (1X) | 33.3 | imbalance, misalignment |
| BPFO | 236 | outer-race defect |
| BPFI | 297 | inner-race defect |
| BSF | 278 | rolling-element (roller) defect |
| FTF | 15 | cage defect |

Section 2.1 (envelope analysis): "A rising peak in the envelope spectrum at BPFO (236 Hz) and its harmonics, together with rising kurtosis, indicates an outer-race defect such as spalling. Inner-race defects show at BPFI (297 Hz) with 1X sidebands."

## 3. Spares (SYNTHETIC part numbers for the demo)
| Part no. | Description | Qty per job |
|---|---|---|
| BRG-ZA2115 | Rexnord ZA-2115 double-row bearing | 1 |
| SEAL-07-DE | Drive-end lip seal kit | 1 |
| OIL-ISO68-5L | ISO VG 68 turbine oil, 5 L | 2 |

## 4. Lubrication spec
Section 4: "Oil ISO VG 68. Normal housing temperature 40-55 °C. Above 70 °C investigate lubrication; above 90 °C stop the machine."
