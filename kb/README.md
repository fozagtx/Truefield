# Knowledge library (`truefield-kb`) sources

| File | Content | Origin |
|---|---|---|
| `asset-PUMP-07.md` | Rig description, fault frequencies, spares, lubrication spec | Rig data from the NASA IMS readme; plant fields SYNTHETIC |
| `past-work-orders.md` | 12 historic work orders | SYNTHETIC |
| `loto-PUMP-07.md` | Plant LOTO procedure for PUMP-07 | SYNTHETIC, written to be consistent with OSHA 1910.147 |
| OSHA 29 CFR 1910.147 | The control of hazardous energy | Imported with `import-url`: https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147 |
| Bearing failure-analysis guide | e.g. SKF bearing damage and failure analysis | Upload a public OEM PDF yourself after checking its licence (not committed) |

`python scripts/seed_library.py` uploads the markdown files and imports the OSHA page into `EVOLUS_LIBRARY_ID`.
Closed work orders are written back as `WO-xxxx.md` documents at runtime (FR-9).
