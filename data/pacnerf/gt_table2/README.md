# PAC-NeRF GT_table2

These five JSON files are byte-for-byte copies of
`PAC_NERF_original_data/GT_table2/` in the original PhysFlow workspace.
They contain numbered instances grouped by material (`0`–`9`, or `0`–`4`
for sand), including an initial velocity `v` and material parameters.

This dataset is separate from [`../groundtruth.json`](../groundtruth.json),
which records the nine named PAC-NeRF Table 1 scenes (Letter, Play-Doh,
etc.). Do not use these numbered instances as the material ground truth for
those named scenes without an explicit instance mapping.

The JSON payloads are kept unchanged so existing readers can use the
original filenames and keys (`E`, `nu`, `mu`, `kappa`, `ys`, `eta`, `fa`, `v`).
