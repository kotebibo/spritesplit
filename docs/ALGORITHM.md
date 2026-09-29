# How spritesplit decomposes a sprite

Every step is deterministic, uses no ML, and exists because a simpler idea
failed on real pixel art. Module names are in `src/spritesplit/`.

```
PNG ─► upscale detection ─► palette ─► background ─► outline ─► ramps
    ─► per-pixel ramp assignment ─► base per ramp ─► layers ─► .ora
```

**Invariant:** stack the layers (outline over highlights over shading over
flats over background, each opaque pixel replacing what's below) and you get
the input back exactly. Every fixture test checks this.

---

## 1. Upscale detection (`image.py`)

Sprites are often exported at 2×–8×. If an image is an exact k× nearest-neighbour
upscale, **every run of identical pixels starts at a multiple of k**, in every
row and column. So:

```
k = gcd(width, height, every x where a colour changes, every y where a colour changes)
```

If all boundaries sit on multiples of k in both directions, each k×k block has
no internal change, so it's uniform. No separate check is needed. The
decomposition runs at native resolution; `--scale` re-upscales the output.

(Arbitrary, non-integer grid alignment, such as diffusion output, is a separate
problem: FFT/autocorrelation grid snapping. That belongs to the generation phase.)

## 2. Palette (`palette.py`, `color.py`)

`np.unique` over packed RGBA values gives the colour table plus an index map.
Palette order depends only on the colours, never on their positions, so every
later "ties go to the lower index" rule is stable.

Each colour is converted to **OKLab**, a perceptual space where L is lightness
and (a, b) is the chroma plane. Values are rounded to 4 decimals because cube
roots can differ in the last bit between platforms.

## 3. Background (`background.py`)

1. Any transparent pixels → they are the background.
2. Otherwise, a colour present in ≥ 2 corners is the background colour, and a
   flood fill from every border pixel of that colour, **over palette indices**,
   marks the background. Indices can't "bleed" into a similar sprite colour
   the way an RGB tolerance can.
3. Otherwise, there is no background.

## 4. Outline (`outline.py`)

"Darkest colour = outline" breaks on black hats, dark boots and shadowed
sides. Instead:

**Depth.** Peel the sprite like an onion. Shell 1 is the pixels touching the
outside, shell 2 is those touching shell 1, and so on. The *edge* is the part
of shell 1 that touches real background; a sprite cut by the image border has
no outline along the cut.

A colour *c* is an **outline of thickness t** (try t = 1, then 2) when:

| Test | Why |
|---|---|
| *c* covers ≥ 25% of the edge | it draws on the silhouette |
| *c* fills shells 2..t (≥ 50% of its shell-1 count) | the band really is t thick |
| *c* in shell t+1 ≤ 0.6 × *c* in shell t | **it stops**: a fill colour keeps going inward |
| the outline colours cover ≥ 40% of the edge on **each** of the four sides | **it encloses**: a dark terminator crescent is a band that stops, but only on one side |

Outline pixels are then the outline colour within the band, plus
outline-colour pixels anywhere that are *thin* (not part of any one-colour 2×2
square), which are internal lines. A 1-px line, whether straight, diagonal or
staircase, never contains a one-colour 2×2 square; a filled area almost always
does.

- **Black hat:** the hat is fat and deep, so it stays a fill (its own material).
  The "stops" test is relative, so the hat continuing inward doesn't disqualify
  the rest of the black line, which does stop.
- **Sel-out** (no single outline colour): accept edge pixels that are ≥ 0.03 L
  darker than the pixels just inside them, if that holds for ≥ 50% of the edge.
- The 2-px case exists because real art uses it: Kenney's *Tiny Dungeon* has
  2-px outlines on every character.

## 5. Ramps: the core (`ramps.py`)

A **ramp** is one material at several light levels. The naive idea is to
cluster colours with similar hue and chroma. That fails, because skilled
artists **hue-shift**: shadows drift toward blue or purple and highlights
toward yellow, so a skin ramp can span 40–60° of hue end to end. What stays
small is each individual *step*. So ramps are **chained, not clustered**.

**Adjacency.** Count 4-neighbour pairs between every two colours among fill
pixels (not background, not outline). Shades of one material touch each other;
different materials are usually separated by outline. Colours that never touch
are never linked.

**Step compatibility**, darker *i* → lighter *j*:

```
0.02 ≤ L_j − L_i ≤ 0.40          a real lightness step, not a jump
|ab_j − ab_i| ≤ 0.10              small drift in the chroma plane
adjacency(i, j) ≥ 1
cost = (ΔL / 0.40)² + (Δab / 0.10)²
```

Using distance in the a/b plane, rather than hue angle, means greys (where hue
is noise) and hue wrap-around need no special cases. Squared distances only:
`+ − ×` are exactly rounded in IEEE 754, while `sqrt` and `hypot` go through
libm, and a one-ulp difference could flip a tie on another OS.

**Mutual best.** Each colour picks its cheapest lighter partner (`best_up`)
and its cheapest darker partner (`best_down`). Keep link *i → j* only if
`best_up[i] == j` **and** `best_down[j] == i`. Each colour then has at most one
link up and one down, so the links form disjoint chains ordered by lightness:
the ramps. No threshold on the ramp as a whole is needed.

**Shared colours.** Limited palettes reuse colours across materials (the
darkest skin shadow is also the dark wood). A *one-sided* best link, where *j*
wants *i* as its darker step but *i* chose someone else, marks *i* as
**shared** into *j*'s ramp.

## 6. Per-pixel assignment (`decompose.py: assign_pixels`)

The palette → ramp mapping is not a function once colours are shared, so
shared colours are resolved **per pixel**. Each connected blob of a shared
colour counts its borders with unambiguous pixels: every neighbour pair is one
vote for that neighbour's ramp. The primary ramp came from a mutual-best link,
which is stronger evidence than a one-sided link, so a shared ramp wins only
with **more than 2×** the primary's votes.

## 7. Base and layers (`decompose.py`)

- **Base** of a ramp = the chain member covering the most pixels. Artists lay
  down the base first and shade on top. Ties go to the darker member.
- **flats** = every fill pixel painted with its ramp's base.
- **shading** = fill pixels darker (L) than their base.
- **highlights** = fill pixels at least as light as their base but not the base
  colour. The ≥ matters: a shared colour exactly as light as another ramp's base
  must land somewhere, or the split would lose pixels.

## 8. Output (`pngenc.py`, `ora.py`)

`.ora` is a ZIP of PNGs plus `stack.xml`, stdlib only. For byte-identical output:

- **PNG:** our own encoder writes deflate *stored* blocks (fully specified by
  RFC 1951) plus CRC32/Adler-32. zlib's compressed output varies by zlib
  version, and Linux Python links the system zlib.
- **ZIP:** fixed entry order, `ZIP_STORED`, 1980-01-01 timestamps, and a fixed
  "made by" system. Otherwise `zipfile` stamps the host OS.

CI runs the golden tests on Linux, Windows and macOS against the same
committed bytes.

## Tuning

All thresholds live in `OutlineParams` and `RampParams`, which are frozen
dataclasses with a comment on each field. When you change one, regenerate the
goldens (`scripts/update_golden.py`) and **look at** the layer PNG diffs.
The fixture tests (`tests/test_fixtures.py`) check meaning; the goldens check
bytes.

## Known failure modes

- Two materials sharing one ramp (Kenney skin + leather) merge. Colour alone
  can't separate them.
- A 1-px dark shading band that wraps most of an outline-less sprite can pass
  as an outline.
- Anti-aliasing pixels between two ramps get whichever ramp their colour
  chains into.
- Dithering is kept as-is: its pixels go to shading or highlights by lightness,
  which is usually what you want.
