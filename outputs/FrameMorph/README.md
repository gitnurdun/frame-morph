# FrameMorph 0.8 — slide studio

A C++17 / FLTK desktop editor with a CUDA animation renderer, running in Ubuntu through WSLg. Version 0.8 fixes a transparency mismatch between authored keyframes and animation, eliminating the measured brightness flash on stationary translucent shapes and text. Version 0.7 added bounded GPU artwork caching, grouped canvas movement/nudging, and multi-object alignment/distribution. Version 0.6 introduced an in-app F1 guide, asynchronous formula/image/code/SVG studios, full LaTeX, image processing and pixel formulas, image composition, more shapes and motion presets, and fourteen construct/destruct effects with timing and easing. Timeline rendering remains C++/CUDA. The optional full LaTeX and SVG backends reuse the user-approved installed TinyTeX and PyMuPDF; no packages were installed.

## Open it

**Open Toolkit Demo.cmd** opens the combined LaTeX, code, SVG, image-filter and updater demonstration. Press **F1** or **Help / F1** for the built-in guide.

Double-click **Open FrameMorph.cmd** for a new white slide, or **Open Demo.cmd** for the clean, editable studio example. **Open Properties Demo.cmd** opens the new combined name matching, appearance effects, and updater example. **Open Updater Demo.cmd** opens the new orbit/rotation/deformation example. **Open Motion Demo.cmd** opens the new shape/rotation/function example; its video is `FrameMorph-motion-demo.mp4`. **Open Transform Demo.cmd** opens the older mathematical deformation example.

The existing application folder has been updated. Existing project folders and recovery files are retained. A window that was already open continues running its previous executable; open the launcher to use the new editor.

## Start with the canvas

1. Click **Rectangle**, **Circle**, **Line**, **Arrow**, **Text**, **LaTeX**, a graph tool, or **Picture** in the ribbon. Ellipses, triangles, diamonds, stars, hexagons, source code blocks and SVG source are under **More...**.
2. Drag on the slide to create an object. Click once to use the default size. The new object is selected and receives a unique name automatically.
3. Click an object to show **Properties**. Edit its name, X/Y coordinates, width, height, rotation, opacity, color, font, stroke, text, or mathematics in that one panel. Type a value and press Enter or leave the field to commit it. Text and formula contents use **Update text or formula**.
4. Drag the selected object to move it. Drag any of its eight square handles to resize it; drag the circular handle to rotate it. The artwork follows the pointer. Shift constrains movement, preserves resize proportions, or snaps rotation to 15-degree increments.
5. Duplicate the slide, change the next pose, then press **Play**. Named objects establish the connections between slides.

Selecting is the default tool. Clicking an empty area deselects the object and reveals slide properties. Shift-click or drag a selection rectangle to select several objects. Drag any selected member to move the whole selection, or use arrow keys to nudge it. Each gesture is one Undo operation. The center buttons center the selection as a group. **More...** provides edge alignment and even horizontal/vertical spacing (three or more objects). Resize/rotation handles continue to edit the active object individually. Properties provides grouping and layer operations. Double-click text or a mathematical object to focus its content editor. The mouse wheel zooms; right or middle drag pans. **Fit slide** resets the view.

Common geometry and appearance edits use cached artwork, including for equations. Moving, resizing, rotating, changing opacity, and recoloring do not invoke Python or rebuild mathematical notation. Native construction is retained for later text, font, stroke, and formula edits. A completed drag is one undo operation; clicking an object is not an editing operation. Escape cancels an unfinished drag.

### Names and animation

Objects receive names such as `__rectangle_1`, `__circle_2`, and `__text_1`. A provided name overrides the automatic name. Names are unique by default on a slide; repeated names can still be entered deliberately for split/merge animation.

The **Animate across slides** checkbox controls the `__` prefix. Matching is exact and case sensitive. Duplicating a slide preserves names and creates new permanent object IDs. Pasting on the same slide assigns a distinct name; pasting onto a slide without that name preserves it.

Matching `__` names transform automatically. Plain names present in both frames use the default residual crossfade. Updater-generated frames replay their saved functions. Layer order, updater drafts, and appearance effects belong to each object’s Properties; the separate Animation, Layers, and Updaters tabs have been removed. Existing advanced connection data is preserved and remains available in project JSON.

**Frame Properties** appears when you click an empty canvas area, select a slide thumbnail, or press **Frame properties** at the top of the inspector. Set its **Time step to next frame** and hold duration here. Ordinary outgoing transitions adopt this time step. Changing the time step of an already committed updater saves the setting for a new generation; it does not silently retime the saved path.

**Construct / Destruct** in Object Properties applies only when the object appears or disappears between the two frames. Choices are Default (fade), None (cut), Fade, Grow/Shrink, four slide directions, four wipe directions, Spin & grow, and Pop. Set Effect starts/ends as percentages and choose Smooth, Linear, Ease in or Ease out. Matched objects ignore these settings. Repeated plain names pair existing instances by position; only extra instances receive entrance/exit effects. Repeated `__` names retain the split/merge convention.

### Keyboard shortcuts

| Action | Shortcut |
| --- | --- |
| Save / Open | Ctrl+S / Ctrl+O |
| Undo / Redo | Ctrl+Z / Ctrl+Y |
| Copy / Paste | Ctrl+C / Ctrl+V |
| Duplicate object on canvas, or slide in the slide strip | Ctrl+D |
| Select all objects | Ctrl+A |
| Delete selected objects, or the focused slide thumbnail | Delete or Backspace |
| Nudge selected objects | Arrow keys; Shift for 10 pixels |
| Play / pause | Space |
| Cancel creation or a drag | Escape |

Text inputs retain normal text-editing shortcuts while focused. Click a slide thumbnail before deleting a slide; click an object before deleting it. Deletion is undoable and does not ask for confirmation. Deleting the final slide creates a fresh blank canvas; Undo restores it. Deleting an extracted pixel object erases its footprint with the chosen flat background color.

## Work with imported images

**Import slides** turns PNG/JPG files into fitted keyframes. **Picture** inserts an editable PNG/JPG or SVG layer on the current slide.

To extract an object from a keyframe, choose **Extract region** and click a color region. Shift adds disconnected pieces; Alt subtracts. Add/erase brushes correct the mask. In **Properties**, click **Create object**, then edit the resulting object through Properties or canvas handles.

Selection compares colors with the clicked seed in perceptual color space and uses eight-connectivity. Tolerance and brush radius appear in Frame Properties; Extract region scrolls there automatically. The background picker identifies the flat color used to remove selected pixels from imported images. Moving a selected pixel object lifts it into an editable image object and removes its old footprint. Mask corrections remain available for these lifted pixel objects.

This works best for flat diagrams, equations, and text. Photographic background reconstruction is outside this version.

## Draw with mathematics

Coordinates and size fields accept arithmetic expressions such as `960+120*cos(pi/4)`.

**Explicit graphs:** choose `y = f(x)` and enter `y=sin(x)` or `exp(-x^2/8)*sin(2*x)`. Graph range is `xmin,xmax,ymin,ymax`, for example `-2*pi,2*pi,-1.5,1.5`.

**Implicit curves:** choose `f(x,y) = 0`, for example:

```text
x^2+y^2=4
x^4+y^4=1
(x^2+y^2-1)^3-x^2*y^3=0
(x^2+y^2-1)*(x^2+y^2-0.25)=0
```

Fill paints the region where the left side minus the right side is nonpositive. Otherwise the zero contour is outlined. Set a suitable graph range. Very small features and zero sets without a sign change may require a different expression or larger drawing.

Expressions support `+ - * / ^`, parentheses, `pi`, `e`, and common trigonometric, inverse trigonometric, exponential, logarithmic, square-root, rounding, and hyperbolic functions. Multiplication is explicit: use `2*x`. Angles in expressions are radians; the object rotation field is degrees.

**LaTeX:** the ribbon opens a preview studio. Enter a math fragment or choose one of ten templates. **Auto** tries fast Mathtext and falls back to full LaTeX; select **Fast Mathtext** or **Full LaTeX** explicitly when needed. Matrices, aligned derivations, cases, integrals and mathematical symbols are supported by installed TinyTeX/pdflatex with amsmath/amssymb and PyMuPDF. The optional backend was approved by the user; no package installation or Manim-environment change was made.

Preview runs off the UI thread, cancels stale requests, retains errors and invalid input, and enables Apply only for the current successful preview. Ctrl+Enter previews; Alt+Enter applies. Use Fit, Actual pixels or 200% to inspect the artwork; preview memory remains bounded to its viewport. Load or save source text directly from the studio. Formula input is a fragment rather than a full TeX document: preambles, file commands and shell escape are unsupported. A LaTeX compile is limited to 20 seconds.

Formula source, backend choice and transparent artwork are embedded in the project. Moving, resizing, reopening and exporting cached formulas do not require Python or LaTeX; constructing or editing them does. `FRAMEMORPH_MATH_PYTHON` selects a Python interpreter; otherwise the existing Anaconda base is used when available. See `OPTIONAL-BACKENDS.md`.

## Image studio

Select an image and click **Edit content...** or **Image filters / pixel formulas...** in Object Properties. For a fitted imported keyframe, **Frame Properties → Edit slide image...** lifts its background into an editable layer while preserving authored overlays. Slides with extracted raw-pixel objects require editing those objects separately first.

Crop uses original source pixels. Output width and height set the image and canvas-object size; optionally lock output aspect to the crop, with width driving height. Filters include grayscale, sepia, inversion, sharpening, edges and threshold, plus brightness, contrast, saturation, blur, flips and a detail-preserving tint. White tint preserves the original colours. Alpha-aware resize and blur avoid dark transparent-edge halos. Resizing/moving later remains available through canvas handles and Properties.

Original image pixels and the edit recipe survive save/reopen. Each Preview starts from that original, so blur and resize do not accumulate. Reset to original removes processing settings. Undo restores a previous edit. Reconstructing native SVG/code/formula content resets its raster edits.

Per-pixel maps run after filters, with all output channels reading the same pre-map pixel. RGBA and normalized x/y coordinates range from 0 to 1; output channels clamp to that range. Example syntax:

```text
(r,g,b,a):1-r,1-g,1-b,a
(r,g,b,a):g,b,r,a
(r,g,b,a):r,g,b,a*x
(r,g,b,a):clamp(1.2*r,0,1),g,b,a
```

Expressions include min, max, pow, atan2, clamp, sin/cos, abs, sqrt, log and exp. Invalid/non-finite output is rejected without altering the original. Processing runs in the preview worker; playback uses its cached result. Processed images are bounded to 8192 pixels per side and 16 megapixels.

Shift-select multiple objects and use **Combine images** for an alpha-composited overlay, horizontal row or vertical stack. Overlay follows layer order; rows/columns follow the object list. It creates one portable image object and replaces the selected objects in one undoable action.

## Source code and SVG

**More... → Source code block** opens an editor with Python, C++, JavaScript or plain text, dark/light themes, line numbers and font size. Code is rendered as artwork and never executed. Python uses its standard tokenizer; C++/JavaScript use lightweight lexical colouring. Load/save source files and edit the saved content later through Properties.

**Picture** accepts SVG files; **More... → SVG artwork** accepts SVG source directly. Source is embedded; installed PyMuPDF renders supported paths, fills, strokes and text to transparent artwork. The detail setting controls its raster resolution. External resources, scripts and foreignObject are unsupported; SVG filters/font rendering can differ from a browser. Cached artwork moves and exports without the optional backend.

## Modify an object with a function

Select an object. In **Properties → Map this object**, type a map and click **Apply update**. It changes the current object, preserves its permanent ID and name, and clears the function input after success. An invalid expression leaves the object and input untouched. Each successful apply is one undo action.

The prefix chooses the parser; no type selector is needed:

| Input | Result |
| --- | --- |
| `(x,y):x+1,y+2` | Move right 1 unit and up 2 units |
| `z:z+2` | Move right 2 units |
| `z:z*exp(i*pi/4)` | Rotate 45° counterclockwise about the object center |
| `(x,y):2*x,0.5*y` | Stretch horizontally and compress vertically |
| `(x,y):x+0.3*y,y` | Apply a shear matrix |
| `(x,y):x+0.35*sin(1.8*y),y` | Bend the geometry with a coordinate function |
| `z:z+0.2*z^2` | Apply a nonlinear complex map |
| `z:z+origin` | Move the object center to the scene center |
| `(x,y):x+origin_x,y+origin_y` | The equivalent point map |
| `(x,y):x+20*px,y` | Move right exactly 20 pixels |

**Every map starts in the object's current center coordinates.** X points right, Y points up; `z=x+i*y`. The default is 100 pixels per mathematical unit, adjustable through **Pixels per unit**. Thus `z:z+2` moves right 200 pixels at that setting. Width/height and ordinary position fields continue to use canvas pixels, with Y pointing down.

To author an animation: **duplicate the slide**, select and update whichever objects need new poses, then press **Play**. Several objects can be changed separately on the same duplicate. The map tool does not create slides. Successive maps compose in order, with a fresh center for each apply. Their recipe is saved so the incoming animation can deform the original artwork; duplicating again starts a fresh recipe for that next transition.

Mappings bake the resulting shape into an editable image object: position, size, rotation, tint, opacity, further maps, and undo remain available. Native text/formula construction on a nonlinearly mapped result is flattened; undo returns to the original construction. Original source slides stay editable. Extreme expansion, folds, singularities, clipping at the canvas border, and details finer than the deformation mesh can limit the result. An entirely off-canvas or empty result is rejected. Up to 64 sequential maps may be accumulated in one pose.

Complex expressions support `z`, `i`, `pi`, `e`, `sin`, `cos`, `tan`, `exp`, `log`/`ln`, `sqrt`, `conj`, `abs`, `arg`, `re`/`real`, and `im`/`imag`. Use explicit multiplication (`2*x`, `2*z`). Matrix operations can be entered directly as point expressions, for example `(x,y):a*x+b*y,c*x+d*y` with numerical coefficients in place of a, b, c, d.

### Built-in variables

**Variables and examples…** in Properties explains the coordinate conventions.

| Context | Available variables |
| --- | --- |
| Numeric property fields, in pixels | `scene.width`, `scene.height`, `scene.origin_x`, `scene.origin_y`, `scene.left`, `scene.right`, `scene.top`, `scene.bottom` |
| Selected object, in pixels | `object.x`, `object.y`, `object.width`, `object.height`, `object.center_x`, `object.center_y` |
| Object maps, in mathematical units | `scene.width`, `scene.height`, `object.width`, `object.height` |
| Scene center expressed in the map's local coordinates | `origin_x`, `origin_y`; aliases `scene.origin_x`, `scene.origin_y`; complex `origin` |
| Scene borders expressed in map coordinates | `scene.left`, `scene.right`, `scene.top`, `scene.bottom` |
| Object center measured from the scene center, X right / Y up | `center_x`, `center_y` |
| Unit helpers inside maps | `px` (one pixel), `pixels_per_unit`, `tau` (2π) |

For example, the ordinary X property `scene.origin_x-object.width/2` centers an object horizontally. A map `(x,y):x+origin_x,y+origin_y` moves the whole object to the scene center. Variables are evaluated when Apply commits the edit; they are not live constraints. Graph content and motion functions have their own x/y/t contexts and do not currently expose these property/map variables.

### Motion quality and playback

Preview uses a 60 Hz presentation timer. It prepares correspondence asynchronously before advancing the initial playback clock, polls completed frames before requesting another, and keeps rendering off the UI thread. Actual presentation rate depends on scene complexity and the display; export remains deterministic at the chosen frame rate.

Equivalent native constructions use their original transparent artwork for continuous position, size, angle, opacity, and color interpolation. This prevents a thin rotating object from collapsing or needlessly fading through a contour correspondence. Different shapes still use contour morphs, with denser boundary samples, row-binned CUDA rasterization, and 4×4 coverage antialiasing. Invalid self-intersecting correspondence is reported and falls back to the explicitly identified fade-through mode. Complex shape/text changes may still require component pairing or boundary anchors; this release does not claim complete Manim behavior.

## Generate motion from object updaters

This is a second authoring workflow alongside ordinary keyframes. An updater records the entire trajectory, even when the start and end images are identical.

1. Select an object on the source slide. Scroll its **Properties** to **Updater function**.
2. Enter a function, set pixels per unit, and click **Save updater**. Repeat for other objects as needed.
3. Open **Frame properties**. Set **Time step to next frame**, for example 2 seconds. Saved updater objects appear in this frame’s list; click an entry to edit its function.
4. Click **Create next slide from updaters** in **Frame properties**. It inserts one new endpoint slide after the source and commits the full continuous functions on the incoming transition. Press Play or export as usual.

**Only this Generate action executes draft updaters.** Ordinary New and Duplicate do not execute them. Duplicate creates a normal pose with cleared updater drafts. Objects without an updater keep their pose. Saved source functions remain available for editing; generated target slides start without active drafts. Generation is one undo action. If you generate again, it inserts a new endpoint and preserves existing later slides; Delete removes an unwanted previous result.

Accept `(x,y,t):new_x,new_y`, `(x,y):new_x,new_y`, or `z:new_z`. The coordinates are the original object-centered coordinates, using the chosen pixels per unit. Every requested frame is evaluated from the original pose, rather than repeatedly transforming the last frame. Scrubbing, reverse seeks, preview, and export therefore agree. These are deterministic parametric maps; stateful physics/integration callbacks are outside this version.

| Updater | Motion |
| --- | --- |
| `z:z+2*(exp(i*tau*t)-1)` | A complete circle with radius 2 units, returning to the initial pose |
| `(x,y,t):x+2*(cos(tau*t)-1),y+2*sin(tau*t)` | The same orbit as a point map |
| `z:z*exp(i*tau*t)` | One full counterclockwise rotation about the object center |
| `z:z+10*px*seconds` | Move right at 10 pixels per second |
| `(x,y,t):(1+t)*x,(1+t)*y` | Grow to twice the original scale |
| `z:z+0.2*t*z^2` | Continuous nonlinear deformation |

`t = elapsed_seconds / duration` always runs from 0 to 1. `duration` is the step length in seconds; `seconds = t*duration`. With a two-second step, `seconds=2*t`. There is no automatic multiplication of mapped coordinates by two. A function written in normalized time follows the same path more slowly when a longer duration is chosen; a seconds-based velocity travels farther during a longer step.

The updater must reproduce the source pose at `t=0`, preventing an initial jump. Hence the circular translation includes `cos(tau*t)-1`, rather than just `cos(tau*t)`. Representative times are preflighted, and every rendered point is checked for finite coordinates. The endpoint must remain nonempty and at least partly on the canvas. Motion outside the canvas is clipped. Regenerate from the source when changing a committed updater interval or altering its source artwork; ordinary manual target edits clear its incoming recipe and return to pose interpolation.

Expressions compile once during transition preparation. A conservative symbolic analysis recognizes maps affine in the spatial variables, even with time-dependent coefficients. Translation, rotation, scale, shear, and circular translation use a four-corner textured mesh. Spatially nonlinear functions use a denser bounded mesh. Complex functions, expensive formulas, large deformations, and many objects still have a cost; see the measured comparison in `verification-v0.4.md`.

Projects using the expanded features use format version 4 (ordinary older-style projects may still save version 3). Frame time steps use format version 3 to retain frame time steps and lifecycle effects. This build reads formats 1–4; their outgoing transition durations become frame time steps. A closed loop preserves exact original endpoint artwork and construction; other generated endpoints are embedded image objects with a saved function recipe.

## Measure performance before changing backends

Run `bash run.sh profile updater-project --samples 120 --width 960 --height 540 --output profile.json`. Choose another transition with `--transition 2` or full export resolution with `--width 1920 --height 1080`.

The report separates uninstrumented warm frame timings from a second CUDA-event run showing uploads, background, rasterization, clearing, compositing, output conversion, and download. It also reports actual copy byte counts and a pixel-preservation check. Cold first-frame time includes preparation, initial allocations, and first rendering. Profiling is off during ordinary editing/export. CUDA events do not establish hardware compute/memory utilization or a DRAM roofline; consult `PERFORMANCE.md` for the evidence and next-library proposal.

The compositor now clears and blends each lineage's occupied region, preserving overlapping split/merge coverage. A reference test compares this optimization with full-frame clearing across contours with holes, overlapping copies, clipped sprites, and warped meshes, requiring identical pixels.

### Motion functions

Existing advanced connection functions remain supported in project JSON. Their `t` runs from zero to one over the connection's timing. For new authoring in the editor, use the object's Updater function. Examples of legacy connection expressions:

```text
Extra Y offset: -160*sin(pi*t)
Scale: 1+0.08*sin(pi*t)
Extra rotation: 45*sin(pi*t)
Opacity: 1-0.35*sin(pi*t)^2
```

For seamless joins, offsets and extra rotation should return zero at both endpoints; scale and opacity should return one.

## Save, recovery, and export

Projects contain versioned JSON, copied images, masks, transparent artwork, original inserted assets, native construction, and animation recipes. Asset references are relative. Explicit saves commit their manifest after asset writes; export replaces its final filename only after encoding succeeds.

Autosave runs every minute when there are changes. Saved projects recover from their autosave when reopened. Untitled projects use separate folders under `recovery/`, named by their first slide ID, so separate sessions do not overwrite one another. The first release's existing recovery files remain in place. Use Open to open a recovery folder.

Undo retains up to 16 states within a 256 MiB budget for unique image buffers. Preview and export use the same renderer. Export supports cancellation, 1920×1080/60 fps H.264 MP4 through libx264, and lossless PNG sequences. Other aspect ratios are padded; MP4 dimensions must be even.

## Headless interface

Run these from Ubuntu in the application directory:

```bash
bash run.sh info
bash run.sh validate studio-project
bash run.sh frame studio-project --time 2.6 --output frame.png
bash run.sh render studio-project --output movie.mp4 --width 1920 --height 1080 --fps 60
bash run.sh render studio-project --output frames --png-sequence --fps 30
bash run.sh sample-motion fresh-motion-project
bash run.sh sample-toolkit fresh-toolkit-project
bash run.sh sample-properties fresh-properties-project
bash run.sh sample-updaters fresh-updater-project
bash run.sh profile updater-project --output profile.json
```

The legacy headless `generate` command remains for existing scripts; its staged-slide workflow has been removed from the editor. Use `--ffmpeg /path/to/ffmpeg` for a specific encoder.

## Build and scope

The package targets Ubuntu 24.04, WSLg, and this RTX 3050's CUDA architecture 8.6. The compiler is not needed to run it. Build with `bash scripts/build-local.sh`; set `FRAMEMORPH_CUDA_ARCH` for another NVIDIA architecture. Dependencies and checksum-verified CUDA tools are isolated under `/var/tmp/framemorph-*`. No Linux NVIDIA driver installation or Conda changes are performed.

The source separates project/timeline/render planning, drawing and formula construction, cached object editing, mathematical transformations, CUDA rendering, and FLTK interaction. Private runtime libraries and their notices are included. The existing FFmpeg installation remains external.

This is a 2D editor, with a raster-backed canvas and saved native construction. It does not provide full PowerPoint file compatibility, full Manim scripting, 3D cameras, audio editing, semantic OCR, or photo background reconstruction. See `verification-v0.8.md` for current checks and measurements, and `verification-v0.7.md` for the GPU cache experiment.
## GPU input caching

The renderer retains unchanged backgrounds and artwork on the GPU, with a 128 MiB input cache and a 128-entry limit. This bounds both device allocations and the image storage kept alive by the cache; cropped views count their full parent allocation. Temporary or externally owned memory bypasses caching. Geometry buffers and rendered output remain separate reusable allocations. Preview and export use the same renderer.

For comparison, `framemorph profile <project> --output profile.json --uncached` disables resident input caching. Reports include explicit upload bytes, cache hits/misses, resident input storage and warm frame timings. This diagnoses transfer costs; it does not establish whether the GPU is compute-bound or DRAM-bound. See `verification-v0.7.md` for the cache experiment and `verification-v0.8.md` for transparency continuity and current measurements.
