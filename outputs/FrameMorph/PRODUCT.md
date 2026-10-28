# Product direction

FrameMorph combines slide-based composition with precise mathematical animation.

The central workflow is **compose → duplicate → edit objects → animate**. Imported images and native objects share the same canvas. Each object's Properties panel provides direct geometry, appearance, and center-relative mathematical maps. Apply changes that object and clears the expression; it never creates a slide. Several objects can be edited separately on a duplicate. Names establish continuity between keyframes; correspondence, timing, paths, and functions make that continuity controllable.

The editor has one context-sensitive Properties inspector. Objects own their geometry, appearance, immediate maps, updater functions, layer order, and construct/destruct choices. Frames own hold duration, the time step to the next frame, and Create next slide from updaters. Ordinary adjacent `__` names transform automatically; lifecycle choices apply only to objects that appear or disappear. Saved updater paths take precedence when explicitly generated. New and Duplicate never execute draft updaters.

Visual controls should handle ordinary work directly. Expressions should add precision without forcing users to write programs. Drawing a circle from a toolbox, defining it by an implicit equation, or selecting it from an image should all produce an object that participates in the same animation workflow.

Preserve editability: keep original image assets, drawing parameters, text, equations, expressions, and explicit correspondence choices. Exported video is an output of the project, not the project itself.

Prioritize predictable results, useful recovery, exact keyframes, responsive preview, and understandable controls. Extend the application through shared authoring and animation interfaces rather than building disconnected effects.

The current delivery establishes the 2D authoring and CUDA rendering foundation. Future expansion can build on this model for reusable object libraries, richer constraints, camera tracks, audio, additional mathematical primitives, and more advanced scene scripting.
