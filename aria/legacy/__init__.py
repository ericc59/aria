"""Legacy ARC heuristic stack.

This subpackage contains the original family-based observation, refinement,
structural-edit, and offline-search modules.  They remain fully functional and
are re-exported at their original ``aria.*`` paths for backward compatibility.

New work should build on the generalized core (sketch, sketch_compile,
sketch_fit, graph, runtime, verify) rather than extending these modules.
"""
