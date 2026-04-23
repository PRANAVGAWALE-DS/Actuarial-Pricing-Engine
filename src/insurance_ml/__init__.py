"""Insurance ML Pipeline - Zero-redundancy architecture"""

__version__ = "5.2.0"

# ✅ NO module-level imports of heavy dependencies
# Let each module handle its own imports when needed

# Only expose version and entry points
__all__ = [
    "__version__",
]
