"""
Mutations Module

Exports all mutations from storage, VM, and network modules.
"""

from . import storage
from . import vm
from . import network

# Combine all mutations
ALL_MUTATIONS = (
    storage.ALL_MUTATIONS +
    vm.ALL_MUTATIONS +
    network.ALL_MUTATIONS
)

__all__ = ["storage", "vm", "network", "ALL_MUTATIONS"]
