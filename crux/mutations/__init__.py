"""
Mutations Module

Exports all mutations from all resource type modules.
"""

from . import storage
from . import vm
from . import network
from . import keyvault
from . import appservice
from . import containerregistry
from . import database
from . import loadbalancer

# Combine all mutations
ALL_MUTATIONS = (
    storage.ALL_MUTATIONS +
    vm.ALL_MUTATIONS +
    network.ALL_MUTATIONS +
    keyvault.ALL_MUTATIONS +
    appservice.ALL_MUTATIONS +
    containerregistry.ALL_MUTATIONS +
    database.ALL_MUTATIONS +
    loadbalancer.ALL_MUTATIONS
)

def get_all_mutations():
    """
    Get all available mutations.

    Returns:
        List of Mutation objects from all modules
    """
    return ALL_MUTATIONS


__all__ = [
    "storage",
    "vm",
    "network",
    "keyvault",
    "appservice",
    "containerregistry",
    "database",
    "loadbalancer",
    "ALL_MUTATIONS",
    "get_all_mutations",
]
