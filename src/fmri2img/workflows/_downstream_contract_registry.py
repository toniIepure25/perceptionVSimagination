from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fmri2img.workflows.audit_public_nod_shared_only_downstream_contract import (
    build_public_nod_shared_only_downstream_contract_audit,
)
from fmri2img.workflows.audit_shared_private_smoke_downstream_contract import (
    build_shared_private_smoke_downstream_contract_audit,
)


DownstreamAuditBuilder = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class DownstreamContractAuditRegistration:
    experiment_name: str
    bundle_family: str
    builder: DownstreamAuditBuilder


DOWNSTREAM_CONTRACT_AUDIT_REGISTRY: dict[str, DownstreamContractAuditRegistration] = {
    "public_nod_imagenet_run10_shared_only_smoke": DownstreamContractAuditRegistration(
        experiment_name="public_nod_imagenet_run10_shared_only_smoke",
        bundle_family="public_nod_imagenet_run10_shared_only_smoke",
        builder=build_public_nod_shared_only_downstream_contract_audit,
    ),
    "shared_private_smoke": DownstreamContractAuditRegistration(
        experiment_name="shared_private_smoke",
        bundle_family="shared_private_smoke",
        builder=build_shared_private_smoke_downstream_contract_audit,
    ),
}


def get_downstream_contract_audit_registration(experiment_name: str) -> DownstreamContractAuditRegistration | None:
    return DOWNSTREAM_CONTRACT_AUDIT_REGISTRY.get(experiment_name)


def list_registered_downstream_contract_audits() -> list[str]:
    return sorted(DOWNSTREAM_CONTRACT_AUDIT_REGISTRY)
