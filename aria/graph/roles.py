"""General-purpose semantic role bindings derived from perception."""

from __future__ import annotations

import numpy as np

from aria.types import LegendInfo, ObjectNode, PartitionScene, RoleBinding, SceneRole


def infer_roles(
    objects: tuple[ObjectNode, ...],
    partition: PartitionScene | None,
    legend: LegendInfo | None,
) -> tuple[RoleBinding, ...]:
    bindings: list[RoleBinding] = []

    if partition is not None:
        bindings.append(RoleBinding(
            role=SceneRole.SEPARATOR,
            color=int(partition.separator_color),
            tags=(
                f"rows:{partition.n_rows}",
                f"cols:{partition.n_cols}",
                "uniform" if partition.is_uniform_partition else "non_uniform",
            ),
        ))

    if legend is not None:
        bindings.append(RoleBinding(
            role=SceneRole.LEGEND,
            bbox=legend.region_bbox,
            tags=(legend.edge, f"entries:{len(legend.entries)}"),
        ))
        for index, entry in enumerate(legend.entries):
            bindings.append(RoleBinding(
                role=SceneRole.LEGEND,
                color=int(entry.key_color),
                tags=(f"entry:{index}", f"maps_to:{entry.value_color}"),
            ))

    for obj in objects:
        if obj.size == 1:
            bindings.append(RoleBinding(
                role=SceneRole.MARKER,
                object_id=obj.id,
                bbox=_bbox_to_region(obj),
                color=int(obj.color),
                tags=("singleton",),
            ))
        if _is_frame_object(obj):
            bindings.append(RoleBinding(
                role=SceneRole.FRAME,
                object_id=obj.id,
                bbox=_bbox_to_region(obj),
                color=int(obj.color),
            ))

    return tuple(bindings)


def _bbox_to_region(obj: ObjectNode) -> tuple[int, int, int, int]:
    x, y, w, h = obj.bbox
    return (y, x, y + h - 1, x + w - 1)


def _is_frame_object(obj: ObjectNode) -> bool:
    mask = obj.mask
    rows, cols = mask.shape
    if rows < 3 or cols < 3:
        return False

    border = np.zeros_like(mask, dtype=np.bool_)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(mask[border]):
        return False
    return not np.any(mask[1:-1, 1:-1])
