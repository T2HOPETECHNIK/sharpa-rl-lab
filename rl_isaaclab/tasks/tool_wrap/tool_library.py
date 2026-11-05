'''
Procedural elongated tools: a handle (cylinder, capsule or flat box) along the tool's local +x, optionally with a
head at the +x end (hammer-like block or spatula-like blade). Each tool is one rigid body whose mass and centre of
mass come from per-shape densities, so a heavy head shifts the COM the way a real hammer does.

The tool id is written onto the root prim (TOOL_ID_ATTR) so the env can read back which tool each env received
from MultiAssetSpawnerCfg.
'''

from __future__ import annotations

import random
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdGeom

TOOL_ID_ATTR = "sharpa:tool_id"


@clone
def spawn_tool(
    prim_path: str,
    cfg: "ToolSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    root.CreateAttribute(TOOL_ID_ATTR, Sdf.ValueTypeNames.Int).Set(int(cfg.tool_id))

    material_path = None
    if cfg.physics_material is not None:
        material_path = f"{prim_path}/physics_material"
        cfg.physics_material.func(material_path, cfg.physics_material)

    length, r, ht = cfg.handle_length, cfg.handle_radius, cfg.handle_half_thickness
    # (name, prim type, attributes, translation, scale, density)
    shapes = []
    if cfg.handle_shape == "cylinder":
        shapes.append(("handle", "Cylinder", {"radius": r, "height": length, "axis": "X"}, (0.0, 0.0, 0.0), None, cfg.handle_density))
    elif cfg.handle_shape == "capsule":
        shapes.append(("handle", "Capsule", {"radius": r, "height": max(length - 2.0 * r, 1e-3), "axis": "X"}, (0.0, 0.0, 0.0), None, cfg.handle_density))
    elif cfg.handle_shape == "box":
        shapes.append(("handle", "Cube", {"size": 1.0}, (0.0, 0.0, 0.0), (length, 2.0 * r, 2.0 * ht), cfg.handle_density))
    else:
        raise ValueError(f"Unknown handle shape: {cfg.handle_shape}")
    hx, hy, hz = cfg.head_size
    if cfg.head == "block":
        shapes.append(("head", "Cube", {"size": 1.0}, (0.5 * (length + hx), 0.0, 0.0), (hx, hy, hz), cfg.head_density))
    elif cfg.head == "blade":
        # blade bottom flush with the handle bottom so the tool lies flat
        shapes.append(("head", "Cube", {"size": 1.0}, (0.5 * (length + hx), 0.0, -(ht - 0.5 * hz)), (hx, hy, hz), cfg.head_density))

    create_prim(f"{prim_path}/geometry", "Xform", stage=stage)
    for name, prim_type, attributes, pos, scale, density in shapes:
        path = f"{prim_path}/geometry/{name}"
        prim = create_prim(path, prim_type, translation=pos, scale=scale, attributes=attributes, stage=stage)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        if cfg.collision_props is not None:
            schemas.define_collision_properties(path, cfg.collision_props, stage=stage)
        schemas.define_mass_properties(path, sim_utils.MassPropertiesCfg(density=density), stage=stage)
        if material_path is not None:
            bind_physics_material(path, material_path, stage=stage)

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    return root


@configclass
class ToolSpawnerCfg(RigidObjectSpawnerCfg):
    func: Callable = spawn_tool

    tool_id: int = 0
    handle_shape: str = "cylinder"          # "cylinder" | "capsule" | "box"
    handle_length: float = 0.25
    handle_radius: float = 0.015            # round handles: radius; box: half width (local y)
    handle_half_thickness: float = 0.015    # box: half thickness (local z); round: equals handle_radius
    head: str = "none"                      # "none" | "block" | "blade"
    head_size: tuple[float, float, float] = (0.0, 0.0, 0.0)  # full extents along local x, y, z
    handle_density: float = 700.0
    head_density: float = 3000.0
    color: tuple[float, float, float] = (0.55, 0.38, 0.22)
    physics_material: sim_utils.RigidBodyMaterialCfg | None = None


def build_tool_specs(num_tools: int = 32, seed: int = 0) -> list[dict]:
    '''Deterministic list of tool parameter dicts (keys match ToolSpawnerCfg fields).'''
    rng = random.Random(seed)
    kinds = ["cylinder"] * 8 + ["capsule"] * 3 + ["box"] * 3 + ["hammer"] * 4 + ["spatula"] * 2
    specs = []
    for tool_id in range(num_tools):
        kind = kinds[tool_id % len(kinds)]
        spec = dict(tool_id=tool_id, handle_density=rng.uniform(500.0, 1200.0), head="none", head_size=(0.0, 0.0, 0.0))
        if kind in ("cylinder", "capsule", "hammer"):
            r = rng.uniform(0.010, 0.0225)
            spec.update(handle_shape="capsule" if kind == "capsule" else "cylinder", handle_radius=r, handle_half_thickness=r,
                        handle_length=rng.uniform(0.14, 0.24) if kind == "hammer" else rng.uniform(0.15, 0.30))
        else:
            hw, ht = rng.uniform(0.012, 0.020), rng.uniform(0.008, 0.013)
            spec.update(handle_shape="box", handle_radius=hw, handle_half_thickness=ht,
                        handle_length=rng.uniform(0.14, 0.22) if kind == "spatula" else rng.uniform(0.15, 0.28))
        if kind == "hammer":
            hz = rng.uniform(max(2.0 * spec["handle_radius"] + 0.004, 0.024), 0.036)
            spec.update(head="block", head_size=(rng.uniform(0.022, 0.032), rng.uniform(0.07, 0.11), hz),
                        head_density=rng.uniform(2500.0, 5000.0), color=(0.35, 0.35, 0.38))
        elif kind == "spatula":
            spec.update(head="blade", head_size=(rng.uniform(0.06, 0.09), rng.uniform(0.05, 0.08), rng.uniform(0.003, 0.005)),
                        head_density=2700.0, color=(0.7, 0.7, 0.72))
        else:
            spec["color"] = (rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.6), rng.uniform(0.1, 0.4))
        specs.append(spec)
    return specs


def tool_metadata(spec: dict) -> dict:
    '''Geometry the env needs per tool (all in the tool's local frame, metres).'''
    r, ht = spec["handle_radius"], spec["handle_half_thickness"]
    half_length = 0.5 * spec["handle_length"]
    rest_height = ht if spec["handle_shape"] == "box" else r
    if spec["head"] == "block":
        rest_height = max(rest_height, 0.5 * spec["head_size"][2])
    return {
        "radius": 0.5 * (r + ht) if spec["handle_shape"] == "box" else r,  # equivalent radius for the cage check
        "half_length": half_length,
        "rest_height": rest_height,
        # stretch of handle the hand may be placed over (keeps the 8 cm wide hand off the head)
        "grip_min": -half_length + 0.04,
        "grip_max": half_length - 0.04,
    }
