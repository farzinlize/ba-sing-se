"""Seeded low-poly scenery, batched geometry, and final-scene sightline checks.

The model assembly and batching extend the vegetation comparison prototype.
Placement never modifies terrain heights or the selected camera.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json

import numpy as np

from .biomes import terrain_normals
from .config import PipelineConfig
from .terrain import Terrain
from .viewpoint import FRAME_LIMIT, NoSuitableViewError, Viewpoint, hill_is_visible


@dataclass(frozen=True)
class Instance:
    kind: str
    position: tuple[float, float, float]
    size: float
    rotation: float
    tint: float
    variant: int
    lod: int


@dataclass
class Scene:
    batches: list
    instances: list[Instance]
    seeds: dict[str, int]
    removed: int = 0

    def metadata(self) -> dict:
        records = [asdict(item) for item in self.instances]
        return {
            "enabled": True, "seeds": self.seeds, "counts": dict(sorted(Counter(i.kind for i in self.instances).items())),
            "instances": len(self.instances), "batches": len(self.batches),
            "triangles": sum(mesh.n_cells for mesh in self.batches),
            "removed_obstructions": self.removed,
            "placement_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "visibility": "final decoration meshes ray-tested against every selected summit",
        }


def scenery_seeds(config: PipelineConfig) -> dict[str, int]:
    # The first two child streams remain the existing terrain and camera streams.
    children = np.random.SeedSequence(config.seed).spawn(4)
    return {name: int(override if override is not None else children[index].generate_state(1, dtype=np.uint64)[0])
            for name, override, index in (("vegetation", config.vegetation_seed, 2),
                                           ("decoration", config.decoration_seed, 3))}


def _combine(parts):
    import pyvista as pv
    points, faces, colors, offset = [], [], [], 0
    for mesh, color in parts:
        mesh = mesh.triangulate()
        points.append(mesh.points)
        faces.append(mesh.faces.reshape(-1, 4)[:, 1:] + offset)
        colors.append(np.broadcast_to(color, (mesh.n_points, 3)))
        offset += mesh.n_points
    triangles = np.vstack(faces)
    mesh = pv.PolyData(np.vstack(points), np.column_stack((np.full(len(triangles), 3), triangles)))
    mesh.point_data["rgb"] = np.clip(np.vstack(colors), 0, 255).astype(np.uint8)
    return mesh


def _lobe(center, scale, rng, lod):
    import pyvista as pv
    mesh = pv.Icosphere(nsub=0 if lod else 1)
    mesh.points *= rng.uniform(0.88, 1.12, (mesh.n_points, 1))
    mesh.points = mesh.points * np.asarray(scale) + center
    return mesh


@lru_cache(maxsize=48)
def prototype(kind: str, variant: int, lod: int):
    """Reusable unit models; trunks/rocks extend below their terrain anchor."""
    import pyvista as pv
    names = ("pine", "oak", "birch", "shrub", "grass", "rock", "cloud")
    rng = np.random.default_rng(1000 + names.index(kind) * 20 + variant)
    if kind == "grass":
        points, faces = [], []
        for blade in range(6):
            angle = rng.uniform(0, 2 * np.pi)
            side = np.array([np.cos(angle), np.sin(angle), 0])
            base = rng.uniform(-0.12, 0.12, 3)
            base[2] = -0.08
            points.extend((base - side * 0.035, base + side * 0.035,
                           base + side * 0.15 + [0, 0, rng.uniform(0.4, 0.85)]))
            faces.extend((3, blade * 3, blade * 3 + 1, blade * 3 + 2))
        return _combine([(pv.PolyData(np.array(points), faces), [157, 171, 76])])
    if kind == "rock":
        parts = [(_lobe([0, 0, -0.18], [0.65, 0.5, 0.6], rng, lod), [157, 150, 133])]
        if variant:
            parts.append((_lobe([0.36, 0.12, -0.22], [0.45, 0.36, 0.55], rng, lod), [145, 144, 130]))
        return _combine(parts)
    if kind == "cloud":
        # Broad, shallow puffs instead of floating spherical balls.
        return _combine([(_lobe([offset, rng.uniform(-0.1, 0.1), rng.uniform(0, 0.15)],
                               [0.4, 0.25, 0.22 + 0.12 * (1 - abs(offset))], rng, 0), [237, 245, 249])
                         for offset in (-0.65, -0.3, 0, 0.3, 0.6)])
    if kind in ("shrub", "cloud"):
        color = np.array([102, 140, 66] if kind == "shrub" else [237, 245, 249])
        count = 2 if lod else 5
        parts = [(_lobe([rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2), 0.2],
                        [0.34, 0.3, 0.32], rng, lod), color * rng.uniform(0.94, 1.04)) for _ in range(count)]
        return _combine(parts)
    wood = [111, 87, 59] if kind != "birch" else [186, 179, 155]
    trunk = pv.Cylinder(center=(0, 0, 0.28), direction=(0, 0, 1), radius=0.02,
                        height=0.76, resolution=6)
    parts = [(trunk, wood)]
    if kind == "pine":
        # Offset, unequal whorls give branched silhouettes rather than one cone.
        tiers = 4 if lod else 7
        for tier in range(tiers):
            t = tier / tiers
            bottom, height = 0.15 + t * 0.66, 0.3 - t * 0.13
            crown = pv.Cone(center=(rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), bottom + height / 2),
                            direction=(0, 0, 1), height=height, radius=0.23 * (1 - t * 0.85),
                            resolution=7 if lod else 11)
            crown.points[:, :2] *= rng.uniform(0.8, 1.2, (crown.n_points, 1))
            parts.append((crown, np.array([59, 119, 64]) * rng.uniform(0.86, 1.12)))
    else:
        foliage = np.array([100, 153, 59] if kind == "oak" else [145, 177, 77])
        for j in range(3 if lod else 9):
            angle = j * 2.399 + rng.uniform(-0.3, 0.3)
            spread = rng.uniform(0.04, 0.18) * (1 if kind == "oak" else 0.65)
            center = np.array([np.cos(angle) * spread, np.sin(angle) * spread, rng.uniform(0.45, 0.75)])
            radius = rng.uniform(0.15, 0.21)
            parts.append((_lobe(center, [radius, radius * 0.9, radius * (1 if kind == "oak" else 1.15)], rng, lod),
                          foliage * rng.uniform(0.88, 1.1)))
            if not lod:
                start = np.array([0, 0, 0.35])
                delta = center - start
                parts.append((pv.Cylinder(center=(start + center) / 2, direction=delta,
                                          height=np.linalg.norm(delta), radius=0.01, resolution=5), wood))
    return _combine(parts)


def batch_instances(items: list[Instance], identifiers: list[int]):
    """Vectorized transformed copies: one actor per species/variant/detail level."""
    import pyvista as pv
    source = prototype(items[0].kind, items[0].variant, items[0].lod)
    size = np.array([item.size for item in items])
    angles = np.array([item.rotation for item in items])
    points = source.points[None, :, :] * size[:, None, None]
    x, y = points[..., 0].copy(), points[..., 1].copy()
    cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
    points[..., 0], points[..., 1] = x * cosine - y * sine, x * sine + y * cosine
    points += np.array([item.position for item in items])[:, None, :]
    triangles = source.faces.reshape(-1, 4)[:, 1:][None, :, :] + np.arange(len(items))[:, None, None] * source.n_points
    triangles = triangles.reshape(-1, 3)
    mesh = pv.PolyData(points.reshape(-1, 3), np.column_stack((np.full(len(triangles), 3), triangles)))
    tint = np.array([item.tint for item in items])
    mesh.point_data["rgb"] = np.clip(source["rgb"][None, :, :] * tint[:, None, None], 0, 255).astype(np.uint8).reshape(-1, 3)
    mesh.cell_data["instance_id"] = np.repeat(identifiers, source.n_cells)
    mesh.field_data["kind"] = [items[0].kind]
    return mesh


def _protect_summits(allowed, points, sizes, terrain, camera):
    allowed = allowed.copy()
    origin = np.asarray(camera.position)
    for row, col in camera.visible_hills:
        summit = np.array([terrain.x[col], terrain.y[row], terrain.heights[row, col]])
        vector = summit - origin
        t = np.clip(((points[:, :2] - origin[:2]) @ vector[:2]) / np.dot(vector[:2], vector[:2]), 0, 1)
        offset = np.linalg.norm(points[:, :2] - (origin[:2] + t[:, None] * vector[:2]), axis=1)
        z = origin[2] + t * vector[2]
        corridor = (offset < sizes * 0.75 + 3) & (z > points[:, 2] - sizes) & (z < points[:, 2] + sizes + 3)
        summit_buffer = np.linalg.norm(points[:, :2] - summit[:2], axis=1) < 22
        allowed &= ~(corridor | summit_buffer)
    return allowed


def _place(points, allowed, weights, sizes, count, rng, radius_factor, occupied=None):
    """Weighted sampling with footprint spacing; no overlapping tree crowns."""
    if not count:
        return []
    candidates = np.flatnonzero(allowed)
    priorities = -np.log(np.maximum(rng.random(len(candidates)), 1e-12)) / np.maximum(weights[candidates], 0.001)
    candidates = candidates[np.argsort(priorities)]
    bins = defaultdict(list)
    cell_size = 2 * max(float(sizes.max()) * radius_factor, max((r for _, r in occupied or []), default=0), 0.5)
    for position, radius in occupied or []:
        key = tuple(np.floor(np.asarray(position[:2]) / cell_size).astype(int))
        bins[key].append((np.asarray(position[:2]), radius))
    selected = []
    for i in candidates:
        xy, radius = points[i, :2], sizes[i] * radius_factor
        bx, by = np.floor(xy / cell_size).astype(int)
        neighbors = [item for x in range(bx - 1, bx + 2) for y in range(by - 1, by + 2) for item in bins[(x, y)]]
        if any(np.linalg.norm(xy - location) < radius + other_radius for location, other_radius in neighbors):
            continue
        selected.append(int(i))
        bins[(bx, by)].append((xy, radius))
        if len(selected) >= count:
            break
    return selected


def generate_instances(terrain: Terrain, camera: Viewpoint, config: PipelineConfig) -> list[Instance]:
    seeds = scenery_seeds(config)
    vegetation_streams = np.random.SeedSequence(seeds["vegetation"]).spawn(3)
    rock_stream, cloud_stream = np.random.SeedSequence(seeds["decoration"]).spawn(2)
    xx, yy = np.meshgrid(terrain.x, terrain.y)
    points = np.column_stack((xx.ravel(), yy.ravel(), terrain.heights.ravel()))
    distance = np.linalg.norm(points - camera.position, axis=1)
    slope = np.degrees(np.arccos(np.clip(terrain_normals(terrain)[..., 2], 0, 1))).ravel()
    base = max(float(terrain.heights.min()), terrain.sea_level)
    relative = (points[:, 2] - base) / max(float(terrain.heights.max()) - base, 1e-8)
    land = (points[:, 2] > terrain.sea_level + 1) & (distance > max(30, config.tree_height_max * 1.5))
    plant_land = land & (slope < 43) & (relative < 0.98)
    if config.snow_line is not None:
        plant_land &= points[:, 2] < config.snow_line
    area_hectares = config.extent ** 2 / 10000
    result = []
    # Clusters are independent of tree count and use physical map coordinates.
    cluster_rng = np.random.default_rng(np.random.SeedSequence([seeds["vegetation"], 17]))
    clusters = np.zeros(len(points))
    for cx, cy, radius in zip(cluster_rng.uniform(0, config.extent, 14), cluster_rng.uniform(0, config.extent, 14),
                              cluster_rng.uniform(0.05, 0.13, 14) * config.extent):
        clusters += np.exp(-((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) / (2 * radius ** 2))
    clusters = np.clip(clusters / max(np.quantile(clusters, 0.85), 1e-8), 0, 1)
    occupied = []
    for kind, stream, density in (("tree", vegetation_streams[0], config.tree_density),
                                  ("shrub", vegetation_streams[1], config.ground_cover_density * 6),
                                  ("grass", vegetation_streams[2], config.ground_cover_density * 18),
                                  ("rock", rock_stream, config.rock_density)):
        if density == 0:
            continue
        rng = np.random.default_rng(stream)
        if kind == "tree":
            sizes = rng.uniform(config.tree_height_min, config.tree_height_max, len(points))
            allowed, weights, radius = plant_land, (0.16 + clusters ** 3) * np.clip(1 - slope / 60, 0.1, 1), 0.43
        elif kind == "rock":
            sizes = rng.uniform(2, 9, len(points))
            sizes *= np.where(rng.random(len(points)) < 0.12, 2, 1)
            allowed, weights, radius = land & (slope > 15), 0.1 + (slope / 60) ** 2 + relative.clip(0, 1), 0.95
        else:
            sizes = rng.uniform(1.4, 3.3, len(points)) if kind == "shrub" else rng.uniform(0.5, 1.1, len(points))
            allowed = plant_land & (distance < (900 if kind == "shrub" else 350))
            weights, radius = 0.4 + clusters, 0.65
        allowed = _protect_summits(allowed, points, sizes, terrain, camera)
        chosen = _place(points, allowed, weights, sizes, int(area_hectares * density), rng, radius,
                        occupied if kind == "rock" else None)
        for i in chosen:
            species = kind
            if kind == "tree":
                weights_species = np.array([1.2 + relative[i] if s == "pine" else
                                            1.2 + clusters[i] if s == "oak" else 0.7 + clusters[i]
                                            for s in config.tree_species])
                species = str(rng.choice(config.tree_species, p=weights_species / weights_species.sum()))
            result.append(Instance(species, tuple(float(v) for v in points[i]), float(sizes[i]),
                                   float(rng.uniform(0, np.pi * 2)), float(rng.uniform(0.9, 1.08)),
                                   int(rng.integers(2)), int(distance[i] > 550)))
            if kind == "tree":
                occupied.append((points[i], sizes[i] * 0.43))
    if config.clouds:
        rng = np.random.default_rng(cloud_stream)
        forward = np.subtract(camera.focal_point, camera.position)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, camera.up)
        for _ in range(9):
            size = float(rng.uniform(90, 170))
            position = (float(rng.uniform(-0.7, 1.7) * config.extent), float(rng.uniform(-0.7, 1.7) * config.extent),
                        float(terrain.heights.max() + config.extent * 0.18 + size))
            delta = np.asarray(position) - camera.position
            half_height = (delta @ forward) * np.tan(np.radians(camera.field_of_view_degrees / 2))
            # Keep clouds wholly in the sky frame; avoid cropped, distracting near clouds.
            if (half_height <= 0 or abs(delta[2]) + size * 0.6 > 0.9 * half_height
                    or abs(delta @ right) + size * 1.2 > 0.9 * half_height * config.width / config.height):
                continue
            result.append(Instance("cloud", position, size, float(rng.uniform(0, 2 * np.pi)), 1.0, int(rng.integers(2)), 0))
    return result


def clear_summit_sightlines(batches, terrain: Terrain, camera: Viewpoint):
    """Remove whole obstructing instances using intersections with their actual triangles."""
    blocked = set()
    for mesh in batches:
        for row, col in camera.visible_hills:
            target = (terrain.x[col], terrain.y[row], terrain.heights[row, col])
            _, cell_ids = mesh.ray_trace(camera.position, target)
            blocked.update(int(v) for v in mesh.cell_data["instance_id"][cell_ids])
    cleaned = []
    for mesh in batches:
        keep = ~np.isin(mesh.cell_data["instance_id"], list(blocked))
        if np.all(keep):
            cleaned.append(mesh)
        elif np.any(keep):
            kind = str(mesh.field_data["kind"][0])
            mesh = mesh.extract_cells(keep).extract_surface(algorithm="dataset_surface")
            mesh.field_data["kind"] = [kind]
            cleaned.append(mesh)
    return cleaned, blocked


def build_scene(terrain: Terrain, camera: Viewpoint, config: PipelineConfig) -> Scene:
    if camera.pitch_degrees != 0 or camera.position[2] != camera.focal_point[2] or tuple(camera.up) != (0, 0, 1):
        raise NoSuitableViewError("Final scene requires a horizontal camera with Z up")
    if len(camera.visible_hills) < config.min_visible_hills or camera.ground_height >= terrain.heights.max():
        raise NoSuitableViewError("Final scene requires a lower hill and multiple visible summits")
    forward = np.subtract(camera.focal_point, camera.position)
    forward /= np.linalg.norm(forward)
    for row, col in camera.visible_hills:
        delta = np.array([terrain.x[col], terrain.y[row], terrain.heights[row, col]]) - camera.position
        depth = delta @ forward
        half_height = depth * np.tan(np.radians(camera.field_of_view_degrees / 2))
        if (depth <= 0 or abs(delta[2]) > FRAME_LIMIT * half_height + 1e-7
                or abs(delta @ np.cross(forward, camera.up)) > FRAME_LIMIT * half_height * config.width / config.height + 1e-7
                or not hill_is_visible(terrain, camera.row, camera.column, camera.position[2], row, col)):
            raise NoSuitableViewError("Final terrain does not preserve summit visibility/framing")
    items = generate_instances(terrain, camera, config)
    groups = defaultdict(list)
    for identifier, item in enumerate(items):
        groups[(item.kind, item.variant, item.lod)].append(identifier)
    batches = [batch_instances([items[i] for i in ids], ids) for ids in groups.values()]
    batches, removed = clear_summit_sightlines(batches, terrain, camera)
    return Scene(batches, [item for i, item in enumerate(items) if i not in removed], scenery_seeds(config), len(removed))
