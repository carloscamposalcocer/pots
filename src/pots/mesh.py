"""Field -> triangle mesh (slab-wise marching cubes), and mesh cleanup."""
import logging
import time

import numpy as np
import trimesh

log = logging.getLogger(__name__)

SLAB = 24             # z layers per marching-cubes slab (bounds peak memory)


def build(field, rmax, zmax, vox):
    """Mesh field < 0 inside the box |x|, |y| <= rmax, 0 <= z <= zmax (plus padding)."""
    from skimage.measure import marching_cubes

    pad = 2.0
    xs = np.arange(-rmax - pad, rmax + pad + vox, vox, dtype=np.float32)
    zs = np.arange(-pad, zmax + pad + vox, vox, dtype=np.float32)
    X2, Y2 = np.meshgrid(xs, xs, indexing="ij")
    n_slabs = (len(zs) - 2) // SLAB + 1 if len(zs) > 1 else 0
    log.info(
        "grid %d x %d x %d (rmax=%.1f mm, zmax=%.1f mm, voxel=%.3f mm, %d slabs)",
        len(xs), len(xs), len(zs), rmax, zmax, vox, n_slabs,
    )
    t0 = time.perf_counter()
    verts, faces, off, k0, slab = [], [], 0, 0, 0
    while k0 < len(zs) - 1:
        k1 = min(k0 + SLAB, len(zs) - 1)
        zz = zs[k0:k1 + 1]
        slab += 1
        log.debug("slab %d/%d  z=%.1f-%.1f mm", slab, n_slabs, float(zz[0]), float(zz[-1]))
        X = np.repeat(X2[:, :, None], len(zz), 2)
        Y = np.repeat(Y2[:, :, None], len(zz), 2)
        Z = np.broadcast_to(zz, X.shape).astype(np.float32)
        F = field(X, Y, Z).astype(np.float32)
        if F.min() < 0 < F.max():
            v, f, _, _ = marching_cubes(F, 0.0)
            v[:, 2] += k0                     # stay in index space -> exact seams
            verts.append(v); faces.append(f + off); off += len(v)
            log.debug("  marching cubes: %s verts, %s faces", f"{len(v):,}", f"{len(f):,}")
        else:
            log.debug("  skipped empty slab")
        k0 = k1
    if not verts:
        raise RuntimeError("field never crossed zero; nothing to mesh")
    m = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    # merge in index space first, scale after: this is what makes the slab
    # seams watertight
    m.merge_vertices(digits_vertex=3)
    m.vertices = m.vertices * vox + np.array([xs[0], xs[0], zs[0]])
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    log.info("meshed %s faces in %.1f s", f"{len(m.faces):,}", time.perf_counter() - t0)
    return m


def decimate(mesh, target_faces):
    import fast_simplification
    n0 = len(mesh.faces)
    if n0 <= target_faces:
        return mesh
    v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces,
                                        target_reduction=1 - target_faces / n0)
    return trimesh.Trimesh(v, f)


def clean(mesh, target_faces):
    """Decimate, drop floating specks, make watertight, sit on z = 0."""
    import pymeshfix
    t0 = time.perf_counter()
    n0 = len(mesh.faces)
    if n0 > target_faces:
        log.info("decimating %s -> %s faces", f"{n0:,}", f"{target_faces:,}")
        mesh = decimate(mesh, target_faces)
        log.info("decimated to %s faces", f"{len(mesh.faces):,}")
    else:
        log.info("skipping decimation (%s faces <= budget %s)", f"{n0:,}", f"{target_faces:,}")
    parts = mesh.split(only_watertight=False)
    mesh = max(parts, key=lambda p: len(p.faces))
    log.info("kept largest of %d components (%s faces)", len(parts), f"{len(mesh.faces):,}")
    log.info("repairing mesh")
    fix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    fix.repair(joincomp=False, remove_smallest_components=True)
    mesh = trimesh.Trimesh(fix.points, fix.faces)
    trimesh.repair.fix_normals(mesh)
    mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])
    log.info("cleaned in %.1f s", time.perf_counter() - t0)
    return mesh
