"""Verify appearance transfer on independent analytic colored cube fixtures."""
import importlib.util
from pathlib import Path
import numpy as np
import trimesh


spec = importlib.util.spec_from_file_location("budget_colors_check", Path(__file__).with_name("budget_preparation.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
original = trimesh.creation.box(extents=(2,2,2))
original.visual = trimesh.visual.ColorVisuals(mesh=original,
    vertex_colors=np.column_stack((((original.vertices+1)*127.5).round().astype(np.uint8), np.full(len(original.vertices),255,np.uint8))))
query = np.array([[-1,-1,-1],[1,1,1],[1,.25,-.25],[1.2,0,0]],np.float64)
mesh = trimesh.Trimesh(query.copy(),np.array([[0,1,2],[1,2,3]]),process=False)
before = mesh.vertices.copy(),mesh.faces.copy()
actual = tool.transfer_vertex_colors(original,mesh)
expected = np.column_stack((((query.clip(-1,1)+1)*127.5).round().astype(np.uint8),np.full(len(query),255,np.uint8)))
assert np.max(np.abs(actual.astype(int)-expected.astype(int))) <= 1
assert np.array_equal(mesh.vertices,before[0]) and np.array_equal(mesh.faces,before[1])
print("PASS: analytic corner/face/outside-nearest colors, alpha and unchanged geometry")
