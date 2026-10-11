"""Extract the pinned legacy search, changing only mesh/debug-file I/O."""
import ast

METHOD = "NVIDIA legacy estimate_metric_scale candidate search"
REVISION = "71fa7cbe46081467edadd11ab534b0c14aa9d913"


def adapted_search_source(source: str, debug_directory: str) -> str:
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "estimate_metric_scale")
    # Preserve the complete search loop, depth filtering, sample count, FP calls,
    # candidate ranges and Chamfer/IoU ranking. Replace the terminal UV-only OBJ
    # writer with a vertex-color-preserving writer for the actual input surface.
    class MeshIO(ast.NodeTransformer):
        changed_debug = 0
        changed_writer = 0

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == "debug_dir" for t in node.targets):
                self.changed_debug += 1
                node.value = ast.Constant(debug_directory)
            if any(isinstance(t, ast.Name) and t.id == "tex" for t in node.targets):
                return None
            return self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id in {"mesh_hy_orig", "tex"}:
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "save_obj":
                self.changed_writer += 1
                return ast.parse("_save_scaled_mesh(hy_file, mesh_out, best_scale)").body[0]
            return self.generic_visit(node)

    adapter = MeshIO()
    function = adapter.visit(function)
    if (adapter.changed_debug, adapter.changed_writer) != (1, 1):
        raise ValueError("Pinned legacy mesh I/O locations changed")
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))) + "\n"
