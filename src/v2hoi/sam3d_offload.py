"""Stage residency for the unmodified NVIDIA SAM3D sampling/decoding methods.

Only parameter storage devices change. All checkpoint loads remain strict;
dtype, conditioning, sampler settings and output formats remain upstream.
"""
from contextlib import contextmanager
import torch
from v2d.sam3d.lib.inference_pipeline_modified import InferencePipelinePointMap as OfficialPipeline


class InferencePipelinePointMap(OfficialPipeline):
    def instantiate_and_load_from_pretrained(self, config, ckpt_path,
            state_dict_fn=None, state_dict_key="state_dict", device="cuda"):
        # Avoid allocating a temporary random GPU model before its checkpoint
        # is strictly loaded. The original loader and precision are unchanged.
        with torch.device("cpu"):
            return super().instantiate_and_load_from_pretrained(config, ckpt_path,
                state_dict_fn=state_dict_fn, state_dict_key=state_dict_key, device="cpu")

    @contextmanager
    def resident(self, *models):
        models = [m for m in models if m is not None]
        try:
            for model in models:
                model.to(self.device)
            yield
        finally:
            for model in models:
                model.cpu()
            torch.cuda.empty_cache()

    def compute_pointmap(self, image, pointmap=None):
        if pointmap is not None:
            return super().compute_pointmap(image, pointmap)
        previous = self.depth_model.device
        try:
            self.depth_model.device = self.device
            with self.resident(self.depth_model.model):
                return super().compute_pointmap(image, pointmap)
        finally:
            self.depth_model.device = previous

    def sample_sparse_structure(self, *args, **kwargs):
        with self.resident(self.models["ss_generator"], self.models["ss_decoder"],
                self.condition_embedders["ss_condition_embedder"]):
            return super().sample_sparse_structure(*args, **kwargs)

    def sample_slat(self, *args, **kwargs):
        with self.resident(self.models["slat_generator"],
                self.condition_embedders["slat_condition_embedder"]):
            return super().sample_slat(*args, **kwargs)

    def decode_slat(self, slat, formats=("mesh", "gaussian")):
        names = {"mesh": "slat_decoder_mesh", "gaussian": "slat_decoder_gs",
                 "gaussian_4": "slat_decoder_gs_4"}
        with self.resident(*(self.models[names[name]] for name in formats)):
            return super().decode_slat(slat, formats)
