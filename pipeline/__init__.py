"""
Room Visualizer Pipeline Package
A hybrid multi-stage architectural visualizer combining semantic priors,
monocular 3D depth geometry, surface texture continuity, and photometric relighting
for both Floors and Walls.
"""

import cv2
import numpy as np
from .segmentation import SemanticSegmenter
from .depth_engine import DepthGeometryEngine
from .surface_analyzer import SurfaceTextureAnalyzer
from .occlusion_detector import OcclusionDetector
from .mask_refiner import MaskRefiner
from .perspective_engine import PerspectiveEngine
from .lighting_engine import LightingEngine
from .compositor import VisualizerCompositor
from .wall_engine import WallEngine

class RoomVisualizerPipeline:
    """
    Unified 10-stage architectural visualizer orchestrator for floors and walls.
    """
    def __init__(self, seg_model=None, seg_processor=None, depth_model=None, depth_processor=None, device="cpu"):
        self.device = device
        self.segmenter = SemanticSegmenter(model=seg_model, processor=seg_processor, device=device)
        self.depth_engine = DepthGeometryEngine(model=depth_model, processor=depth_processor, device=device)
        self.surface_analyzer = SurfaceTextureAnalyzer()
        self.occlusion_detector = OcclusionDetector()
        self.mask_refiner = MaskRefiner()
        self.perspective_engine = PerspectiveEngine()
        self.lighting_engine = LightingEngine()
        self.compositor = VisualizerCompositor()
        self.wall_engine = WallEngine()

    def process_room(self, image_bgr, tile_bgr=None, tile_config=None, wall_tile_bgr=None, wall_tile_config=None, generate_debug=False):
        """
        Executes the complete multi-stage pipeline on a room image.
        Supports simultaneous or independent Floor and Wall photorealistic surface synthesis.
        Returns:
          - 'clean_mask': Final refined boolean floor mask (H, W)
          - 'clean_wall_mask': Final refined boolean wall mask (H, W)
          - 'floor_quad': 4-corner perspective ground quad
          - 'wall_quads': List of 4-corner perspective wall quads
          - 'wall_planes': Partitioned physical wall planes
          - 'room_metrics': Estimated dimensions, sqft, and tile count
          - 'depth_map': Monocular depth map
          - 'result_bgr': Composite result if floor or wall tile is provided
          - 'debug_stages': Dictionary of visual debug stage images
        """
        H, W = image_bgr.shape[:2]
        debug_stages = {}

        # Stage 1: Semantic Priors (SegFormer)
        seg_map, rough_floor = self.segmenter.infer_semantic_map(image_bgr)
        if generate_debug:
            debug_stages["1_original"] = image_bgr.copy()
            debug_stages["2_raw_segformer"] = self.segmenter.visualize_mask(image_bgr, rough_floor)

        # Stage 2: Monocular 3D Depth
        depth_map = self.depth_engine.infer_depth(image_bgr)
        if generate_debug:
            debug_stages["4_depth_map"] = self.depth_engine.visualize_depth(depth_map)

        # Stage 3: Occlusion & Furniture Geometry Shielding (Strict Barrier)
        obstacle_mask = self.occlusion_detector.detect_obstacles(seg_map, depth_map, W, H)
        if generate_debug:
            debug_stages["6_occlusion_mask"] = self.occlusion_detector.visualize_obstacles(image_bgr, obstacle_mask)

        # Stage 4: Surface Continuity & Texture Analysis (Bounded by Obstacle Shield)
        surface_mask = self.surface_analyzer.analyze_floor_surface(image_bgr, rough_floor, depth_map, obstacle_mask)
        if generate_debug:
            debug_stages["3_surface_analysis"] = self.surface_analyzer.visualize_surface(image_bgr, surface_mask)

        # Stage 5: 3D Ground Plane Estimation
        ground_plane_info = self.depth_engine.fit_ground_plane(rough_floor, depth_map, W, H)
        if generate_debug:
            debug_stages["5_floor_plane"] = self.depth_engine.visualize_plane(image_bgr, ground_plane_info, rough_floor)

        # Stage 6: Multi-Signal Mask Fusion & Edge Refinement (Floor)
        final_floor_mask = self.mask_refiner.refine_mask(
            rough_floor=rough_floor,
            surface_mask=surface_mask,
            obstacle_mask=obstacle_mask,
            ground_plane_info=ground_plane_info,
            image_bgr=image_bgr,
            depth_map=depth_map,
            depth_engine=self.depth_engine
        )
        if generate_debug:
            debug_stages["7_refined_mask"] = self.mask_refiner.visualize_refined(image_bgr, final_floor_mask)

        # Stage 7: Perspective Ground Quad & Coordinated 3D Room-Box Geometry
        floor_quad = self.perspective_engine.compute_ground_quad(
            final_mask=final_floor_mask,
            ground_plane_info=ground_plane_info,
            W=W, H=H
        )

        # Stage 8: 3D Multi-Plane Wall Segmentation & Coordinated Room-Box Geometry
        clean_wall_mask = self.wall_engine.extract_wall_mask(seg_map, floor_mask=final_floor_mask, depth_map=depth_map, image_bgr=image_bgr)
        room_box = self.perspective_engine.compute_room_box_geometry(floor_quad, clean_wall_mask, W, H)
        wall_planes = self.wall_engine.partition_wall_planes(clean_wall_mask, depth_map, W, H, floor_quad=floor_quad, room_box=room_box)
        wall_quads = [p["quad"] for p in wall_planes] if wall_planes else [room_box["back_wall_quad"]]

        # Extract Guided Foreground Alpha Matte for Occluding Furniture, Openings & Decor
        fg_alpha = self.occlusion_detector.extract_foreground_matte(
            image_bgr=image_bgr,
            obstacle_mask=obstacle_mask,
            floor_mask=final_floor_mask,
            wall_mask=clean_wall_mask,
            seg_map=seg_map
        )

        # Room Dimension & Material Estimation
        room_metrics = self.wall_engine.estimate_room_metrics(floor_quad, clean_wall_mask, W, H)

        # Determine target surfaces
        has_floor = (tile_bgr is not None and (tile_config or {}).get("target_surface", "floor") != "wall")
        w_tile = wall_tile_bgr if wall_tile_bgr is not None else (tile_bgr if (tile_config or {}).get("target_surface") == "wall" else None)
        w_cfg = wall_tile_config if wall_tile_config is not None else (tile_config if (tile_config or {}).get("target_surface") == "wall" else {})
        has_wall = (w_tile is not None and np.any(clean_wall_mask))

        # Stage 8b: Suppress and smoothly blur small unidentified clutter on active surfaces only
        cleaned_image_bgr, clutter_mask = self.occlusion_detector.suppress_and_blur_clutter(
            image_bgr=image_bgr,
            seg_map=seg_map,
            floor_mask=final_floor_mask if has_floor else None,
            wall_mask=clean_wall_mask if has_wall else None
        )

        # Surface Synthesis & Compositing
        result_bgr = None
        composite_canvas = cleaned_image_bgr.copy()

        # Extract Ambient Luminance & Contact Shadow Maps from Original Photo
        lum_map = self.lighting_engine.extract_ambient_luminance(image_bgr)
        contact_sh = self.lighting_engine.extract_contact_shadow(image_bgr)
        floor_footprint = self.lighting_engine.extract_furniture_footprint(seg_map, final_floor_mask, is_wall=False, max_dist=12)
        wall_footprint = self.lighting_engine.extract_furniture_footprint(seg_map, clean_wall_mask, is_wall=True, max_dist=10)

        # 1. Floor Surface Synthesis
        f_ppm = 200.0
        if has_floor:
            f_cfg = tile_config or {}
            f_pattern = f_cfg.get("pattern", "grid")
            f_gw = f_cfg.get("grout_width", 0)
            f_gc = f_cfg.get("grout_color", (200, 200, 200))
            f_scale = f_cfg.get("scale", 1.0)
            f_tw = f_cfg.get("tile_width_mm", 600.0)
            f_th = f_cfg.get("tile_height_mm", 600.0)
            f_ppm = f_cfg.get("pixels_per_meter", 200.0)
            f_shadow = f_cfg.get("shadow_strength", 0.55)
            f_finish = f_cfg.get("finish", "satin")
            f_slab = bool(f_cfg.get("slab", False))

            f_tx, f_ty, f_pw, f_ph = self.perspective_engine.plan_tiling(
                floor_quad=floor_quad, W=W, H=H, ppm=f_ppm,
                tile_w_mm=f_tw, tile_h_mm=f_th, tile_scale=f_scale,
                is_wall=False
            )
            flat_floor = self.perspective_engine.build_surface(
                tile_bgr=tile_bgr, pw=f_pw, ph=f_ph, tx=f_tx, ty=f_ty,
                pattern=f_pattern, grout_w=f_gw, grout_col=f_gc,
                slab=f_slab
            )
            warped_floor = self.perspective_engine.warp_to_quad(flat_floor, floor_quad, W, H)
            if generate_debug:
                debug_stages["8_warped_flooring"] = warped_floor.copy()

            lighting_map = self.lighting_engine.extract_lighting_map(composite_canvas, final_floor_mask)
            ao_map = self.lighting_engine.compute_contact_ao(final_floor_mask, obstacle_mask)
            specular_map = self.lighting_engine.extract_specular_map(composite_canvas, final_floor_mask)

            composite_canvas = self.compositor.composite(
                room_bgr=composite_canvas,
                warped_material=warped_floor,
                floor_mask=final_floor_mask,
                lighting_map=lighting_map,
                shadow_strength=f_shadow,
                finish=f_finish,
                ao_map=ao_map,
                fg_alpha=fg_alpha,
                specular_map=specular_map,
                luminance_map=lum_map,
                contact_shadow_map=contact_sh,
                contact_footprint=floor_footprint
            )
            result_bgr = composite_canvas

        # 2. Wall Surface Synthesis
        w_tile = wall_tile_bgr if wall_tile_bgr is not None else (tile_bgr if (tile_config or {}).get("target_surface") == "wall" else None)
        w_cfg = wall_tile_config if wall_tile_config is not None else (tile_config if (tile_config or {}).get("target_surface") == "wall" else {})

        has_wall = False
        if w_tile is not None and np.any(clean_wall_mask):
            has_wall = True
            w_pattern = w_cfg.get("pattern", "grid")
            w_gw = w_cfg.get("grout_width", 0)
            w_gc = w_cfg.get("grout_color", (200, 200, 200))
            w_scale = w_cfg.get("scale", 1.0)
            w_tw = w_cfg.get("tile_width_mm", 600.0)
            w_th = w_cfg.get("tile_height_mm", 600.0)
            w_ppm = w_cfg.get("pixels_per_meter", 200.0)
            w_shadow = w_cfg.get("shadow_strength", 0.55)
            w_finish = w_cfg.get("finish", "satin")
            w_slab = bool(w_cfg.get("slab", False))

            for plane in (wall_planes or [{"mask": clean_wall_mask, "quad": wall_quads[0]}]):
                p_mask = plane["mask"]
                p_quad = plane["quad"]
                w_tx, w_ty, w_pw, w_ph = self.perspective_engine.plan_tiling(
                    floor_quad=p_quad, W=W, H=H, ppm=w_ppm,
                    tile_w_mm=w_tw, tile_h_mm=w_th, tile_scale=w_scale,
                    is_wall=True
                )
                flat_wall = self.perspective_engine.build_surface(
                    tile_bgr=w_tile, pw=w_pw, ph=w_ph, tx=w_tx, ty=w_ty,
                    pattern=w_pattern, grout_w=w_gw, grout_col=w_gc,
                    slab=w_slab
                )
                composite_canvas = self.wall_engine.warp_wall_material(
                    room_bgr=composite_canvas,
                    tile_surface=flat_wall,
                    dest_quad=p_quad,
                    mask_bool=p_mask,
                    shadow_strength=w_shadow,
                    finish=w_finish,
                    fg_alpha=fg_alpha,
                    luminance_map=lum_map,
                    contact_shadow_map=contact_sh,
                    contact_footprint=wall_footprint
                )
            result_bgr = composite_canvas

        # 3. Wall-Floor Transition Strip (Skirting Board)
        if has_floor or has_wall:
            composite_canvas, skirting_mask = self.wall_engine.generate_skirting_strip(
                composite_canvas,
                floor_mask=final_floor_mask,
                wall_mask=clean_wall_mask,
                ppm=f_ppm,
                height_cm=10.0,
                material="mdf_dark_grey",
                luminance_map=lum_map,
                seg_map=seg_map,
                has_floor=has_floor
            )
            # Re-apply foreground alpha matte to strictly guarantee occluding furniture sits in front of skirting
            if fg_alpha is not None:
                fg_a = fg_alpha[:, :, None] if fg_alpha.ndim == 2 else fg_alpha
                composite_canvas = composite_canvas * (1.0 - fg_a) + image_bgr.astype(np.float32) * fg_a
                composite_canvas = np.clip(composite_canvas, 0, 255).astype(np.uint8)

            # Strict preservation invariant: If surface was not targeted, keep original pixels bit-exact
            if not has_floor and final_floor_mask is not None:
                composite_canvas[final_floor_mask] = image_bgr[final_floor_mask]
            if not has_wall and clean_wall_mask is not None:
                composite_canvas[clean_wall_mask] = image_bgr[clean_wall_mask]

            result_bgr = composite_canvas

        if generate_debug and result_bgr is not None:
            debug_stages["9_final_composite"] = result_bgr.copy()

        return {
            "seg_map": seg_map,
            "clean_mask": final_floor_mask,
            "clean_wall_mask": clean_wall_mask,
            "floor_quad": floor_quad,
            "room_box": room_box,
            "wall_quads": wall_quads,
            "wall_planes": wall_planes,
            "fg_alpha": fg_alpha,
            "room_metrics": room_metrics,
            "depth_map": depth_map,
            "result_bgr": result_bgr,
            "debug_stages": debug_stages
        }
