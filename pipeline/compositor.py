"""
Stage 10: Physically Plausible Visualizer Compositor & 9-Stage Debug Visualizer
"""

import cv2
import numpy as np

class VisualizerCompositor:
    def __init__(self):
        pass

    def composite(self, room_bgr, warped_material, floor_mask, lighting_map=None, shadow_strength=0.55, finish="satin", ao_map=None, fg_alpha=None, specular_map=None, luminance_map=None, contact_shadow_map=None, contact_footprint=None):
        """
        Blends the perspective-warped material into the room photograph with physically plausible lighting,
        luminance modulation (0.4 + 0.6 * luminance_map), contact shadow recreation, natural specular reflections,
        and strict foreground object occlusion shield.
        """
        H, W = room_bgr.shape[:2]

        if luminance_map is not None:
            # Physically grounded luminance modulation: result = tile_texture * (0.4 + 0.6 * luminance_map)
            amb_mod = 0.40 + 0.60 * luminance_map[..., None]
            material_lit = warped_material.astype(np.float32) * amb_mod

            # Object contact shadows (bed, boxes, footwear) multiply-blended at 0.75 strength
            if contact_shadow_map is not None and contact_footprint is not None and np.any(contact_footprint > 0.01):
                fp = contact_footprint[..., None]
                c_mod = 0.25 + 0.75 * contact_shadow_map[..., None]
                contact_blended = warped_material.astype(np.float32) * c_mod
                material_lit = material_lit * (1.0 - fp) + contact_blended * fp

            if ao_map is not None:
                material_lit = material_lit * ao_map[:, :, None]

            combined_lighting = luminance_map
        else:
            # 1. Combine Ambient Lighting Map and Contact AO
            combined_lighting = lighting_map.copy() if lighting_map is not None else np.ones((H, W), dtype=np.float32)
            if ao_map is not None:
                combined_lighting = combined_lighting * ao_map

            # 2. Apply Lighting & Shadows to Warped Material
            effective_shadow = (1.0 - shadow_strength) + shadow_strength * combined_lighting
            material_lit = warped_material.astype(np.float32) * effective_shadow[:, :, None]

        # 3. Specular Response & Natural Reflection Modulation
        # Pure light sheen highlight reflecting ambient light sources (NOT old tile pattern!)
        highlight = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        if finish in ("glossy", "polished"):
            if specular_map is not None:
                spec = specular_map[:, :, None]
            else:
                spec = np.clip((combined_lighting - 0.75) / 0.25, 0.0, 1.0)[:, :, None]
            sheen = 0.22 * spec
            material_lit = material_lit * (1.0 - sheen) + highlight * sheen
        elif finish == "satin":
            if specular_map is not None:
                spec = specular_map[:, :, None]
            else:
                spec = np.clip((combined_lighting - 0.82) / 0.18, 0.0, 1.0)[:, :, None]
            sheen = 0.10 * spec
            material_lit = material_lit * (1.0 - sheen) + highlight * sheen

        # 4. Crisp Anti-Aliased Edge Feathering Blend (3x3)
        alpha_mask = cv2.GaussianBlur(floor_mask.astype(np.float32), (3, 3), 0)[:, :, None]
        composite = material_lit * alpha_mask + room_bgr.astype(np.float32) * (1.0 - alpha_mask)

        # 5. Strict Foreground Re-Composition (Layer 4)
        if fg_alpha is not None:
            fg_a = fg_alpha[:, :, None] if fg_alpha.ndim == 2 else fg_alpha
            composite = composite * (1.0 - fg_a) + room_bgr.astype(np.float32) * fg_a

        return np.clip(composite, 0, 255).astype(np.uint8)

    def build_debug_montage(self, debug_stages_dict):
        """
        Arranges all 9 debug stages into a clean 3x3 grid montage with headers.
        """
        stage_keys = [
            ("1_original", "1. Original Room Photo"),
            ("2_raw_segformer", "2. Raw SegFormer Candidate"),
            ("3_surface_analysis", "3. Surface & Texture Analysis"),
            ("4_depth_map", "4. Monocular Depth Map"),
            ("5_floor_plane", "5. 3D RANSAC Ground Plane"),
            ("6_occlusion_mask", "6. Occlusion & Furniture Barrier"),
            ("7_refined_mask", "7. Final Refined Floor Mask"),
            ("8_warped_flooring", "8. Perspective Warped Material"),
            ("9_final_composite", "9. Final Physical Composite")
        ]

        # Find target dimension
        first_img = next(iter(debug_stages_dict.values()))
        th, tw = 360, 480

        panels = []
        for key, title in stage_keys:
            if key in debug_stages_dict and debug_stages_dict[key] is not None:
                img = debug_stages_dict[key]
                resized = cv2.resize(img, (tw, th))
            else:
                resized = np.zeros((th, tw, 3), dtype=np.uint8)

            # Add title banner
            banner = np.zeros((40, tw, 3), dtype=np.uint8)
            cv2.putText(banner, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            panel = np.vstack([banner, resized])
            panels.append(panel)

        # 3x3 Grid
        row1 = np.hstack(panels[0:3])
        row2 = np.hstack(panels[3:6])
        row3 = np.hstack(panels[6:9])
        montage = np.vstack([row1, row2, row3])

        return montage
