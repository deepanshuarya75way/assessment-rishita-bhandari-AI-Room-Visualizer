"""
Stage 9: Photometric Lighting & Shadow Preservation Engine
Decomposes original room illumination to extract ambient lighting, sunlight, and contact shadows.
"""

import cv2
import numpy as np

class LightingEngine:
    def __init__(self):
        pass

    def extract_lighting_map(self, room_bgr, floor_mask):
        """
        Extracts the smooth, normalized room lighting multiplier across the floor or active surface.
        Uses large-kernel morphological opening and Gaussian blurring to strip out high-frequency
        textures, seams, and clutter while preserving natural window illumination and room shadows.
        """
        H, W = room_bgr.shape[:2]
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)

        # Alternating morphological opening and closing to strip out old tile grout lines, borders, and textures
        k_open = max(15, (int(W * 0.025) | 1))
        open_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        gray_co = cv2.morphologyEx(gray, cv2.MORPH_OPEN, open_elem)
        gray_co = cv2.morphologyEx(gray_co, cv2.MORPH_CLOSE, open_elem)

        # Wide Gaussian blur to capture pure ambient room light field
        blur_k = max(31, (int(W * 0.08) | 1))
        gray_blurred = cv2.GaussianBlur(gray_co.astype(np.float32), (blur_k, blur_k), 0)

        floor_px = gray_blurred[floor_mask > 0] if np.any(floor_mask > 0) else []
        if len(floor_px) > 0:
            white_pt = float(np.percentile(floor_px, 94))
            shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
            shadow_map = np.clip(0.68 * shadow_map + 0.32, 0.32, 1.0)
        else:
            shadow_map = np.ones((H, W), dtype=np.float32)

        return shadow_map

    def extract_specular_map(self, room_bgr, active_mask):
        """
        Extracts natural specular reflection highlights from windows and ambient light sources
        present on glossy/polished surfaces in the original room.
        Strips out all high-frequency tile patterns and grout lines so that only true
        macro light source sheens are captured.
        """
        H, W = room_bgr.shape[:2]
        gray_u8 = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Strip high-frequency tile seams and patterns
        k_seam = max(9, int(W * 0.02) | 1)
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_seam, k_seam))
        co = cv2.morphologyEx(gray_u8, cv2.MORPH_OPEN, elem)
        co = cv2.morphologyEx(co, cv2.MORPH_CLOSE, elem)

        # 2. Smooth macro light field to capture window illumination zones without tile edges
        smooth_light = cv2.GaussianBlur(co.astype(np.float32), (0, 0), sigmaX=max(20.0, float(W) * 0.03))

        active_px = smooth_light[active_mask > 0] if np.any(active_mask > 0) else []
        if len(active_px) == 0:
            return np.zeros((H, W), dtype=np.float32)

        p80 = float(np.percentile(active_px, 80))
        p98 = max(float(np.percentile(active_px, 98)), p80 + 10.0)

        # 3. Specular sheen ramp from macro light field
        specular = np.clip((smooth_light - p80) / max(1.0, p98 - p80), 0.0, 1.0)
        specular = cv2.GaussianBlur(specular, (0, 0), sigmaX=max(10.0, float(W) * 0.015))
        specular[active_mask == 0] = 0.0

        return specular

    def compute_contact_ao(self, active_mask, obstacle_mask, radius_px=18, intensity=0.45):
        """
        Computes Contact Ambient Occlusion (AO) shadows under objects, furniture bases, and baseboards.
        Gives realistic physical grounding so tiles don't look 'flat' or 'floating'.
        """
        H, W = active_mask.shape[:2]
        
        # Source of occlusion shadows: obstacle borders and boundaries
        obs_u8 = (obstacle_mask > 0).astype(np.uint8)
        if not np.any(obs_u8 > 0):
            return np.ones((H, W), dtype=np.float32)

        # Distance transform from obstacles
        non_obs = (1 - obs_u8) * 255
        dist = cv2.distanceTransform(non_obs.astype(np.uint8), cv2.DIST_L2, 5)

        # Normalize distance within shadow radius
        rad = max(6.0, float(radius_px))
        ao_norm = np.clip(dist / rad, 0.0, 1.0)

        # Falloff curve (quadratic for soft, natural penumbra)
        ao = 1.0 - (1.0 - ao_norm) ** 1.8 * float(intensity)
        ao = np.clip(ao, 1.0 - intensity, 1.0).astype(np.float32)

        # Restrict AO to active floor/wall regions
        final_ao = np.ones((H, W), dtype=np.float32)
        final_ao[active_mask > 0] = ao[active_mask > 0]

        return final_ao

    def extract_ambient_luminance(self, room_bgr, sigma=None):
        """
        Extracts smooth, normalized macro ambient luminance field from original room photo (0.0 to 1.0).
        Strips out all high-frequency tile designs, grout lines, seams, and patterns via
        alternating morphological opening & closing, followed by adaptive macro Gaussian blurring,
        while strictly preserving macro lighting gradients, window-light falloff, and room illumination.
        """
        H, W = room_bgr.shape[:2]
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Alternating morphological filter to erase both dark seams (grout) and bright patterns
        k_seam = max(9, int(W * 0.02) | 1)
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_seam, k_seam))
        co = cv2.morphologyEx(gray, cv2.MORPH_OPEN, elem)
        co = cv2.morphologyEx(co, cv2.MORPH_CLOSE, elem)

        # 2. Adaptive wide Gaussian blur scaled to room geometry to capture pure ambient lighting field
        sigma_val = float(sigma) if (sigma is not None and sigma > 12.0) else max(25.0, float(W) * 0.035)
        luminance_map = cv2.GaussianBlur(co.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma_val)
        return luminance_map

    def extract_contact_shadow(self, room_bgr, sigma=None):
        """
        Extracts contact shadow intensity from original room photo while stripping out
        tile seams and high-frequency surface patterns from beneath furniture.
        """
        H, W = room_bgr.shape[:2]
        gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)

        # Close dark grout lines and seams inside shadow zones
        k_seam = max(9, int(W * 0.015) | 1)
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_seam, k_seam))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, elem)

        sigma_val = float(sigma) if (sigma is not None and sigma > 6.0) else max(8.0, float(W) * 0.012)
        contact_shadow = cv2.GaussianBlur(closed.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma_val)
        return contact_shadow

    def extract_furniture_footprint(self, seg_map, surface_mask, is_wall=False, max_dist=12):
        """
        Identifies the physical contact-shadow footprint directly beneath/adjacent to objects
        (bed, boxes, footwear on floor; clothes, whiteboard on wall) within 5-12px of contact edge.
        """
        H, W = surface_mask.shape[:2]
        if is_wall:
            casters = np.isin(seg_map, [22, 92, 100, 111, 127]).astype(np.uint8)
        else:
            casters = np.isin(seg_map, [7, 41, 115, 138, 142]).astype(np.uint8)

        if not np.any(casters > 0):
            return np.zeros((H, W), dtype=np.float32)

        dist = cv2.distanceTransform((1 - casters).astype(np.uint8), cv2.DIST_L2, 5)
        footprint_norm = np.clip(dist / float(max_dist), 0.0, 1.0)
        footprint_weight = (1.0 - footprint_norm) ** 1.5
        footprint_weight[surface_mask == 0] = 0.0
        footprint_weight[casters > 0] = 0.0
        return footprint_weight.astype(np.float32)

    def modulate_surface_lighting(self, warped_tile, luminance_map, contact_shadow_map=None, contact_footprint=None,
                                  ambient_floor=0.40, ambient_weight=0.60, contact_floor=0.25, contact_weight=0.75):
        """
        Multiply-blends original luminance into perspective tile texture:
        result = tile_texture * (ambient_floor + ambient_weight * luminance_map)
        Where furniture contact footprint is present, multiply-blends contact shadow at higher strength (weight 0.75).
        """
        amb_mod = float(ambient_floor) + float(ambient_weight) * luminance_map[..., None]
        blended = warped_tile.astype(np.float32) * amb_mod

        if contact_shadow_map is not None and contact_footprint is not None and np.any(contact_footprint > 0.01):
            fp = contact_footprint[..., None]
            c_mod = float(contact_floor) + float(contact_weight) * contact_shadow_map[..., None]
            # Deeper contact occlusion in furniture footprint
            contact_blended = warped_tile.astype(np.float32) * c_mod
            blended = blended * (1.0 - fp) + contact_blended * fp

        return np.clip(blended, 0, 255).astype(np.uint8)

