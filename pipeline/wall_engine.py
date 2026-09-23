"""
Stage: 3D Multi-Plane Wall Detection, Occlusion Shielding, Perspective Warping & Dimension Estimation
"""

import cv2
import numpy as np

# Structural & major furniture objects that must NOT be tiled over on walls
WALL_OCCLUDER_IDS = {
    3,   # floor
    5,   # ceiling
    7,   # bed
    8,   # windowpane
    9,   # window
    10,  # cabinet / credenza / wardrobe
    14,  # door
    15,  # table / desk
    18,  # curtain
    19,  # chair
    22,  # painting / art / picture frame / poster
    23,  # sofa / couch
    24,  # shelf / bookcase
    27,  # mirror
    30,  # armchair
    31,  # seat / bench
    44,  # chest of drawers
    49,  # countertop
    50,  # desk
    57,  # headboard
    64,  # coffee table
    65,  # sink / washbasin
    69,  # bookcase
    75,  # swivel chair
    80,  # bathtub
    35,  # television / screen
    41,  # box / storage crate / trunk
    89,  # television / screen
    92,  # apparel / clothing / hanging clothes / coats / jackets
    100, # poster / bulletin board / wall picture
    111, # bulletin board / board
    115, # bag / handbag / backpack
    127, # whiteboard / blackboard / signboard
    128, # refrigerator
    138, # footwear / shoes
    142, # shoe / footwear
}

class WallEngine:
    def __init__(self):
        pass

    def extract_wall_mask(self, seg_map, floor_mask=None, depth_map=None, image_bgr=None):
        """
        Extracts clean, occlusion-free wall mask.
        Enforces strict mutual exclusivity with floor_mask and preserves doors,
        windows, paintings, TVs, ceilings, and all foreground furniture.
        Preserves hanging clothes, boxes, footwear, and wall-mounted objects (whiteboard, hooks).
        """
        H, W = seg_map.shape[:2]

        # Raw wall class (ADE20K ID 0 = wall)
        wall_raw = (seg_map == 0).astype(np.uint8) * 255
        if not np.any(wall_raw > 0):
            return np.zeros((H, W), dtype=bool)

        # Build strict wall occluder mask with protected structural objects only
        occluders = np.isin(seg_map, list(WALL_OCCLUDER_IDS)).astype(np.uint8) * 255

        # Preserve wall-mounted objects (whiteboard, hung bags/hangers, switchboards)
        if image_bgr is not None:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            s_chan = hsv[:, :, 1]
            v_chan = hsv[:, :, 2]
            b_chan, g_chan, r_chan = image_bgr[:, :, 0], image_bgr[:, :, 1], image_bgr[:, :, 2]

            # Whiteboard/signboard/paper: achromatic white region on wall (low saturation, high value, neutral BGR)
            is_wb = (s_chan < 26) & (v_chan > 175) & (np.abs(r_chan.astype(int) - b_chan.astype(int)) <= 15) & (np.abs(g_chan.astype(int) - b_chan.astype(int)) <= 15)
            # Distinct wall-hung items (pouches, hangers) with distinct hue/chroma from ambient wall
            is_hung = (hsv[:, :, 0] >= 42) & (hsv[:, :, 0] <= 145) & (s_chan > 60)

            cand_mounted = (is_wb | is_hung) & (wall_raw > 0)
            num_m, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(cand_mounted.astype(np.uint8))
            for lbl in range(1, num_m):
                area_m = stats_m[lbl, cv2.CC_STAT_AREA]
                y_m = stats_m[lbl, cv2.CC_STAT_TOP]
                h_m = stats_m[lbl, cv2.CC_STAT_HEIGHT]
                w_m = stats_m[lbl, cv2.CC_STAT_WIDTH]
                if 50 < area_m < (W * H * 0.15) and y_m > (H * 0.12) and (y_m + h_m) < (H * 0.92) and w_m < (W * 0.45):
                    occluders[labels_m == lbl] = 255

        # Strict Mutual Exclusivity: Zero overlap between floor and wall
        if floor_mask is not None:
            occluders[floor_mask > 0] = 255

        # Subtract occluders
        wall_clean = wall_raw.copy()
        wall_clean[occluders > 0] = 0

        # Regularize the wall-floor baseboard junction (remove neural network scallop ripples)
        if floor_mask is not None and np.any(wall_clean > 0) and np.any(floor_mask > 0):
            ys_w, xs_w = np.where(wall_clean > 0)
            if len(xs_w) > 20:
                bot_w = {}
                for px, py in zip(xs_w, ys_w):
                    if px not in bot_w or py > bot_w[px]:
                        bot_w[px] = py
                cols = sorted(bot_w.keys())
                if len(cols) > 15:
                    vx = np.array(cols, dtype=np.float32)
                    vy = np.array([bot_w[c] for c in cols], dtype=np.float32)
                    A = np.column_stack([vx, np.ones(len(vx), dtype=np.float32)])
                    m_b, c_b = np.linalg.lstsq(A, vy, rcond=None)[0]
                    # Snap boundary points close to baseline to eliminate wavy scalloping
                    for c in cols:
                        y_line = int(round(m_b * c + c_b))
                        curr_bot = bot_w[c]
                        if abs(curr_bot - y_line) < 14 and 0 <= y_line < H:
                            if y_line > curr_bot:
                                wall_clean[curr_bot:min(y_line + 1, H), c] = 255
                            elif y_line < curr_bot:
                                wall_clean[y_line + 1:curr_bot + 1, c] = 0

        # Morphological Clutter Hole Healing on wall surface (micro-voids only, max 200px)
        cnts_w, hier_w = cv2.findContours(wall_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_w is not None:
            max_hole_area = min(200, int(W * H * 0.0003))
            for i, c in enumerate(cnts_w):
                if hier_w[0][i][3] != -1:  # Internal hole enclosed by wall
                    if cv2.contourArea(c) < max_hole_area:
                        cv2.drawContours(wall_clean, [c], -1, 255, thickness=-1)

        # Enforce strict protection of all structural occluders and floor mutual exclusivity
        wall_clean[occluders > 0] = 0
        if floor_mask is not None:
            wall_clean[floor_mask > 0] = 0

        # Retain significant connected wall structures
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_clean, connectivity=8)
        clean_mask = np.zeros((H, W), dtype=bool)
        min_wall_area = W * H * 0.002

        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_wall_area:
                clean_mask[labels == lbl] = True

        return clean_mask

    def partition_wall_planes(self, wall_mask, depth_map, W, H, floor_quad=None, room_box=None):
        """
        Partitions the wall mask into discrete 3D physical wall planes with slope-aware perspective quads.
        Unifies all fronto-parallel back wall segments (split by paintings, lamps, sofas) into a single
        continuous architectural feature wall plane with seamlessly aligned tiling.
        """
        if not np.any(wall_mask):
            return []

        wall_u8 = (wall_mask.astype(np.uint8) * 255)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_u8, connectivity=8)

        # Baseline coordinates from floor quad if available
        if floor_quad and len(floor_quad) >= 4:
            P_FL, P_FR = floor_quad[0], floor_quad[1]
            back_quad_default = [
                [float(min(0.0, P_FL[0])), 0.0],
                [float(max(float(W), P_FR[0])), 0.0],
                [float(max(float(W), P_FR[0])), float(P_FR[1])],
                [float(min(0.0, P_FL[0])), float(P_FL[1])]
            ]
        else:
            ys_w, xs_w = np.where(wall_mask)
            top_y = float(ys_w.min()) if len(ys_w) else 0.0
            bot_y = float(ys_w.max()) if len(ys_w) else float(H * 0.6)
            back_quad_default = [
                [0.0, 0.0],
                [float(W), 0.0],
                [float(W), bot_y],
                [0.0, bot_y]
            ]

        back_wall_mask = np.zeros((H, W), dtype=bool)
        planes = []

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < (W * H * 0.002):
                continue

            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]
            wall_h = float(h)

            comp_mask = (labels == lbl)

            # Analyze component bottom edge slope and perspective baseline
            c_ys, c_xs = np.where(comp_mask)
            col_bots = {}
            for px, py in zip(c_xs, c_ys):
                if px not in col_bots or py > col_bots[px]:
                    col_bots[px] = py
            
            c_cols = sorted(col_bots.keys())
            if len(c_cols) > 10:
                c_vx = np.array(c_cols, dtype=np.float32)
                c_vy = np.array([col_bots[col] for col in c_cols], dtype=np.float32)
                A = np.column_stack([c_vx, np.ones(len(c_vx), dtype=np.float32)])
                comp_slope, comp_intercept = np.linalg.lstsq(A, c_vy, rcond=None)[0]
                comp_slope = float(np.clip(comp_slope, -0.65, 0.65))
            else:
                comp_slope, comp_intercept = 0.0, float(y + h)
            
            x1 = float(x)
            x2 = float(x + w)
            y_bl = float(np.clip(comp_slope * x1 + comp_intercept, 0, H))
            y_br = float(np.clip(comp_slope * x2 + comp_intercept, 0, H))
            # If component spans across room center or has substantial width, it is part of the Back Wall
            is_broad = (w >= W * 0.35) or (x1 < W * 0.40 and x2 > W * 0.60)

            if comp_slope > 0.09 and (x2 < W * 0.45) and not is_broad:
                plane_name = "Left Wall"
                plane_id = "left"
                h_near = wall_h * 1.25
                h_far = wall_h * 0.82
                quad = [
                    [x1, float(max(0.0, y_bl - h_near))],
                    [x2, float(max(0.0, y_br - h_far))],
                    [x2, y_br],
                    [x1, y_bl]
                ]
                if (h_near > 30) and (h_far > 30) and (w > 30):
                    planes.append({
                        "id": f"{plane_id}_{len(planes)+1}",
                        "name": plane_name,
                        "mask": comp_mask,
                        "quad": quad,
                        "bbox": [x, y, w, h],
                        "area_px": int(area)
                    })
                else:
                    back_wall_mask |= comp_mask
            elif comp_slope < -0.09 and (x1 > W * 0.55) and not is_broad:
                plane_name = "Right Wall"
                plane_id = "right"
                h_far = wall_h * 0.82
                h_near = wall_h * 1.25
                quad = [
                    [x1, float(max(0.0, y_bl - h_far))],
                    [x2, float(max(0.0, y_br - h_near))],
                    [x2, y_br],
                    [x1, y_bl]
                ]
                if (h_near > 30) and (h_far > 30) and (w > 30):
                    planes.append({
                        "id": f"{plane_id}_{len(planes)+1}",
                        "name": plane_name,
                        "mask": comp_mask,
                        "quad": quad,
                        "bbox": [x, y, w, h],
                        "area_px": int(area)
                    })
                else:
                    back_wall_mask |= comp_mask
            else:
                # Accumulate into unified back wall
                back_wall_mask |= comp_mask

        # Add unified back wall plane if any components exist
        if np.any(back_wall_mask):
            ys_b, xs_b = np.where(back_wall_mask)
            bbox_b = [int(xs_b.min()), int(ys_b.min()), int(xs_b.max() - xs_b.min()), int(ys_b.max() - ys_b.min())]
            planes.insert(0, {
                "id": "back_wall_1",
                "name": "Back Wall",
                "mask": back_wall_mask,
                "quad": back_quad_default,
                "bbox": bbox_b,
                "area_px": int(np.count_nonzero(back_wall_mask))
            })

        return planes

    def compute_wall_quads(self, wall_mask, floor_quad=None, W=1024, H=768):
        """
        Generates fallback perspective quad for vertical wall planes with aligned baseline.
        """
        if not np.any(wall_mask):
            return [[float(W * 0.05), float(H * 0.05)], [float(W * 0.95), float(H * 0.05)],
                    [float(W * 0.95), float(H * 0.65)], [float(W * 0.05), float(H * 0.65)]]

        ys, xs = np.where(wall_mask)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        if floor_quad and len(floor_quad) >= 2:
            P_FL = floor_quad[0]
            P_FR = floor_quad[1]
            far_dx = max(1.0, P_FR[0] - P_FL[0])
            m = (P_FR[1] - P_FL[1]) / far_dx
            c = P_FL[1] - m * P_FL[0]
            y_bl = float(m * x_min + c)
            y_br = float(m * x_max + c)
            h = max(y_max - y_min, H * 0.3)
            return [
                [x_min, max(0.0, y_bl - h)],
                [x_max, max(0.0, y_br - h)],
                [x_max, y_br],
                [x_min, y_bl]
            ]

        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]

    def warp_wall_material(self, room_bgr, tile_surface, dest_quad, mask_bool, shadow_strength=0.55, finish="satin", fg_alpha=None, luminance_map=None, contact_shadow_map=None, contact_footprint=None):
        """
        Applies photorealistic perspective homography warping and illumination preservation to walls.
        Respects foreground object occlusion so decor/doors/windows remain on top.
        """
        H, W = room_bgr.shape[:2]
        ph, pw = tile_surface.shape[:2]

        if not np.any(mask_bool):
            return room_bgr

        # 1. Perspective Homography
        src_pts = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
        dst_pts = np.array(dest_quad, dtype=np.float32)
        if len(dst_pts) != 4 or cv2.contourArea(dst_pts) < 150.0:
            return room_bgr
        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(tile_surface, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        except Exception:
            return room_bgr

        # 2. Lighting & Shadow Modulation
        if luminance_map is not None:
            # Physically grounded luminance modulation: result = tile_texture * (0.4 + 0.6 * luminance_map)
            amb_mod = 0.40 + 0.60 * luminance_map[..., None]
            material_lit = warped.astype(np.float32) * amb_mod

            # Contact shadows for wall-mounted objects (clothes, whiteboard, etc.)
            if contact_shadow_map is not None and contact_footprint is not None and np.any(contact_footprint > 0.01):
                fp = contact_footprint[..., None]
                c_mod = 0.25 + 0.75 * contact_shadow_map[..., None]
                contact_blended = warped.astype(np.float32) * c_mod
                material_lit = material_lit * (1.0 - fp) + contact_blended * fp
        else:
            gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            blur_k = int(W * 0.06) | 1
            gray_blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

            wall_px = gray_blurred[mask_bool]
            if len(wall_px) > 0:
                white_pt = float(np.percentile(wall_px, 92))
                shadow_map = np.clip(gray_blurred / max(white_pt, 1.0), 0.0, 1.0)
                shadow_map = np.clip(0.65 * shadow_map + 0.35, 0.0, 1.0)
            else:
                shadow_map = np.ones((H, W), dtype=np.float32)

            light_gain = (1.0 - shadow_strength) + shadow_strength * shadow_map
            material_lit = warped.astype(np.float32) * light_gain[:, :, None]

        # Specular Highlight for Glossy/Polished Marble Walls (Pure luminous highlight, NOT old wall pattern)
        if finish in ("glossy", "polished"):
            ref_map = luminance_map if luminance_map is not None else (cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
            specular = np.clip((ref_map - 0.78) / 0.22, 0.0, 1.0)[:, :, None]
            sheen = 0.16 * specular
            highlight = np.array([255.0, 255.0, 255.0], dtype=np.float32)
            material_lit = material_lit * (1.0 - sheen) + highlight * sheen

        # 3. Crisp Anti-Aliased Edge Feathering Blend (3x3)
        alpha_mask = cv2.GaussianBlur(mask_bool.astype(np.float32), (3, 3), 0)[:, :, None]

        # 3b. Architectural Baseboard Crease AO (Contact Shadow along bottom baseline)
        dist_to_edge = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
        crease_ao = np.clip(dist_to_edge / 6.0, 0.0, 1.0)
        crease_factor = 0.85 + 0.15 * crease_ao[:, :, None]
        material_lit = material_lit * crease_factor

        composite = material_lit * alpha_mask + room_bgr.astype(np.float32) * (1.0 - alpha_mask)

        # 4. Strictly re-composite foreground objects on top if fg_alpha is provided
        if fg_alpha is not None:
            fg_a = fg_alpha[:, :, None] if fg_alpha.ndim == 2 else fg_alpha
            composite = composite * (1.0 - fg_a) + room_bgr.astype(np.float32) * fg_a

        return np.clip(composite, 0, 255).astype(np.uint8)

    def estimate_room_metrics(self, floor_quad, wall_mask, W, H, ppm=200.0):
        """
        Estimates real-world room metrics including floor area (sq meters), perimeter,
        wall area, and recommends tile box quantities with 10% wastage.
        """
        if floor_quad is None or len(floor_quad) != 4:
            return {}

        pts = np.array(floor_quad, dtype=np.float32)
        # 2D polygon area in pixel space
        px_area = 0.5 * np.abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
        
        # Scale to real-world metric meters
        area_m2 = max(1.0, float(px_area / (ppm * ppm)))
        perimeter_m = float(cv2.arcLength(pts, True) / ppm)

        wall_px_count = int(np.count_nonzero(wall_mask)) if wall_mask is not None else 0
        wall_area_m2 = float(wall_px_count / (ppm * ppm)) if wall_px_count > 0 else (perimeter_m * 2.8 * 0.6)

        tile_size_m2 = 0.6 * 0.6 # standard 600x600mm
        floor_tiles_needed = int(np.ceil((area_m2 * 1.10) / tile_size_m2))
        wall_tiles_needed = int(np.ceil((wall_area_m2 * 1.10) / tile_size_m2))

        return {
            "floor_area_m2": round(area_m2, 2),
            "wall_area_m2": round(wall_area_m2, 2),
            "perimeter_m": round(perimeter_m, 2),
            "floor_boxes_needed": int(np.ceil(floor_tiles_needed / 4)),
            "wall_boxes_needed": int(np.ceil(wall_tiles_needed / 4)),
            "recommended_wastage_pct": 10
        }

    def generate_skirting_strip(self, image_bgr, floor_mask, wall_mask, ppm=200.0, height_cm=10.0, material="mdf_dark_grey", luminance_map=None, seg_map=None, has_floor=True):
        """
        Inserts a perspective-aligned skirting board (baseboard) along the floor-wall boundary contour.
        - Height: 8-12 cm scaled via door reference (ppm) with perspective depth tapering
        - Material: Matte dark grey MDF with top bullnose bevel highlight
        - Contact shadows: 1-2px soft-edged shadow directly above on wall side, and below on floor side.
        - Multiply-blends original room luminance so skirting reacts naturally to lighting.
        - Strictly shields structural objects (door, bed, box, window, hanging clothes) to guarantee 0px diff.
        - Strictly shields floor pixels: skirting board sits on the wall base and NEVER cuts into floor.
        - If has_floor is False (only wall tiles applied), floor is strictly 100% untouched.
        """
        H, W = image_bgr.shape[:2]
        if floor_mask is None or not np.any(floor_mask > 0) or wall_mask is None or not np.any(wall_mask > 0):
            return image_bgr, np.zeros((H, W), dtype=bool)

        ppm_val = float(ppm) if ppm and ppm > 1.0 else (W / 4.5)
        strip_h = max(6, int(round((float(height_cm) / 100.0) * ppm_val)))

        fm_u8 = (floor_mask > 0).astype(np.uint8)
        wm_u8 = (wall_mask > 0).astype(np.uint8)

        # 1. Contact interface strictly where wall directly meets floor
        dil_fm = cv2.dilate(fm_u8, np.ones((5, 5), np.uint8))
        contact = (dil_fm > 0) & (wm_u8 > 0)

        ys_c, xs_c = np.where(contact)
        if len(xs_c) < 10:
            return image_bgr, np.zeros((H, W), dtype=bool)

        # Per-column contact point (the bottom edge of the wall meeting the floor)
        col_pts = {}
        for x, y in zip(xs_c, ys_c):
            if x not in col_pts or y > col_pts[x]:
                col_pts[x] = y

        contact_cols = np.array(sorted(col_pts.keys()))
        contact_ys = np.array([col_pts[c] for c in contact_cols], dtype=np.float32)

        # Baseline fitted strictly to the true wall-floor contact zone
        if len(contact_cols) >= 5:
            m_base, c_base = np.polyfit(contact_cols, contact_ys, 1)
        else:
            m_base, c_base = 0.0, float(contact_ys[0])

        span_min = int(contact_cols.min())
        span_max = int(contact_cols.max())
        span_cols = np.arange(span_min, span_max + 1)
        y_fit = m_base * span_cols + c_base

        y_min = float(np.min(y_fit))
        y_max = float(np.max(y_fit))

        # Build skirting polygon strip with perspective depth scaling
        skirting_mask = np.zeros((H, W), dtype=np.uint8)
        bevel_mask = np.zeros((H, W), dtype=np.uint8)

        for idx_c, c in enumerate(span_cols):
            y_base = int(round(y_fit[idx_c]))
            if y_base < 0 or y_base >= H or c < 0 or c >= W:
                continue
            depth_scale = 0.70 + 0.30 * (float(y_base) - y_min) / max(1.0, y_max - y_min)
            col_strip_h = max(5, int(round(strip_h * depth_scale)))

            # Sits at bottom of wall, rising up into the wall
            y_bot = y_base
            y_top = max(0, y_bot - col_strip_h)

            skirting_mask[y_top:y_bot + 1, c] = 255
            bevel_mask[y_top:min(y_top + 2, y_bot), c] = 255

        # Horizontal closure for architectural continuity
        skirting_mask = cv2.morphologyEx(skirting_mask, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
        bevel_mask = cv2.bitwise_and(bevel_mask, skirting_mask)

        # STRICT ARCHITECTURAL FLOOR PROTECTION:
        # A skirting board is an architectural wall element attached to the base of the wall.
        # It strictly never overwrites floor pixels!
        skirting_mask[fm_u8 > 0] = 0
        bevel_mask[fm_u8 > 0] = 0

        # Structural protection shield (door, bed, box, clothes, window)
        structural_shield = None
        if seg_map is not None:
            structural_shield = np.isin(seg_map, [7, 8, 9, 14, 22, 41, 92, 100, 111, 115, 127, 138, 142])
            skirting_mask[structural_shield] = 0
            bevel_mask[structural_shield] = 0

        # 1. Base Material: Matte dark grey MDF skirting (BGR: 44, 46, 48)
        skirting_canvas = np.zeros_like(image_bgr, dtype=np.float32)
        skirting_canvas[:] = (44.0, 46.0, 48.0)
        skirting_canvas[bevel_mask > 0] = (78.0, 80.0, 84.0)

        # 2. Modulate with room luminance if provided
        if luminance_map is not None:
            lum_mod = 0.40 + 0.60 * luminance_map[..., None]
            skirting_canvas = skirting_canvas * lum_mod

        # 3. Contact shadows:
        # Above strip on wall side (always applies if wall is active)
        top_dil = cv2.dilate(skirting_mask, np.ones((3, 3), np.uint8))
        top_shadow_zone = ((top_dil > 0) & (skirting_mask == 0) & (wm_u8 > 0)).astype(np.float32)

        # Below strip on floor side:
        # ONLY applies if new floor material was applied (has_floor is True)!
        # If user only applied wall tiles, floor must remain 100% bit-exact untouched!
        if has_floor:
            bot_dil = cv2.dilate(skirting_mask, np.ones((3, 3), np.uint8))
            bot_shadow_zone = ((bot_dil > 0) & (skirting_mask == 0) & (fm_u8 > 0)).astype(np.float32)
        else:
            bot_shadow_zone = np.zeros((H, W), dtype=np.float32)

        if structural_shield is not None:
            top_shadow_zone[structural_shield] = 0.0
            bot_shadow_zone[structural_shield] = 0.0

        top_shadow_blur = cv2.GaussianBlur(top_shadow_zone, (3, 3), 0)
        bot_shadow_blur = cv2.GaussianBlur(bot_shadow_zone, (3, 3), 0)

        out_img = image_bgr.astype(np.float32).copy()
        out_img = out_img * (1.0 - 0.28 * top_shadow_blur[..., None])
        if has_floor:
            out_img = out_img * (1.0 - 0.32 * bot_shadow_blur[..., None])

        strip_alpha = (skirting_mask > 0).astype(np.float32)[..., None]
        out_img = skirting_canvas * strip_alpha + out_img * (1.0 - strip_alpha)

        if structural_shield is not None:
            out_img[structural_shield] = image_bgr[structural_shield]

        return np.clip(out_img, 0, 255).astype(np.uint8), (skirting_mask > 0)
