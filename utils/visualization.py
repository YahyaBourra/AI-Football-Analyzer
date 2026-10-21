"""
Visualization Utilities Module

Drawing helpers for the annotated output video (all colors are BGR, as used by OpenCV):
- Team-colored ellipses under the players' feet, bounding boxes and track IDs (GK / REF tags)
- Floating player tags with the current smoothed speed (km/h)
- Ball marker (filled = detected, hollow = interpolated) and a marker above the ball holder
- Picture-in-Picture (PiP) tactical map in the top-right corner
- Team heatmaps on the tactical map (TeamHeatmap) and team convex hulls (controlled area)
- Scoreboard overlay with Ball Possession %, pass counts and a top-distance leaderboard
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ColorScheme:
    """
    BGR color scheme for visualization.
    """
    team_a: Tuple[int, int, int] = (0, 0, 255)  # Red
    team_b: Tuple[int, int, int] = (255, 0, 0)  # Blue
    referee: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    unknown: Tuple[int, int, int] = (160, 160, 160)  # Gray
    ball: Tuple[int, int, int] = (0, 255, 0)  # Green
    holder: Tuple[int, int, int] = (0, 215, 255)  # Gold

    def for_team(self, team: str) -> Tuple[int, int, int]:
        return getattr(self, team, self.unknown)

    def as_dict(self) -> Dict[str, Tuple[int, int, int]]:
        return {'team_a': self.team_a, 'team_b': self.team_b, 'referee': self.referee}


FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_player(frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int], label: str) -> None:
    """
    Draw a player: thin bounding box, ellipse under the feet and an ID tag (in place).
    """
    x1, y1, x2, y2 = bbox
    cx, w = (x1 + x2) // 2, x2 - x1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.ellipse(frame, (cx, y2), (max(w // 2, 8), max(w // 6, 3)), 0, -45, 235, color, 2, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    tx1, ty1 = cx - tw // 2 - 4, y2 + 8
    cv2.rectangle(frame, (tx1, ty1), (tx1 + tw + 8, ty1 + th + 6), color, -1)
    cv2.putText(frame, label, (tx1 + 4, ty1 + th + 2), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_triangle_marker(frame: np.ndarray, tip: Tuple[int, int], color: Tuple[int, int, int],
                         size: int = 10, filled: bool = True) -> None:
    """
    Draw a downward-pointing triangle whose tip is at `tip` (in place).
    """
    x, y = tip
    pts = np.array([[x, y], [x - size, y - 2 * size], [x + size, y - 2 * size]], dtype=np.int32)
    if filled:
        cv2.drawContours(frame, [pts], 0, color, -1)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 1)
    else:
        cv2.drawContours(frame, [pts], 0, color, 2, cv2.LINE_AA)


class TeamHeatmap:
    """
    Accumulates where each team's players are on the pitch and renders it as a heatmap.

    Positions (pitch meters) are binned into a `cell_m` grid per team; the density is
    smoothed with a Gaussian (`sigma_m`) and drawn on the tactical map in the team color,
    with opacity proportional to the time spent in each area.
    """

    def __init__(self, tactical_map, teams=('team_a', 'team_b'), cell_m: float = 1.0, sigma_m: float = 2.5):
        self.tm = tactical_map
        self.cell_m = cell_m
        self.sigma_cells = sigma_m / cell_m
        self.shape = (int(np.ceil(tactical_map.dims.width / cell_m)), int(np.ceil(tactical_map.dims.length / cell_m)))
        self.grids = {t: np.zeros(self.shape, np.float32) for t in teams}

    def add(self, team: str, position: Tuple[float, float]) -> None:
        if team not in self.grids:
            return
        col = int(np.clip(position[0] / self.cell_m, 0, self.shape[1] - 1))
        row = int(np.clip(position[1] / self.cell_m, 0, self.shape[0] - 1))
        self.grids[team][row, col] += 1

    def density(self, team: str) -> np.ndarray:
        """Smoothed density normalized to [0, 1], at map resolution (pitch area only)."""
        d = cv2.GaussianBlur(self.grids[team], (0, 0), self.sigma_cells)
        d = d / d.max() if d.max() > 0 else d
        size = (int(self.tm.dims.length * self.tm.scale), int(self.tm.dims.width * self.tm.scale))
        return cv2.resize(d, size, interpolation=cv2.INTER_LINEAR)

    def render(self, colors: ColorScheme, max_alpha: float = 0.7) -> np.ndarray:
        """Pitch template with both teams' heatmaps blended in their team colors."""
        img = self.tm.pitch_template.astype(np.float32)
        m = self.tm.margin
        for team in self.grids:
            d = self.density(team)
            alpha = (max_alpha * d ** 0.7)[..., None]
            region = img[m:m + d.shape[0], m:m + d.shape[1]]
            region[:] = region * (1 - alpha) + np.array(colors.for_team(team), np.float32) * alpha
        out = img.astype(np.uint8)
        out[self.tm.pitch_template == 255] = 255  # Keep the pitch markings visible on top
        return out

    def save(self, path_prefix: str) -> List[str]:
        """Save one classic (JET colormap) heatmap image per team; returns the file paths."""
        paths, m = [], self.tm.margin
        for team in self.grids:
            img = self.tm.pitch_template.copy()
            d = self.density(team)
            jet = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
            alpha = (0.75 * np.clip(d * 1.5, 0, 1))[..., None]
            region = img[m:m + d.shape[0], m:m + d.shape[1]].astype(np.float32)
            img[m:m + d.shape[0], m:m + d.shape[1]] = (region * (1 - alpha) + jet * alpha).astype(np.uint8)
            img[self.tm.pitch_template == 255] = 255
            name = 'Team A' if team == 'team_a' else 'Team B'
            cv2.putText(img, f"{name} heatmap", (30, 55), FONT, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            path = f"{path_prefix}_heatmap_{team}.png"
            cv2.imwrite(path, img)
            paths.append(path)
        return paths


_RADAR_CACHE: Dict[Tuple[int, int, int, float, float], np.ndarray] = {}


def draw_pitch_radar(width: int, height: int, pad: int = 12,
                     pitch_dims: Tuple[float, float] = (105.0, 68.0)) -> np.ndarray:
    """
    Draw a 2D top-down football pitch template for the live minimap/radar using OpenCV.
    Cached by size so per-frame radar rendering only copies the static pitch array.
    """
    pitch_l, pitch_w = pitch_dims
    key = (width, height, pad, pitch_l, pitch_w)
    if key in _RADAR_CACHE:
        return _RADAR_CACHE[key].copy()

    canvas = np.full((height, width, 3), (28, 95, 32), dtype=np.uint8)
    iw, ih = width - 2 * pad, height - 2 * pad

    def pt(xm: float, ym: float) -> Tuple[int, int]:
        return (pad + int(round((xm / pitch_l) * iw)), pad + int(round((ym / pitch_w) * ih)))

    for i in range(10):
        x0, y0 = pt(i * (pitch_l / 10.0), 0.0)
        x1, y1 = pt((i + 1) * (pitch_l / 10.0), pitch_w)
        color = (38, 128, 42) if i % 2 == 0 else (32, 112, 36)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)

    white = (240, 240, 240)
    cv2.rectangle(canvas, pt(0, 0), pt(pitch_l, pitch_w), white, 2, cv2.LINE_AA)
    cv2.line(canvas, pt(pitch_l / 2, 0), pt(pitch_l / 2, pitch_w), white, 1, cv2.LINE_AA)

    center = pt(pitch_l / 2, pitch_w / 2)
    rx = max(int(round((9.15 / pitch_l) * iw)), 4)
    ry = max(int(round((9.15 / pitch_w) * ih)), 4)
    cv2.ellipse(canvas, center, (rx, ry), 0, 0, 360, white, 1, cv2.LINE_AA)
    cv2.circle(canvas, center, 3, white, -1, cv2.LINE_AA)

    cy = pitch_w / 2.0
    half_angle = float(np.degrees(np.arccos((16.5 - 11.0) / 9.15)))
    for side in (0, 1):
        sign = 1.0 if side == 0 else -1.0
        gx = 0.0 if side == 0 else pitch_l
        cv2.rectangle(canvas, pt(gx, cy - 40.32 / 2), pt(gx + sign * 16.5, cy + 40.32 / 2), white, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, pt(gx, cy - 18.32 / 2), pt(gx + sign * 5.5, cy + 18.32 / 2), white, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, pt(gx - sign * 1.6, cy - 7.32 / 2), pt(gx, cy + 7.32 / 2), white, 1, cv2.LINE_AA)
        spot = pt(gx + sign * 11.0, cy)
        cv2.circle(canvas, spot, 2, white, -1, cv2.LINE_AA)
        base_ang = 0.0 if side == 0 else 180.0
        cv2.ellipse(canvas, spot, (rx, ry), base_ang, -half_angle, half_angle, white, 1, cv2.LINE_AA)

    _RADAR_CACHE[key] = canvas
    return canvas.copy()


def draw_minimap(frame: np.ndarray, player_positions: List[Dict],
                 ball_position: Optional[Tuple[float, float]] = None,
                 colors: Optional[ColorScheme] = None,
                 passes: Optional[List[Dict]] = None,
                 hulls: Optional[Dict[str, np.ndarray]] = None,
                 ball_trail: Optional[List[Tuple[float, float]]] = None,
                 message: Optional[str] = None,
                 width_ratio: float = 0.28, margin: int = 15, alpha: float = 0.88,
                 pitch_dims: Tuple[float, float] = (105.0, 68.0)) -> np.ndarray:
    """
    Render a live 2D Minimap (Radar) directly onto the top-right corner of `frame` (in place).

    Scales 2D pitch coordinates (meters, 0..105 x 0..68) into the minimap dimensions and plots:
    - Team A players in Red, Team B players in Blue, Referees in Yellow, Goalkeepers as squares
    - Active ball holder highlighted with a Gold halo ring
    - Ball as a high-visibility Yellow/White radar dot with a short motion trail
    """
    colors = colors or ColorScheme()
    pitch_l, pitch_w = pitch_dims
    pad = 12
    header_h = 26
    w = int(frame.shape[1] * width_ratio)
    pitch_h = int(w * (pitch_w / pitch_l))
    total_h = header_h + pitch_h

    radar_pitch = draw_pitch_radar(w, pitch_h, pad=pad, pitch_dims=pitch_dims)
    iw, ih = w - 2 * pad, pitch_h - 2 * pad

    def to_px(pos: Tuple[float, float]) -> Tuple[int, int]:
        xm = float(np.clip(pos[0], -2.0, pitch_l + 2.0))
        ym = float(np.clip(pos[1], -2.0, pitch_w + 2.0))
        return (pad + int(round((xm / pitch_l) * iw)), pad + int(round((ym / pitch_w) * ih)))

    for team, hull in (hulls or {}).items():
        if len(hull) < 3:
            continue
        poly = np.array([to_px(tuple(p)) for p in hull], dtype=np.int32)
        tcol = colors.for_team(team)
        ov = radar_pitch.copy()
        cv2.fillPoly(ov, [poly], tcol)
        cv2.addWeighted(ov, 0.22, radar_pitch, 0.78, 0, radar_pitch)
        cv2.polylines(radar_pitch, [poly], True, tcol, 1, cv2.LINE_AA)

    for p in (passes or []):
        if p.get('from_pos') is not None and p.get('to_pos') is not None:
            cv2.arrowedLine(radar_pitch, to_px(p['from_pos']), to_px(p['to_pos']),
                            colors.for_team(p['team']), 2, cv2.LINE_AA, tipLength=0.12)

    if ball_trail and len(ball_trail) > 1:
        trail_pts = [to_px(pt) for pt in ball_trail if pt is not None]
        for i in range(1, len(trail_pts)):
            frac = i / len(trail_pts)
            col = (int(120 + 135 * frac), 255, 255)
            cv2.line(radar_pitch, trail_pts[i - 1], trail_pts[i], col, max(1, int(round(2 * frac))), cv2.LINE_AA)

    for p in player_positions:
        pos = to_px(p['pitch_position'])
        team = p.get('team', 'unknown')
        color = colors.for_team(team)
        has_ball = p.get('has_ball', False)
        if has_ball:
            cv2.circle(radar_pitch, pos, 9, colors.holder, 2, cv2.LINE_AA)
        if p.get('is_goalkeeper'):
            cv2.rectangle(radar_pitch, (pos[0] - 6, pos[1] - 6), (pos[0] + 6, pos[1] + 6), color, -1)
            cv2.rectangle(radar_pitch, (pos[0] - 6, pos[1] - 6), (pos[0] + 6, pos[1] + 6), (255, 255, 255), 1)
        elif team == 'referee':
            cv2.circle(radar_pitch, pos, 4, color, -1, cv2.LINE_AA)
            cv2.circle(radar_pitch, pos, 4, (20, 20, 20), 1, cv2.LINE_AA)
        else:
            cv2.circle(radar_pitch, pos, 6, color, -1, cv2.LINE_AA)
            cv2.circle(radar_pitch, pos, 6, (255, 255, 255), 1, cv2.LINE_AA)

    if ball_position is not None:
        bpos = to_px(ball_position)
        cv2.circle(radar_pitch, bpos, 7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(radar_pitch, bpos, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(radar_pitch, bpos, 4, (20, 20, 20), 1, cv2.LINE_AA)

    if message:
        (tw, th), _ = cv2.getTextSize(message, FONT, 0.65, 2)
        org = ((w - tw) // 2, (pitch_h + th) // 2)
        cv2.putText(radar_pitch, message, org, FONT, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(radar_pitch, message, org, FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    panel = np.zeros((total_h, w, 3), dtype=np.uint8)
    panel[:header_h] = (24, 24, 24)
    panel[header_h:] = radar_pitch

    cv2.circle(panel, (12, header_h // 2), 4, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.putText(panel, "2D RADAR", (22, header_h - 8), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    lx = w - 210
    for label, col, is_ball in (("Team A", colors.team_a, False), ("Team B", colors.team_b, False), ("Ball", (0, 255, 255), True)):
        cv2.circle(panel, (lx, header_h // 2), 5 if not is_ball else 4, col, -1, cv2.LINE_AA)
        if is_ball:
            cv2.circle(panel, (lx, header_h // 2), 2, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(panel, (lx, header_h // 2), 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, label, (lx + 9, header_h - 8), FONT, 0.40, (225, 225, 225), 1, cv2.LINE_AA)
        lx += 72

    x, y = frame.shape[1] - w - margin, margin
    roi = frame[y:y + total_h, x:x + w]
    cv2.addWeighted(panel, alpha, roi, 1.0 - alpha, 0, roi)
    cv2.rectangle(frame, (x - 1, y - 1), (x + w, y + total_h), (240, 240, 240), 2, cv2.LINE_AA)
    return panel


def draw_pip(frame: np.ndarray, tactical_map: np.ndarray, width_ratio: float = 0.3,
             margin: int = 15) -> None:
    """
    Overlay a pre-rendered tactical map image as a Picture-in-Picture in the top-right corner (in place).
    """
    w = int(frame.shape[1] * width_ratio)
    h = int(w * tactical_map.shape[0] / tactical_map.shape[1])
    pip = cv2.resize(tactical_map, (w, h), interpolation=cv2.INTER_AREA)
    x, y = frame.shape[1] - w - margin, margin
    frame[y:y + h, x:x + w] = pip
    cv2.rectangle(frame, (x - 2, y - 2), (x + w + 1, y + h + 1), (255, 255, 255), 2)


def draw_scoreboard(frame: np.ndarray, possession: Dict[str, float], pass_stats: Dict[str, Dict],
                    colors: ColorScheme, margin: int = 15) -> None:
    """
    Draw a semi-transparent scoreboard with Ball Possession % (bar) and pass counts (in place).
    """
    pw, ph = 420, 120
    x, y = margin, frame.shape[0] - ph - margin
    roi = frame[y:y + ph, x:x + pw]
    roi[:] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)

    cv2.putText(frame, "BALL POSSESSION", (x + 12, y + 26), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    a, b = possession.get('team_a', 0.0), possession.get('team_b', 0.0)
    bar_x, bar_y, bar_w, bar_h = x + 12, y + 40, pw - 24, 22
    split = bar_x + int(bar_w * a / 100) if a + b > 0 else bar_x + bar_w // 2
    cv2.rectangle(frame, (bar_x, bar_y), (split, bar_y + bar_h), colors.team_a, -1)
    cv2.rectangle(frame, (split, bar_y), (bar_x + bar_w, bar_y + bar_h), colors.team_b, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

    for i, (team, name, pct) in enumerate((('team_a', 'Team A', a), ('team_b', 'Team B', b))):
        tx = x + 12 + i * (pw // 2)
        stats = pass_stats.get(team, {})
        cv2.rectangle(frame, (tx, y + 74), (tx + 14, y + 88), colors.for_team(team), -1)
        cv2.putText(frame, f"{name}: {pct:.1f}%", (tx + 22, y + 87), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Passes: {stats.get('passes', 0)}", (tx + 22, y + 110), FONT, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_leaderboard(frame: np.ndarray, entries: List[Tuple[str, str, float]],
                     colors: ColorScheme, margin: int = 15, max_rows: int = 5) -> None:
    """
    Draw a semi-transparent 'DISTANCE COVERED' leaderboard in the top-left corner.

    Args:
        entries: (display name, team label, distance in meters) sorted by distance desc
    """
    if not entries:
        return
    pw = 250
    ph = 34 + 22 * min(len(entries), max_rows)
    roi = frame[margin:margin + ph, margin:margin + pw]
    roi[:] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    cv2.putText(frame, "DISTANCE COVERED", (margin + 12, margin + 22), FONT, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    for i, (name, team, meters) in enumerate(entries[:max_rows]):
        y = margin + 46 + 22 * i
        cv2.rectangle(frame, (margin + 12, y - 10), (margin + 24, y + 3), colors.for_team(team), -1)
        text = f"{name}  {meters / 1000:.2f} km" if meters >= 1000 else f"{name}  {meters:.0f} m"
        cv2.putText(frame, text, (margin + 32, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray, tracks: List, team_assignments: Dict[int, str],
                   ball_position: Optional[Tuple[float, float]], ball_interpolated: bool,
                   analysis: Dict, pass_stats: Dict[str, Dict],
                   tactical_map_image: Optional[np.ndarray] = None,
                   colors: Optional[ColorScheme] = None, goalkeeper_ids: Optional[set] = None,
                   speeds: Optional[Dict[int, float]] = None,
                   leaderboard: Optional[List[Tuple[str, str, float]]] = None,
                   minimap_players: Optional[List[Dict]] = None,
                   minimap_ball: Optional[Tuple[float, float]] = None,
                   minimap_passes: Optional[List[Dict]] = None,
                   minimap_hulls: Optional[Dict[str, np.ndarray]] = None,
                   ball_trail: Optional[List[Tuple[float, float]]] = None,
                   minimap_message: Optional[str] = None) -> np.ndarray:
    """
    Produce the fully annotated output frame, including the live 2D Minimap (Radar).
    """
    colors = colors or ColorScheme()
    goalkeeper_ids = goalkeeper_ids or set()
    speeds = speeds or {}
    out = frame.copy()

    for track in tracks:
        team = team_assignments.get(track.track_id, 'unknown')
        label = f"GK {track.track_id}" if track.track_id in goalkeeper_ids else (
            f"REF {track.track_id}" if team == 'referee' else f"#{track.track_id}")
        speed = speeds.get(track.track_id)
        if speed is not None and team != 'referee':
            label += f" · {speed:.0f} km/h"
        draw_player(out, track.bbox, colors.for_team(team), label)
        if track.track_id == analysis.get('holder_id'):
            draw_triangle_marker(out, ((track.bbox[0] + track.bbox[2]) // 2, track.bbox[1] - 4), colors.holder)

    if ball_position is not None:
        bx, by = int(ball_position[0]), int(ball_position[1])
        draw_triangle_marker(out, (bx, by - 6), colors.ball, size=8, filled=not ball_interpolated)

    if analysis.get('ball_in_play') is False:
        cv2.putText(out, 'BALL NOT IN PLAY', (out.shape[1] // 2 - 170, 45), FONT, 1.0,
                    (60, 60, 235), 3, cv2.LINE_AA)

    if minimap_players is not None or minimap_message is not None:
        draw_minimap(out, minimap_players or [], minimap_ball, colors=colors,
                     passes=minimap_passes, hulls=minimap_hulls,
                     ball_trail=ball_trail, message=minimap_message)
    elif tactical_map_image is not None:
        draw_pip(out, tactical_map_image)
    draw_scoreboard(out, analysis.get('possession_stats', {}), pass_stats, colors)
    draw_leaderboard(out, leaderboard or [], colors)
    return out
