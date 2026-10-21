"""
AI Football Analyzer Pro - Streamlit Dashboard

Web front-end for the Football Tactical Analysis AI pipeline (Phase 7):
upload a match video, run the full two-pass GPU pipeline (YOLO11 + SAHI ball
detection, ByteTrack, team kits + role overrides, dynamic homography, Kalman
ball trajectory, speed/distance + team-shape analytics) and explore the
annotated video, metric cards, comparison charts, heatmaps and downloads.

    python -m streamlit run app.py

NOTE: the tactical-map calibration is hand-measured for the bundled test clip.
For any other video, map positions / speed / distance are indicative only unless
a calibration JSON is uploaded (sidebar).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from main import run_pipeline, default_model  # noqa: E402
from src.tactical_map import TacticalMap  # noqa: E402

RUNS_DIR = ROOT / 'data' / 'output' / 'app_runs'
TEAM_A_RGB, TEAM_B_RGB = '#e63946', '#3a6fd8'
TEAM_NAMES = {'team_a': 'Team A', 'team_b': 'Team B'}

# Standard 105x68 pitch landmarks (meters; x = along length, y = across width).
# "A end" = the goal the user chooses to be x=0, "B end" = x=105. The pitch is
# symmetric, so either orientation is fine as long as all 4+ points use the
# same one - consistency matters, not which end is labelled A.
LANDMARKS = [
    ('Corner, A end / top touchline (0, 0)', (0.0, 0.0)),
    ('Corner, A end / bottom touchline (0, 68)', (0.0, 68.0)),
    ('Corner, B end / top touchline (105, 0)', (105.0, 0.0)),
    ('Corner, B end / bottom touchline (105, 68)', (105.0, 68.0)),
    ('Penalty box A, outer top (16.5, 13.8)', (16.5, 13.84)),
    ('Penalty box A, outer bottom (16.5, 54.2)', (16.5, 54.16)),
    ('Penalty box A at goal line, top (0, 13.8)', (0.0, 13.84)),
    ('Penalty box A at goal line, bottom (0, 54.2)', (0.0, 54.16)),
    ('Penalty box B, outer top (88.5, 13.8)', (88.5, 13.84)),
    ('Penalty box B, outer bottom (88.5, 54.2)', (88.5, 54.16)),
    ('Penalty box B at goal line, top (105, 13.8)', (105.0, 13.84)),
    ('Penalty box B at goal line, bottom (105, 54.2)', (105.0, 54.16)),
    ('6-yard box A, outer top (5.5, 24.8)', (5.5, 24.84)),
    ('6-yard box A, outer bottom (5.5, 43.2)', (5.5, 43.16)),
    ('6-yard box B, outer top (99.5, 24.8)', (99.5, 24.84)),
    ('6-yard box B, outer bottom (99.5, 43.2)', (99.5, 43.16)),
    ('Penalty spot A (11, 34)', (11.0, 34.0)),
    ('Penalty spot B (94, 34)', (94.0, 34.0)),
    ('Halfway line at top touchline (52.5, 0)', (52.5, 0.0)),
    ('Halfway line at bottom touchline (52.5, 68)', (52.5, 68.0)),
    ('Center spot (52.5, 34)', (52.5, 34.0)),
    ('Center circle top edge (52.5, 24.9)', (52.5, 24.85)),
    ('Center circle bottom edge (52.5, 43.2)', (52.5, 43.15)),
]
MIN_CALIB_POINTS = 4
# Numbered to match the dots drawn on the landmark reference diagram
LM_LABELS = [f'({i}) {lbl}' for i, (lbl, _) in enumerate(LANDMARKS, 1)]
LM_MAP = {f'({i}) {lbl}': pt for i, (lbl, pt) in enumerate(LANDMARKS, 1)}

st.set_page_config(page_title='AI Football Analyzer Pro', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #111827, #1f2937);
        border: 1px solid #374151; border-radius: 12px; padding: 14px 18px;
    }
    div[data-testid="stMetric"] label {color: #9ca3af;}
    .hero {font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;}
    .sub {color: #6b7280; margin-top: -10px;}
</style>
""", unsafe_allow_html=True)


def to_browser_webm(src: str) -> str:
    """Re-encode the OpenCV mp4v render as VP8/WebM so st.video can play it.

    H.264 encoding is unavailable (no openh264 DLL) and the bundled ffmpeg
    binary can't read files in this environment, so we transcode with cv2.
    VP8 in a WebM container plays natively in all modern browsers.
    """
    try:
        import cv2
        dst = str(Path(src).with_name(Path(src).stem + '_web.webm'))
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'VP80'), fps, size)
        if not writer.isOpened():
            writer.release(); cap.release()
            return src
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release(); cap.release()
        return dst
    except Exception:
        return src  # Fallback: some browsers may still play the mp4v file


def metric_card(col, label, value, delta=None):
    col.metric(label, value, delta=delta, delta_color='off')


# ------------------------------------------------------------------ calibration
def read_video_frame(video_path: str, idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_clicks(frame: np.ndarray, clicks) -> np.ndarray:
    """Frame copy with numbered markers at each clicked point (native res coords)."""
    img = frame.copy()
    for i, (x, y) in enumerate(clicks, 1):
        p = (int(round(x)), int(round(y)))
        cv2.circle(img, p, 10, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, str(i), (p[0] + 12, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 3, cv2.LINE_AA)
    return img


@st.cache_resource
def landmark_reference() -> Image.Image:
    """Pitch wireframe with numbered landmark dots for the dropdown list."""
    tm = TacticalMap()
    img = tm.pitch_template.copy()
    for i, (_, (px, py)) in enumerate(LANDMARKS, 1):
        x, y = tm.pitch_to_map((px, py))
        cv2.circle(img, (x, y), 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (x - (7 if i < 10 else 12), y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def fit_homography(clicks, landmark_pts):
    """video px -> pitch m homography. Returns (H, rmse_m or None, error or None)."""
    v = np.array(clicks, np.float32)
    p = np.array(landmark_pts, np.float32)
    H, mask = cv2.findHomography(v, p, cv2.RANSAC if len(v) > 4 else 0, 1.5)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        return None, None, 'degenerate point set (points nearly collinear)'
    proj = cv2.perspectiveTransform(v.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - p, axis=1)
    if mask is not None:
        err = err[mask.ravel().astype(bool)] if mask.any() else err
    return H, float(np.sqrt((err ** 2).mean())), None


def overlay_preview(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Pitch wireframe back-projected onto the frame so the user can eyeball the fit."""
    Hi = np.linalg.inv(H)
    img = frame.copy()

    def to_video(poly):
        pts = cv2.perspectiveTransform(np.array(poly, np.float32).reshape(-1, 1, 2), Hi)
        return pts.reshape(-1, 2).astype(np.int32)

    def poly(pts, closed=True):
        v = to_video(pts)
        if np.isfinite(v).all():
            cv2.polylines(img, [v], closed, (0, 255, 255), 2, cv2.LINE_AA)

    poly([(0, 0), (105, 0), (105, 68), (0, 68)])                    # boundary
    poly([(52.5, 0), (52.5, 68)], closed=False)                     # halfway line
    poly([(0, 13.84), (16.5, 13.84), (16.5, 54.16), (0, 54.16)])    # box A
    poly([(105, 13.84), (88.5, 13.84), (88.5, 54.16), (105, 54.16)])  # box B
    circ = [(52.5 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t))
            for t in np.linspace(0, 2 * np.pi, 48)]
    poly(circ)                                                    # center circle
    return img


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title('AI Football Analyzer Pro')
    st.caption('YOLO11 · ByteTrack · SAHI · Kalman RTS · Homography analytics')
    st.divider()

    video_file = st.file_uploader('Match video (.mp4)', type=['mp4'])
    calib_file = st.file_uploader('Pitch calibration JSON (optional)', type=['json'],
                                  help='{frame: [[[video_x, video_y], [pitch_x, pitch_y]], ...]}')
    fast_preview = st.checkbox('Fast preview (first 300 frames)',
                               help='Demo mode: processes ~300 frames instead of the whole clip')
    run_clicked = st.button('Analyze video', type='primary', use_container_width=True,
                            disabled=video_file is None)

    st.divider()
    st.caption(f'Model: {default_model().name} (fine-tuned)' if default_model().exists() else 'Model: yolo11m.pt (COCO)')
    st.caption('GPU pipeline: ~2-10 min per minute of footage')
    st.caption('Built with Streamlit')

# ------------------------------------------------------------------ header
st.markdown('<div class="hero">Football Tactical Analysis AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Automated detection · tracking · role & team assignment · '
            'physics-based ball trajectory · real-world performance metrics</div>',
            unsafe_allow_html=True)
st.divider()

result_key = None if video_file is None else f'{video_file.name}:{video_file.size}:{fast_preview}'
if 'result_key' not in st.session_state:
    st.session_state.result_key = None
    st.session_state.result = None
if result_key != st.session_state.result_key:
    st.session_state.result_key = result_key
    st.session_state.result = None

# ------------------------------------------------------------------ calibration
st.session_state.setdefault('clicks', [])
st.session_state.setdefault('calibs', {})
input_path = None
if video_file is not None:
    vid_key = f'{video_file.name}:{video_file.size}'
    run_dir = RUNS_DIR / Path(video_file.name).stem
    if st.session_state.get('vid_key') != vid_key:
        st.session_state.vid_key = vid_key
        st.session_state.clicks = []
        st.session_state.last_click = None
        st.session_state.calibs = {}
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'input.mp4').write_bytes(video_file.getvalue())
    input_path = run_dir / 'input.mp4'

    cap0 = cv2.VideoCapture(str(input_path))
    n_frames = max(int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0)
    cap0.release()

    with st.expander('Pitch calibration - click >= 4 landmarks for real-world metrics',
                     expanded=not st.session_state.calibs):
        st.caption('Pick a frame with visible lines, click >= 4 landmarks, then assign each '
                   'click its pitch position from the diagram. Keep one orientation: choose '
                   'ONE goal as the "A" (x=0) end and use A-landmarks for it throughout.')
        calib_frame = st.slider('Frame to calibrate on', 0, n_frames, 0, key='calib_frame')
        if st.session_state.get('clicks_frame') != calib_frame:
            st.session_state.clicks_frame = calib_frame
            st.session_state.clicks = []
            st.session_state.last_click = None

        frame = read_video_frame(input_path, calib_frame)
        col_img, col_ref = st.columns([3, 2])
        with col_img:
            if frame is not None:
                marked = draw_clicks(frame, st.session_state.clicks)
                ratio = min(1.0, 1050 / marked.shape[1])
                pil = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
                if ratio < 1.0:
                    pil = pil.resize((1050, int(marked.shape[0] * ratio)))
                click = streamlit_image_coordinates(pil, key=f'img_{vid_key}_{calib_frame}')
                if click and click != st.session_state.last_click:
                    st.session_state.last_click = click
                    st.session_state.clicks.append(
                        (click['x'] / ratio, click['y'] / ratio))
                    st.rerun()
        with col_ref:
            st.image(landmark_reference(), use_container_width=True,
                     caption='Landmark numbers - match them to the dropdowns below')

        clicks = st.session_state.clicks
        if clicks:
            chosen = []
            for i, (cx, cy) in enumerate(clicks):
                chosen.append(st.selectbox(
                    f'Point {i + 1}  at ({cx:.0f}, {cy:.0f})', LM_LABELS,
                    index=min(i, len(LM_LABELS) - 1), key=f'lm_{i}'))
            b1, b2, b3 = st.columns(3)
            if b1.button('Undo last point'):
                st.session_state.clicks.pop()
                st.session_state.last_click = None
                st.rerun()
            if b2.button('Clear points'):
                st.session_state.clicks = []
                st.session_state.last_click = None
                st.rerun()
            if b3.button('Save calibration', type='primary',
                         disabled=len(clicks) < MIN_CALIB_POINTS):
                pts = [LM_MAP[c] for c in chosen]
                if len(set(pts)) < len(pts):
                    st.error('Each clicked point needs a different landmark.')
                else:
                    H, rmse, err = fit_homography(clicks, pts)
                    if err:
                        st.error(f'Cannot fit homography: {err}')
                    else:
                        st.session_state.calibs[calib_frame] = [
                            [list(map(float, c)), list(map(float, p))]
                            for c, p in zip(clicks, pts)]
                        st.session_state.clicks = []
                        st.session_state.preview = (calib_frame, H, rmse)
                        st.rerun()
        elif frame is not None:
            st.caption('No points clicked yet - click directly on the frame image.')

        if st.session_state.get('preview'):
            pf, pH, prmse = st.session_state.preview
            pframe = read_video_frame(input_path, pf)
            if pframe is not None:
                msg = f'Calibration fit for frame {pf}'
                if prmse is not None:
                    msg += f' - reprojection error ~{prmse:.2f} m'
                st.image(cv2.cvtColor(overlay_preview(pframe, pH), cv2.COLOR_BGR2RGB),
                         caption=msg + '. Verify the wireframe hugs the pitch lines.',
                         use_container_width=True)
        if st.session_state.calibs:
            st.success('Calibration saved for frame(s): ' +
                       ', '.join(str(f) for f in sorted(st.session_state.calibs)) +
                       '. Move the frame slider to add another (e.g. after a camera cut).')

# ------------------------------------------------------------------ run
if run_clicked and video_file is not None:
    run_dir = RUNS_DIR / Path(video_file.name).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(video_file.getvalue())
    calib_path = None
    if st.session_state.calibs:
        calib_path = run_dir / 'calibration.json'
        calib_path.write_text(json.dumps(
            {str(f): pairs for f, pairs in st.session_state.calibs.items()}))
    elif calib_file is not None:
        calib_path = run_dir / 'calibration.json'
        calib_path.write_bytes(calib_file.getvalue())
    output_path = run_dir / 'annotated.mp4'

    progress = st.progress(0.0, text='Initializing pipeline...')

    def cb(phase, i, n):
        if phase == 'pass1':
            progress.progress(0.02 + 0.90 * i / max(n, 1),
                              text=f'Pass 1/2 — detection, tracking, camera motion: frame {i}/{n}')
        else:
            progress.progress(0.92 + 0.08 * i / max(n, 1),
                              text=f'Pass 2/2 — analytics and rendering: frame {i}/{n}')

    with st.spinner('Running GPU analysis (this takes a few minutes)...'):
        try:
            result = run_pipeline(input_path, output_path, ball_smoother='kalman',
                                  calibration=str(calib_path) if calib_path else None,
                                  max_frames=300 if fast_preview else None,
                                  progress_cb=cb, log=lambda *a, **k: None)
        except RuntimeError as e:
            st.error(f'Pipeline failed: {e}')
            st.stop()
    progress.progress(0.95, text='Encoding browser video (WebM)...')
    result['web_video'] = to_browser_webm(result['video'])
    progress.progress(1.0, text='Done.')
    progress.empty()
    st.session_state.result = result
    st.session_state.result_key = result_key

# ------------------------------------------------------------------ results
result = st.session_state.result

if result is None:
    c1, c2, c3 = st.columns(3)
    c1.markdown('**Detection & tracking** — YOLO11 fine-tuned on football footage, SAHI tiled inference for the tiny ball, ByteTrack multi-object tracking.')
    c2.markdown('**Tactical layer** — optical-flow camera compensation, dynamic homography, team convex hulls, heatmaps, possession & pass events.')
    c3.markdown('**Physical metrics** — physics-smoothed ball trajectory (Kalman + RTS), real-world speed and distance per player, team compactness.')
    st.info('Upload a match video in the sidebar to start the analysis.')
else:
    stats = result['stats']
    poss = stats['possession_percent']
    passes = stats['pass_stats']
    phys = stats['physical']
    ball = stats['ball']
    shape = stats.get('team_shape', {})
    roles = stats['roles']['track_ids_per_label']

    if st.session_state.calibs:
        pass  # calibrated interactively for this video
    elif calib_file is None:
        if Path(video_file.name).stem == 'test_match':
            st.info('Using the built-in calibration measured for this clip.')
        else:
            st.warning('No pitch calibration - map positions, speed and distance are '
                       'indicative only. Use the calibration tool above (or upload a '
                       'calibration JSON) for real-world metrics.')

    st.success(f"Analysis complete — {result['frames']} frames @ {result['fps']:.0f} FPS "
               f"({Path(result['model']).name})")

    # ---- annotated video
    st.subheader('Annotated video')
    st.video(result['web_video'])

    # ---- headline metrics
    st.subheader('Match metrics')
    total_passes = sum(s['passes'] for s in passes.values())
    top_speed = phys['top_speed'][0][1]['max_speed_kmh'] if phys['top_speed'] else 0
    top_dist = phys['top_distance'][0][1]['distance_m'] if phys['top_distance'] else 0
    coverage = 100.0 * (ball['detected'] + ball.get('predicted', ball.get('interpolated', 0))) / ball['frames']
    cols = st.columns(6)
    metric_card(cols[0], 'Possession A', f"{poss['team_a']:.1f}%")
    metric_card(cols[1], 'Possession B', f"{poss['team_b']:.1f}%")
    metric_card(cols[2], 'Total passes', str(total_passes))
    metric_card(cols[3], 'Top speed', f'{top_speed:.1f} km/h')
    metric_card(cols[4], 'Max distance', f'{top_dist:.0f} m')
    metric_card(cols[5], 'Ball coverage', f'{coverage:.1f}%')

    # ---- charts row 1
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Ball possession**')
        df = pd.DataFrame({'Team': ['Team A', 'Team B'],
                           'Possession': [poss['team_a'], poss['team_b']]})
        fig = px.pie(df, names='Team', values='Possession', hole=0.55,
                     color='Team', color_discrete_map={'Team A': TEAM_A_RGB, 'Team B': TEAM_B_RGB})
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('**Passes & turnovers**')
        df = pd.DataFrame([
            {'Team': TEAM_NAMES[t], 'Event': 'Completed passes', 'Count': passes[t]['passes']}
            for t in ('team_a', 'team_b')] + [
            {'Team': TEAM_NAMES[t], 'Event': 'Turnovers', 'Count': passes[t]['turnovers']}
            for t in ('team_a', 'team_b')])
        fig = px.bar(df, x='Event', y='Count', color='Team', barmode='group',
                     color_discrete_map={'Team A': TEAM_A_RGB, 'Team B': TEAM_B_RGB})
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        st.markdown('**Team compactness (median)**')
        rows = [{'Team': TEAM_NAMES[t], 'Metric': m, 'Value': s[k]}
                for t, s in shape.items() if t in TEAM_NAMES
                for m, k in [('Hull area (m²)', 'median_area_m2'), ('Width (m)', 'mean_width_m'),
                             ('Depth (m)', 'mean_depth_m')]
                if s.get(k) is not None]
        if rows:
            fig = px.bar(pd.DataFrame(rows), x='Metric', y='Value', color='Team',
                         barmode='group',
                         color_discrete_map={'Team A': TEAM_A_RGB, 'Team B': TEAM_B_RGB})
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No team-shape data — requires a valid pitch calibration '
                    '(see sidebar calibration JSON).')

    # ---- charts row 2
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Distance covered — top players**')
        top = phys['top_distance'][:8]
        df = pd.DataFrame({'Player': [n for n, _ in top][::-1],
                           'Distance (m)': [s['distance_m'] for _, s in top][::-1]})
        fig = px.bar(df, x='Distance (m)', y='Player', orientation='h',
                     color_discrete_sequence=[TEAM_B_RGB])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('**Ball trajectory coverage**')
        filled = ball.get('predicted', ball.get('interpolated', 0))
        df = pd.DataFrame({'Source': ['Detected (YOLO+SAHI)', 'Predicted (Kalman)', 'Missing'],
                           'Frames': [ball['detected'], filled, ball['missing']]})
        fig = px.bar(df, x='Frames', y='Source', orientation='h',
                     color='Source', color_discrete_map={
                         'Detected (YOLO+SAHI)': '#2a9d8f', 'Predicted (Kalman)': '#e9c46a',
                         'Missing': '#6b7280'})
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---- heatmaps
    st.subheader('Team heatmaps')
    h1, h2 = st.columns(2)
    h1.image(result['heatmaps'][0], caption='Team A', use_container_width=True)
    h2.image(result['heatmaps'][1], caption='Team B', use_container_width=True)

    # ---- details + downloads
    with st.expander('Technical details'):
        st.json({'roles': roles,
                 'goalkeepers': stats['roles']['goalkeepers'],
                 'referee_overrides': stats['roles']['referee_overrides'],
                 'ball': {k: v for k, v in ball.items() if k != 'static_objects_removed'},
                 'camera': {'segments': stats['camera']['segments'],
                            'valid_frames': stats['camera']['valid_frames']},
                 'model': stats['model']})

    st.subheader('Downloads')
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button('Annotated video (.mp4)', Path(result['video']).read_bytes(),
                       file_name=Path(result['video']).name)
    d2.download_button('Stats (JSON)', Path(result['stats_path']).read_bytes(),
                       file_name=Path(result['stats_path']).name)
    d3.download_button('Ball trajectory (CSV)', Path(result['ball_csv']).read_bytes(),
                       file_name=Path(result['ball_csv']).name)
    d4.download_button('Camera motion (CSV)', Path(result['motion_csv']).read_bytes(),
                       file_name=Path(result['motion_csv']).name)
