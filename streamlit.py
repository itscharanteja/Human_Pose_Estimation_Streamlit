import streamlit as st
import os
import tempfile
import subprocess
import uuid
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import shutil
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []

def process_video(uploaded_file, video_name=None):
    if uploaded_file is None:
        return None

    video_id = str(uuid.uuid4())[:8]
    
    if video_name is None or video_name.strip() == "":
        video_name = f"Video_{video_id}" 
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_dir = os.path.join(tempfile.gettempdir(), f"pose_{timestamp}_{video_id}")
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    alphapose_dir = os.path.join(temp_dir, "alphapose") 
    motionbert_dir = os.path.join(temp_dir, "motionbert")
    os.makedirs(alphapose_dir, exist_ok=True)
    os.makedirs(motionbert_dir, exist_ok=True)
    
    st.info(f"Processing in directory: {temp_dir}")

    input_video_path = os.path.join(temp_dir, "input_video.mp4")
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"▶️ Running AlphaPose for {video_name}...")
    script_path = "/Users/charan/Documents/Projects/Human_Pose_Estimation_Streamlit/AlphaPose/scripts/inference.sh"
    config = "/Users/charan/Documents/Projects/Human_Pose_Estimation_Streamlit/AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
    ckpt = "/Users/charan/Documents/Projects/Human_Pose_Estimation_Streamlit/AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth"

    alphapose_run = subprocess.run(
        ["bash", script_path, config, ckpt, input_video_path, temp_dir],
        cwd="AlphaPose",
    )

    json_candidates = glob.glob(os.path.join(temp_dir, "**", "alphapose-results.json"), recursive=True)
    if not json_candidates:
        st.error("❌ AlphaPose did not produce 'alphapose-results.json'.")
        return None

    alphapose_json = json_candidates[0]
    
    # Look for 2D pose video from AlphaPose
    alphapose_video = glob.glob(os.path.join(temp_dir, "**", "*vis.mp4"), recursive=True)
    alphapose_video_path = alphapose_video[0] if alphapose_video else None
    
    if alphapose_video_path:
        st.success(f"✅ Found 2D pose video from AlphaPose: {os.path.basename(alphapose_video_path)}")

    st.write(f"🤖 Running MotionBERT for {video_name}...")
    motionbert_cmd = [
        "python", "/Users/charan/Documents/Projects/Human_Pose_Estimation_Streamlit/MotionBERT/infer_wild_2.py",
        "--json_path", alphapose_json,
        "--vid_path", input_video_path,
        "--out_path", motionbert_dir
    ]
    try:
        subprocess.run(motionbert_cmd, check=True)
    except subprocess.CalledProcessError as e:
        st.error("🚫 MotionBERT failed!")
        return None

    # Look for 3D pose video from MotionBERT
    motionbert_video = glob.glob(os.path.join(motionbert_dir, "*.mp4"), recursive=True)
    motionbert_video_path = motionbert_video[0] if motionbert_video else None
    
    if motionbert_video_path:
        st.success(f"✅ Found 3D pose video from MotionBERT: {os.path.basename(motionbert_video_path)}")
    else:
        st.warning("⚠️ No 3D pose video found from MotionBERT output.")

    st.write(f"📊 Extracting Biomechanics for {video_name}...")
    
    npy_files = glob.glob(os.path.join(motionbert_dir, "*.npy"))
    if not npy_files:
        st.error("❌ No NPY files found from MotionBERT output.")
        return None
        
    pred_npy = npy_files[0]  

    try:
        biomech_dir = os.path.join(temp_dir, f"biomech_{video_id}")
        os.makedirs(biomech_dir, exist_ok=True)

        safe_video_name = video_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        unique_npy_name = f"{safe_video_name}_{video_id}.npy"
        unique_npy_path = os.path.join(biomech_dir, unique_npy_name)

        shutil.copy(pred_npy, unique_npy_path)
        
        st.write(f"Running biomechanics extraction on NPY file: {unique_npy_path}")

        biomech_cmd = [
            "python",
            "/Users/charan/Documents/Projects/Human_Pose_Estimation_Streamlit/MotionBERT/extract_biomechanics_single.py",
            "--input", unique_npy_path
        ]
        
        result = subprocess.run(
            biomech_cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=biomech_dir  
        )
        
        st.write("Biomechanics command output:")
        st.code(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)

        expected_biomech_dir = os.path.join(biomech_dir, f"biomech_outputs_{os.path.splitext(unique_npy_name)[0]}")

        if os.path.exists(expected_biomech_dir):
            output_biomech_dir = expected_biomech_dir
            st.success(f"✅ Found biomechanics output directory: {output_biomech_dir}")
        else:
            biomech_output_dirs = glob.glob(os.path.join(biomech_dir, "biomech_outputs_*"))
            if biomech_output_dirs:
                output_biomech_dir = biomech_output_dirs[0]
                st.success(f"✅ Found biomechanics output directory: {output_biomech_dir}")
            else:
                output_lines = result.stdout.split('\n')
                output_dir_from_log = None
                
                for line in output_lines:
                    if "Analysis results will be saved to:" in line:
                        output_dir_from_log = line.split("Analysis results will be saved to:")[1].strip()
                        break
                
                if output_dir_from_log and os.path.exists(output_dir_from_log):
                    output_biomech_dir = output_dir_from_log
                    st.success(f"✅ Found biomechanics output directory from log: {output_biomech_dir}")
                else:
                    st.error("❌ Could not locate biomechanics output directory")
                    st.code(result.stdout)
                    return None

        report_path = os.path.join(output_biomech_dir, "comprehensive_report.json")
        
        if not os.path.exists(report_path):
            st.error(f"❌ Comprehensive report not found at: {report_path}")
            st.code(result.stdout) 
            return None
            
    except subprocess.CalledProcessError as e:
        st.error("❌ Biomechanics extraction failed!")
        st.code(e.stderr)
        return None

    return {
        'video_name': video_name,
        'video_id': video_id,
        'video_path': input_video_path,
        'alphapose_video_path': alphapose_video_path,
        'motionbert_video_path': motionbert_video_path,
        'npy_path': pred_npy,
        'report_path': report_path,
        'biomech_dir': output_biomech_dir,
        'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def display_report(report_path, video_name=None):
    """Display the biomechanics report with proper formatting."""
    if not os.path.exists(report_path):
        st.error(f"❌ Report file not found at: {report_path}")
        return None

    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
            
        summary = report_data.get('summary', {})
        
        if not summary:
            st.error("❌ Report doesn't contain expected 'summary' data")
            return None
            
        if video_name:
            st.header(f"🚴 Analysis for {video_name}")
        
        st.markdown("### 📊 Summary Metrics")
        cols = st.columns(3)
        cols[0].metric("Duration", f"{summary.get('duration_seconds', 'N/A'):.2f} seconds")
        cols[1].metric("Cadence", f"{summary.get('cadence_rpm', 'N/A'):.2f} RPM")
        cols[2].metric("Cycle Consistency", f"{summary.get('cycle_consistency', 'N/A'):.4f}")
        
        return summary
        
    except Exception as e:
        st.error(f"❌ Error processing report: {str(e)}")
        st.exception(e)
        return None

def visualize_3d_pose(npy_path, selected_frame=0):
    try:
        pose_data = np.load(npy_path)
        
        n_frames, n_joints, _ = pose_data.shape
        
        if selected_frame >= n_frames:
            selected_frame = n_frames - 1
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  
            (0, 6), (6, 7), (7, 8), (8, 9),  
            (0, 12), (12, 13), (13, 14), (14, 15),  
            (14, 16), (14, 17),  
            (16, 18), (18, 20), (20, 22),  
            (17, 19), (19, 21), (21, 23)   
        ]
        
        fig = go.Figure()
        
        pose = pose_data[selected_frame]
        x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=4, color='blue'),
            name='Joints'
        ))
        
        for connection in connections:
            if connection[0] < n_joints and connection[1] < n_joints:  
                fig.add_trace(go.Scatter3d(
                    x=[x[connection[0]], x[connection[1]]],
                    y=[y[connection[0]], y[connection[1]]],
                    z=[z[connection[0]], z[connection[1]]],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False
                ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', range=[np.min(x)-0.5, np.max(x)+0.5]),
                yaxis=dict(title='Y', range=[np.min(y)-0.5, np.max(y)+0.5]),
                zaxis=dict(title='Z', range=[np.min(z)-0.5, np.max(z)+0.5]),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500,
            title=f"3D Pose Visualization (Frame {selected_frame+1}/{n_frames})"
        )
        
        return fig, n_frames
        
    except Exception as e:
        st.error(f"Error loading 3D pose data: {str(e)}")
        return None, 0

def compare_reports(summary1, summary2, name1, name2):
    if not summary1 or not summary2:
        st.error("❌ Cannot compare reports: missing data")
        return
    
    if summary1 is summary2:
        st.error("⚠️ Warning: Both reports point to the same data object. This suggests a problem with processing distinct videos.")
    
    st.info(f"Comparing data from: {name1} vs {name2}")
    st.info(f"Summary data object IDs: {id(summary1)} vs {id(summary2)}")
    
    st.header(f"🔄 Comparing: {name1} vs {name2}")  
    
    st.markdown("### 📊 Basic Metrics Comparison")
    basic_metrics = [
        ('Duration (seconds)', 'duration_seconds'),
        ('Cadence (RPM)', 'cadence_rpm'),
        ('Cycle Consistency', 'cycle_consistency')
    ]
    
    comparison_data = []
    for label, key in basic_metrics:
        val1 = summary1.get(key, 'N/A')
        val2 = summary2.get(key, 'N/A')
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            diff_pct = (diff / val1 * 100) if val1 != 0 else 0
            comparison_data.append({
                'Metric': label, 
                name1: f"{val1:.2f}", 
                name2: f"{val2:.2f}", 
                'Difference': f"{diff:.2f} ({diff_pct:.1f}%)"
            })
    
    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data).set_index('Metric'))
    
    st.markdown("### 📏 Range of Motion Comparison")
    
    rom1 = summary1.get('range_of_motion', {})
    rom2 = summary2.get('range_of_motion', {})
    
    if rom1 and rom2:
        fig = make_subplots(rows=2, cols=3, 
                          subplot_titles=["Hip Right", "Knee Right", "Ankle Right", 
                                         "Hip Left", "Knee Left", "Ankle Left"])
        
        joints = [
            ('hip_right', 1, 1), ('knee_right', 1, 2), ('ankle_right', 1, 3),
            ('hip_left', 2, 1), ('knee_left', 2, 2), ('ankle_left', 2, 3)
        ]
        
        for joint, row, col in joints:
            if joint in rom1 and joint in rom2:
                min1, max1 = rom1[joint].get('min', 0), rom1[joint].get('max', 0)
                min2, max2 = rom2[joint].get('min', 0), rom2[joint].get('max', 0)
                
                fig.add_trace(
                    go.Bar(
                        x=[name1], 
                        y=[max1 - min1],
                        base=min1,
                        name=f"{name1} {joint}",
                        marker_color='blue',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Bar(
                        x=[name2], 
                        y=[max2 - min2],
                        base=min2,
                        name=f"{name2} {joint}",
                        marker_color='red',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                fig.update_yaxes(title_text="Angle (degrees)", row=row, col=col)
        
        fig.update_layout(height=600, width=800, title_text=f"Range of Motion: {name1} vs {name2}")
        st.plotly_chart(fig)
        
        rom_comparison = []
        for joint in set(list(rom1.keys()) + list(rom2.keys())):
            if joint in rom1 and joint in rom2:
                range1 = rom1[joint].get('range', 0)
                range2 = rom2[joint].get('range', 0)
                diff = range2 - range1
                diff_pct = (diff / range1 * 100) if range1 != 0 else 0
                
                rom_comparison.append({
                    'Joint': joint.replace('_', ' ').title(),
                    f"{name1} Range": f"{range1:.2f}°",
                    f"{name2} Range": f"{range2:.2f}°",
                    'Difference': f"{diff:.2f}° ({diff_pct:.1f}%)"
                })
        
        st.dataframe(pd.DataFrame(rom_comparison).set_index('Joint'))
        
    st.markdown("### ⚡ Peak Angular Velocity Comparison")
    
    vel1 = summary1.get('peak_angular_velocity', {})
    vel2 = summary2.get('peak_angular_velocity', {})
    
    if vel1 and vel2:
        vel_comparison = []
        for joint in set(list(vel1.keys()) + list(vel2.keys())):
            if joint in vel1 and joint in vel2:
                val1 = vel1.get(joint, 0)
                val2 = vel2.get(joint, 0)
                diff = val2 - val1
                diff_pct = (diff / val1 * 100) if val1 != 0 else 0
                
                vel_comparison.append({
                    'Joint': joint.replace('_', ' ').title(),
                    f"{name1} (deg/s)": f"{val1:.2f}",
                    f"{name2} (deg/s)": f"{val2:.2f}",
                    'Difference': f"{diff:.2f} ({diff_pct:.1f}%)"
                })
        
        st.dataframe(pd.DataFrame(vel_comparison).set_index('Joint'))
        
        comp_df = pd.DataFrame(vel_comparison)
        fig = px.bar(
            comp_df, 
            x='Joint', 
            y=[f"{name1} (deg/s)", f"{name2} (deg/s)"],
            barmode='group',
            title=f"Peak Angular Velocities: {name1} vs {name2}",
            labels={'value': 'Angular Velocity (deg/s)', 'variable': 'Video'}
        )
        st.plotly_chart(fig)

def display_plots(biomech_dir, video_name=None):
    """Display the biomechanics plots from the analysis"""
    plot_files = [
        os.path.join(biomech_dir, "joint_angles_plot.png"),
        os.path.join(biomech_dir, "angular_velocities_plot.png"),
        os.path.join(biomech_dir, "pedaling_cycles.png")
    ]
    
    st.markdown(f"### 📈 Visualization{' for ' + video_name if video_name else ''}")
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            st.image(plot_file)


st.set_page_config(page_title="Cycling Biomechanics Analysis", layout="wide")
st.title("🚴‍♂️ Cycling Biomechanics Analysis & Comparison")

with st.sidebar:
    st.header("📋 Analysis Controls")
    
    with st.form("video_form"):
        uploaded_file = st.file_uploader("Upload a cycling video", type=["mp4", "mov", "avi"])
        video_name = st.text_input("Video name (optional)")
        submit_button = st.form_submit_button("Process Video")
    
    st.header("🎬 Processed Videos")
    if st.session_state.processed_videos:
        for i, video in enumerate(st.session_state.processed_videos):
            st.write(f"{i+1}. {video['video_name']} - {video['processed_at']}")
    else:
        st.write("No videos processed yet.")
    
    if st.button("Clear All Videos"):
        st.session_state.processed_videos = []
        st.rerun()  

if submit_button and uploaded_file:
    with st.spinner("Processing video..."):
        result = process_video(uploaded_file, video_name)
        if result:
            st.session_state.processed_videos.append(result)
            st.success(f"✅ Processing complete for {result['video_name']}!")
            st.rerun() 

num_videos = len(st.session_state.processed_videos)

if num_videos == 0:
    st.info("👆 Upload a video to begin analysis.")
    st.markdown("""
    ### How to use this app:
    1. Upload a cycling video using the sidebar on the left
    2. Wait for the processing to complete
    3. Upload a second video to compare biomechanics
    4. View detailed analysis including 3D pose visualization
    
    The app extracts 3D pose data using AlphaPose and MotionBERT, then analyzes cycling biomechanics.
    """)

elif num_videos == 1:
    video = st.session_state.processed_videos[0]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Biomechanics Analysis", "🦴 3D Pose Visualization", "🎥 Original Video", "🤖 Pose Videos"])
    
    with tab1:
        summary = display_report(video['report_path'], video['video_name'])
        display_plots(video['biomech_dir'], video['video_name'])
    
    with tab2:
        fig, n_frames = visualize_3d_pose(video['npy_path'])
        if fig:
            frame_slider = st.slider("Frame", 0, max(0, n_frames-1), 0)
            updated_fig, _ = visualize_3d_pose(video['npy_path'], frame_slider)
            st.plotly_chart(updated_fig)
    
    with tab3:
        st.subheader(f"Original Video: {video['video_name']}") 
        st.video(video['video_path'])
        
    with tab4:
        pose_video_view_select = st.selectbox(
            "Select video to view 3D pose videos",
            range(num_videos),
            format_func=lambda i: st.session_state.processed_videos[i]['video_name'],
            key="pose_video_view"
        )
        
        selected_pose_video_view = st.session_state.processed_videos[pose_video_view_select]
        st.subheader(f"3D Pose Video for {selected_pose_video_view['video_name']}")
        
        if selected_pose_video_view.get('motionbert_video_path') and os.path.exists(selected_pose_video_view['motionbert_video_path']):
            st.video(selected_pose_video_view['motionbert_video_path'])
        else:
            st.warning("3D pose video not available")

elif num_videos >= 2:
    st.header("🔍 Video Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        video1_idx = st.selectbox(
            "Select first video", 
            range(num_videos),
            format_func=lambda i: st.session_state.processed_videos[i]['video_name']
        )
        video1 = st.session_state.processed_videos[video1_idx]
        st.subheader(f"Video 1: {video1['video_name']}")  
        st.video(video1['video_path'], start_time=0)
    
    with col2:
        video2_idx = st.selectbox(
            "Select second video", 
            range(num_videos),
            index=min(1, num_videos-1),  
            format_func=lambda i: st.session_state.processed_videos[i]['video_name']
        )
        video2 = st.session_state.processed_videos[video2_idx]
        st.subheader(f"Video 2: {video2['video_name']}") 
        st.video(video2['video_path'], start_time=0)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Comparison", "📊 Individual Analysis", "🦴 3D Pose", "🤖 Pose Videos"])
    
    with tab1:
        st.info(f"Video 1: {video1['video_name']} (ID: {video1['video_id']})")
        st.info(f"Video 2: {video2['video_name']} (ID: {video2['video_id']})")
        
        st.info(f"Report paths: \nVideo 1: {video1['report_path']} \nVideo 2: {video2['report_path']}")
        
        if video1['report_path'] == video2['report_path']:
            st.error("⚠️ ERROR: Both videos are using the same report file! This indicates an issue with file paths.")
            st.warning("Please clear all videos and try again with fixed code.")
        else:
            try:
                with open(video1['report_path'], 'r') as f:
                    report1 = json.load(f)
                    summary1 = report1.get('summary', {})
                    
                with open(video2['report_path'], 'r') as f:
                    report2 = json.load(f)
                    summary2 = report2.get('summary', {})
                
                st.success("✅ Successfully loaded both reports for comparison")
                
                compare_reports(summary1, summary2, video1['video_name'], video2['video_name'])
            except Exception as e:
                st.error(f"❌ Error loading or comparing reports: {str(e)}")
                st.exception(e)
    
    with tab2:
        video_select = st.selectbox(
            "Select video to analyze",
            range(num_videos),
            format_func=lambda i: st.session_state.processed_videos[i]['video_name']
        )
        
        selected_video = st.session_state.processed_videos[video_select]
        display_report(selected_video['report_path'], selected_video['video_name'])
        display_plots(selected_video['biomech_dir'], selected_video['video_name'])
    
    with tab3:
        pose_video_select = st.selectbox(
            "Select video for 3D pose visualization",
            range(num_videos),
            format_func=lambda i: st.session_state.processed_videos[i]['video_name']
        )
        
        selected_pose_video = st.session_state.processed_videos[pose_video_select]
        st.subheader(f"3D Pose for: {selected_pose_video['video_name']}")
        fig, n_frames = visualize_3d_pose(selected_pose_video['npy_path'])
        
        if fig:
            frame_slider = st.slider("Frame", 0, max(0, n_frames-1), 0)
            updated_fig, _ = visualize_3d_pose(selected_pose_video['npy_path'], frame_slider)
            st.plotly_chart(updated_fig)
            
    with tab4:
        pose_video_view_select = st.selectbox(
            "Select video to view 3D pose videos",
            range(num_videos),
            format_func=lambda i: st.session_state.processed_videos[i]['video_name'],
            key="pose_video_view"
        )
        
        selected_pose_video_view = st.session_state.processed_videos[pose_video_view_select]
        st.subheader(f"3D Pose Video for {selected_pose_video_view['video_name']}")
        
        if selected_pose_video_view.get('motionbert_video_path') and os.path.exists(selected_pose_video_view['motionbert_video_path']):
            st.video(selected_pose_video_view['motionbert_video_path'])
        else:
            st.warning("3D pose video not available")