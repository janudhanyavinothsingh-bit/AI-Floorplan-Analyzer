import streamlit as st
import cv2
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Floor Plan Analyzer Pro",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================
st.markdown("""
    <style>
        .header-container {
            background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
            padding: 3rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
        }
        
        .header-container h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .header-container p {
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(14, 165, 233, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ==========================================
# PREPROCESSING
# ==========================================
def preprocess_image(file):
    """Enhanced image preprocessing"""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    
    # Resize while keeping aspect ratio
    max_dim = 1200
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Histogram equalization for better contrast
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img_enhanced = cv2.merge([l, a, b])
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    
    return img_rgb

# ==========================================
# CLASSIFICATION
# ==========================================
def classify_room(area):
    """Classify room type based on area"""
    if area is None or area <= 0:
        return "Unknown"
    
    if area > 80000:
        return "Living Room"
    elif area > 50000:
        return "Hall"
    elif area > 35000:
        return "Bedroom"
    elif area > 20000:
        return "Kitchen"
    elif area > 10000:
        return "Bathroom"
    else:
        return "Closet"

# ==========================================
# DETECTION
# ==========================================
def detect_rooms(img_rgb):
    """Advanced room detection"""
    h, w = img_rgb.shape[:2]
    min_area = 800
    max_area = h * w * 0.5
    
    rooms = []
    room_id = 0
    
    # Get unique colors
    pixels = img_rgb.reshape(-1, 3)
    pixels_quantized = (pixels // 20) * 20
    unique_colors = np.unique(pixels_quantized, axis=0)
    
    for color in unique_colors:
        # Create color range
        lower = np.maximum(color - 30, 0)
        upper = np.minimum(color + 30, 255)
        
        mask = cv2.inRange(img_rgb, lower, upper)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                
                # Calculate circularity
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # Filter by shape
                if 0.2 < circularity < 0.95:
                    room_id += 1
                    rooms.append({
                        "id": room_id,
                        "x": int(x),
                        "y": int(y),
                        "w": int(w_rect),
                        "h": int(h_rect),
                        "area": int(area),
                        "perimeter": int(perimeter),
                        "circularity": round(circularity, 3),
                        "type": classify_room(area)
                    })
    
    return rooms

# ==========================================
# STATISTICS
# ==========================================
def calculate_statistics(rooms):
    """Calculate analysis statistics"""
    if not rooms:
        return {}
    
    areas = [room['area'] for room in rooms]
    types = [room['type'] for room in rooms]
    
    stats = {
        'total_rooms': len(rooms),
        'total_area': sum(areas),
        'avg_area': int(np.mean(areas)),
        'max_area': max(areas),
        'min_area': min(areas),
        'room_types': dict(Counter(types)),
        'area_per_sqft': sum(areas) / 1000
    }
    
    return stats

# ==========================================
# VISUALIZATION
# ==========================================
def draw_results(img_rgb, rooms, show_labels=True, show_ids=True):
    """Draw detection results on image"""
    img_display = img_rgb.copy()
    
    colors = {
        'Hall': (255, 107, 107),
        'Bedroom': (74, 144, 226),
        'Kitchen': (76, 175, 80),
        'Bathroom': (255, 152, 0),
        'Living Room': (156, 39, 176),
        'Closet': (158, 158, 158),
        'Unknown': (100, 100, 100)
    }
    
    for room in rooms:
        x, y, w, h = room["x"], room["y"], room["w"], room["h"]
        room_type = room["type"]
        color = colors.get(room_type, (100, 100, 100))
        
        # Draw rectangle
        cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 3)
        
        # Draw label
        if show_labels:
            label = f"{room_type}" + (f" ({room['id']})" if show_ids else "")
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            y_label = max(y - 25, 0)
            
            cv2.rectangle(img_display, (x, y_label), (x + text_size[0] + 10, y_label + text_size[1] + 8), color, -1)
            cv2.putText(img_display, label, (x + 5, y_label + text_size[1] + 5), font, font_scale, (255, 255, 255), thickness)
    
    return img_display

# ==========================================
# EXPORT
# ==========================================
def export_to_csv(rooms):
    """Export results to CSV"""
    df = pd.DataFrame([
        {
            'Room ID': room['id'],
            'Type': room['type'],
            'Area (pixels)': room['area'],
            'Width (pixels)': room['w'],
            'Height (pixels)': room['h'],
            'Perimeter (pixels)': room['perimeter'],
        }
        for room in rooms
    ])
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# HEADER
# ==========================================
st.markdown("""
    <div class="header-container">
        <h1>🏗️ AI Floor Plan Analyzer Pro</h1>
        <p>Advanced intelligent detection and analysis of architectural floor plans</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    show_labels = st.checkbox("Show Room Labels", value=True)
    show_ids = st.checkbox("Show Room IDs", value=True)
    show_confidence = st.checkbox("Show Detection Confidence", value=False)
    
    st.markdown("---")
    
    st.subheader("📊 Analysis History")
    if st.session_state.analysis_history:
        for i, analysis in enumerate(st.session_state.analysis_history[-5:], 1):
            st.write(f"{i}. {analysis['filename']} - {analysis['rooms']} rooms")
    else:
        st.write("No analyses yet")

# ==========================================
# MAIN CONTENT
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Floor Plan")
    uploaded_file = st.file_uploader(
        "Drag and drop your floor plan image here",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )

with col2:
    st.subheader("📋 Quick Guide")
    st.info("""
    **How it works:**
    1. Upload a floor plan image
    2. AI detects rooms automatically
    3. Review and export results
    
    **Best results with:**
    - Color-coded rooms
    - Good resolution
    - Clear boundaries
    """)

st.markdown("---")

# ==========================================
# PROCESSING
# ==========================================
if uploaded_file is not None:
    try:
        with st.spinner("🔍 Analyzing floor plan..."):
            img_rgb = preprocess_image(uploaded_file)
            rooms = detect_rooms(img_rgb)
            stats = calculate_statistics(rooms)
            
            st.session_state.analysis_history.append({
                'filename': uploaded_file.name,
                'rooms': len(rooms),
                'timestamp': datetime.now()
            })
        
        if rooms:
            st.success(f"✅ Successfully detected {len(rooms)} rooms!")
        else:
            st.warning("⚠️ No rooms detected. Try a different image.")
        
        st.markdown("---")
        
        # ==========================================
        # RESULTS TABS
        # ==========================================
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Visualization", "📊 Statistics", "📋 Details", "💾 Export"])
        
        # TAB 1: VISUALIZATION
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                img_display = draw_results(img_rgb, rooms, show_labels, show_ids)
                st.image(img_display, caption="Detected Rooms", use_column_width=True)
            
            with col2:
                st.subheader("Room Legend")
                legend = {
                    'Hall': '🔴',
                    'Bedroom': '🔵',
                    'Kitchen': '🟢',
                    'Bathroom': '🟠',
                    'Living Room': '🟣',
                    'Closet': '⚪',
                }
                for room_type, emoji in legend.items():
                    st.write(f"{emoji} {room_type}")
        
        # TAB 2: STATISTICS
        with tab2:
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rooms", stats['total_rooms'])
                with col2:
                    st.metric("Total Area", f"{stats['total_area']:,} px²")
                with col3:
                    st.metric("Average Area", f"{stats['avg_area']:,} px²")
                with col4:
                    st.metric("Approx. Area", f"{stats['area_per_sqft']:.1f} m²")
                
                st.markdown("---")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Room Type Distribution")
                    room_counts = pd.DataFrame(
                        list(stats['room_types'].items()),
                        columns=['Room Type', 'Count']
                    )
                    st.bar_chart(room_counts.set_index('Room Type'))
                
                with col2:
                    st.subheader("Room Type Breakdown")
                    for room_type, count in stats['room_types'].items():
                        st.write(f"**{room_type}:** {count} room(s)")
        
        # TAB 3: DETAILS
        with tab3:
            if rooms:
                st.subheader("Detailed Room Information")
                
                df_rooms = pd.DataFrame([
                    {
                        'ID': room['id'],
                        'Type': room['type'],
                        'Area (px²)': f"{room['area']:,}",
                        'Width': f"{room['w']}px",
                        'Height': f"{room['h']}px",
                        'Perimeter': f"{room['perimeter']}px",
                        'Confidence': f"{room['circularity']:.2f}"
                    }
                    for room in sorted(rooms, key=lambda x: x['area'], reverse=True)
                ])
                
                st.dataframe(df_rooms, use_container_width=True, hide_index=True)
                
                if show_confidence:
                    st.info("**Confidence Score**: Circularity metric (closer to 0.5-0.7 is better for rooms)")
        
        # TAB 4: EXPORT
        with tab4:
            st.subheader("Export Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = export_to_csv(rooms)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"floor_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                img_display = draw_results(img_rgb, rooms, show_labels, show_ids)
                is_success, buffer = cv2.imencode('.png', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
                
                st.download_button(
                    label="🖼️ Download Image",
                    data=buffer.tobytes(),
                    file_name=f"floor_plan_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            
            with col3:
                summary_text = f"""FLOOR PLAN ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

SUMMARY
Total Rooms: {stats['total_rooms']}
Total Area: {stats['total_area']:,} pixels
Average Area: {stats['avg_area']:,} pixels
Approx. Area: {stats['area_per_sqft']:.1f} m²

ROOM BREAKDOWN
"""
                for room_type, count in stats['room_types'].items():
                    summary_text += f"{room_type}: {count}\n"
                
                st.download_button(
                    label="📄 Download Report",
                    data=summary_text,
                    file_name=f"floor_plan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try uploading a different image.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.875rem; margin-top: 2rem;'>
        <p>AI Floor Plan Analyzer Pro v1.0 | Powered by Advanced Computer Vision</p>
        <p>© 2024 All rights reserved</p>
    </div>
""", unsafe_allow_html=True)
