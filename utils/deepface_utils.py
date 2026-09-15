import os
import logging
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   filename='deepface.log')
logger = logging.getLogger(__name__)

# Try to import DeepFace with error handling
deepface_available = False
try:
    from deepface import DeepFace
    deepface_available = True
    logger.info("DeepFace imported successfully")
except ImportError as e:
    logger.error(f"Error importing DeepFace: {str(e)}")
    
def verify_faces(
    classroom_image_path,
    students,
    threshold=0.4,
    model_name="Facenet512",
    detector_backend="retinaface",
    distance_metric="cosine",
    return_confidence=False,
):
    if not deepface_available:
        logger.error("DeepFace is not available. Cannot verify faces.")
        return ([], []) if return_confidence else []

    if not os.path.exists(classroom_image_path):
        logger.error(f"Classroom image not found: {classroom_image_path}")
        return ([], []) if return_confidence else []

    try:
        dfs = DeepFace.find(
            img_path=classroom_image_path,
            db_path="faces",   # VERY IMPORTANT
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=False,
            align=True,
            silent=True,
        )

        matched_rolls = set()
        confidence_scores = []

        if isinstance(dfs, list):
            for df in dfs:
                if df is None or df.empty:
                    continue

                # keep only strong matches
                if "distance" in df.columns:
                    df = df[df["distance"] <= threshold]

                for _, row in df.iterrows():
                    identity_path = str(row["identity"])

                    # extract roll number from filename
                    roll_no = os.path.splitext(os.path.basename(identity_path))[0]
                    matched_rolls.add(roll_no)

                    if "distance" in row:
                        confidence = max(0.0, 1.0 - float(row["distance"]))
                        confidence_scores.append(confidence)

        logger.info(f"Students passed in: {[student['roll_no'] for student in students]}")
        logger.info(f"Matched rolls from DeepFace: {list(matched_rolls)}")
        
        present_students = [
            student for student in students
            if student["roll_no"] in matched_rolls
        ]

        logger.info(f"Recognized {len(present_students)} students")

        if return_confidence:
            return present_students, confidence_scores

        return present_students

    except Exception as e:
        logger.error(f"Error in verify_faces: {str(e)}")
        return ([], []) if return_confidence else []

def extract_embedding(image_path, model_name="Facenet512"):
    """Extract facial embedding from an image"""
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=False
        )
        
        if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0:
            return embedding_objs[0].get("embedding")
        return None
    except Exception as e:
        logger.error(f"Error extracting embedding: {str(e)}")
        return None

def cosine_distance(vector1, vector2):
    """Calculate cosine distance between two vectors"""
    if vector1 is None or vector2 is None:
        return 1.0  # Maximum distance
    
    try:
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        
        dot = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
            
        cosine_similarity = dot / (norm1 * norm2)
        # Convert similarity to distance (1 - similarity)
        return 1.0 - cosine_similarity
    except Exception as e:
        logger.error(f"Error calculating cosine distance: {str(e)}")
        return 1.0

def detect_faces(image_path):
    """
    Detect all faces in an image
    
    Args:
        image_path: Path to the image
        
    Returns:
        Number of faces detected
    """
    if not deepface_available:
        return 0
        
    try:
        # Extract all faces from the image
        detected_faces = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=False
        )
        
        return len(detected_faces)
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return 0

def detect_faces_with_details(image_path):
    """
    Detect faces in an image and return detailed information
    
    Args:
        image_path: Path to the image
        
    Returns:
        Tuple of (detected_faces, face_locations)
        - detected_faces: List of face objects detected by DeepFace
        - face_locations: List of face location dictionaries with x, y, w, h
    """
    if not deepface_available:
        return [], []
        
    try:
        # Extract all faces from the image with details
        detected_faces = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=False,
            align=True,
            detector_backend="opencv"
        )
        
        # Extract face locations
        face_locations = []
        for face in detected_faces:
            try:
                # Get the region information
                region = face.get('facial_area', {})
                if region:
                    face_locations.append({
                        'x': region.get('x', 0),
                        'y': region.get('y', 0),
                        'w': region.get('w', 0),
                        'h': region.get('h', 0)
                    })
            except Exception as e:
                logger.error(f"Error extracting face location: {str(e)}")
        
        return detected_faces, face_locations
    except Exception as e:
        logger.error(f"Error detecting faces with details: {str(e)}")
        return [], []

def save_session_stats(stats):
    """
    Save recognition session statistics to session state
    
    Args:
        stats: Dictionary with recognition statistics
    """
    try:
        import streamlit as st
        
        # Initialize stats if not already present
        if 'recognition_stats' not in st.session_state:
            st.session_state.recognition_stats = {
                'total_sessions': 0,
                'total_faces_detected': 0,
                'total_students_recognized': 0,
                'avg_processing_time': 0,
                'avg_confidence': 0,
                'recognition_rate': 0,
                'history': []
            }
        
        current_stats = st.session_state.recognition_stats
        
        # Update running statistics
        current_stats['total_sessions'] += 1
        current_stats['total_faces_detected'] += stats.get('detected_faces', 0)
        current_stats['total_students_recognized'] += stats.get('recognized_students', 0)
        
        # Update averages
        current_stats['avg_processing_time'] = (
            (current_stats['avg_processing_time'] * (current_stats['total_sessions'] - 1) + 
             stats.get('processing_time', 0)) / current_stats['total_sessions']
        )
        
        # Update confidence if available
        if 'avg_confidence' in stats:
            current_stats['avg_confidence'] = (
                (current_stats['avg_confidence'] * (current_stats['total_sessions'] - 1) + 
                 stats.get('avg_confidence', 0)) / current_stats['total_sessions']
            )
        
        # Calculate overall recognition rate
        if current_stats['total_faces_detected'] > 0:
            current_stats['recognition_rate'] = (
                current_stats['total_students_recognized'] / 
                current_stats['total_faces_detected'] * 100
            )
        
        # Add to history
        current_stats['history'].append({
            'datetime': stats.get('datetime', ''),
            'processing_time': stats.get('processing_time', 0),
            'detected_faces': stats.get('detected_faces', 0),
            'recognized_students': stats.get('recognized_students', 0),
            'recognition_rate': stats.get('recognition_rate', 0),
            'avg_confidence': stats.get('avg_confidence', 0)
        })
        
        # Keep last session details
        current_stats['last_session'] = {
            'datetime': stats.get('datetime', ''),
            'subject': stats.get('subject', ''),
            'period': stats.get('period', ''),
            'total_faces': stats.get('detected_faces', 0),
            'recognized_count': stats.get('recognized_students', 0)
        }
        
    except Exception as e:
        logger.error(f"Error saving session statistics: {str(e)}") 