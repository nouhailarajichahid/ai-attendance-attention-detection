# AI Attendance & Attention Detection

An AI-based student attendance and attention monitoring system using face recognition and computer vision techniques.

## Overview

This project combines face recognition with attention monitoring to automatically identify students, record attendance, and estimate their attention level during class sessions.

## Features

- Student face recognition for attendance tracking
- Automatic attendance recording
- Attention monitoring
- Head Pose Estimation (HPE)
- Eye Aspect Ratio (EAR) analysis
- Attention scoring
- Student database management
- Attendance data export to Excel
- Face image management
- Graphical user interface

## AI & Computer Vision

### Face Recognition
The system uses **FaceNet512** through DeepFace for student face recognition and identification.

### Attention Detection
The attention monitoring module combines:

- **Head Pose Estimation (HPE)** — analyzes the student's head orientation
- **Eye Aspect Ratio (EAR)** — helps detect eye closure and visual attention
- **Attention Scoring** — combines visual indicators to estimate an attention score

## Tech Stack

- **Language:** Python
- **Face Recognition:** DeepFace / FaceNet512
- **Computer Vision:** OpenCV
- **Attention Detection:** Head Pose Estimation, Eye Aspect Ratio
- **Database:** SQLite
- **Data Export:** Excel
- **GUI:** Python-based interface

## Project Structure

```text
AI Attendance & Attention Detection/
├── app/
├── db/
├── excel_exports/
├── faces/
├── utils/
│   ├── attention_detector.py
│   ├── attention_scorer.py
│   ├── db_utils.py
│   ├── deepface_utils.py
│   ├── eye_detector.py
│   ├── migrate_db.py
│   └── pose_estimation.py
├── add_attention_table.py
├── app.py
├── requirements
├── setup.py
├── run_app
├── run_app.sh
└── .gitignore
