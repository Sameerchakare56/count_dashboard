from fastapi import FastAPI, HTTPException ,Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional , Dict , List
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Float, func, distinct
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import mysql.connector
import numpy as np


# ---- Database connection ----
DATABASE_URL = "mysql+pymysql://sameer:12345@127.0.0.1/people_counting"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---- MySQL config (for direct cursor operations)
db_config = {
    "host": "127.0.0.1",
    "user": "sameer",
    "password": "12345",
    "database": "people_counting"
}

# ---- Table model ----
class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    global_id = Column(String(50)) 
    local_id = Column(String(20))
    camera_id = Column(String(50))
    event_type = Column(String(20))
    ts = Column(DateTime)
    centroid_x = Column(Float)   # ✅ must exist in your DB
    centroid_y = Column(Float)   # ✅ must exist in your DB
    captured_image = Column(LargeBinary)

# ---- FastAPI setup ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_CAPACITY = 50


# -------------------------------------------------------------------
# ✅ DASHBOARD ENDPOINT (unchanged)
# -------------------------------------------------------------------
@app.get("/dashboard")

async def get_dashboard():
    """
    📊 Dashboard endpoint:
    - Aggregates per-camera stats for today (00:00:00 → 23:59:59)
    - Groups datas[] by minute (HH:MM)
    - Groups heatmapData[] by hour (HH:00)
    - Includes entries, exits, unique people, and density-based heatmap
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # -------------------------------
        # 📅 Today's date range
        # -------------------------------
        today = datetime.now()
        day_start = datetime(today.year, today.month, today.day, 0, 0, 0)
        day_end = day_start + timedelta(days=1)

        # -------------------------------
        # 🎯 Fetch all today's data
        # -------------------------------
        cursor.execute("""
            SELECT global_id, camera_id, event_type, ts, centroid_x, centroid_y
            FROM events
            WHERE ts BETWEEN %s AND %s
              AND centroid_x IS NOT NULL
              AND centroid_y IS NOT NULL
            ORDER BY ts ASC
        """, (day_start, day_end))
        rows = cursor.fetchall()

        if not rows:
            return {"status": "no_data", "message": "No data found for today"}

        # -------------------------------
        # 📸 Group by camera
        # -------------------------------
        camera_groups = {}
        for r in rows:
            cam = r["camera_id"]
            if cam not in camera_groups:
                camera_groups[cam] = []
            camera_groups[cam].append(r)

        # -------------------------------
        # 🧩 Build per-camera summary
        # -------------------------------
        Data = {}

        for cam_id, cam_events in camera_groups.items():
            # ✅ Unique global IDs
            unique_global_ids = set(r["global_id"] for r in cam_events if r["global_id"])

            # ✅ Group datas[] by minute
            minute_groups = {}
            for r in cam_events:
                minute_str = r["ts"].strftime("%H:%M")
                if minute_str not in minute_groups:
                    minute_groups[minute_str] = []
                minute_groups[minute_str].append(r)

            # ✅ Group heatmapData[] by hour
            hour_groups = {}
            for r in cam_events:
                hour_str = r["ts"].strftime("%H:00")  # group by hour only
                if hour_str not in hour_groups:
                    hour_groups[hour_str] = []
                hour_groups[hour_str].append(r)

            datas = []
            heatmap_points = []

            # -------------------------------
            # ⏱ Minute-wise 'datas'
            # -------------------------------
            for minute, events_in_minute in sorted(minute_groups.items()):
                xs = np.array([float(r["centroid_x"]) for r in events_in_minute])
                ys = np.array([float(r["centroid_y"]) for r in events_in_minute])

                count = len(set(r["global_id"] for r in events_in_minute))

                datas.append({
                    "time": minute,     # ✅ "12:03"
                    "count": count,
                    "color": "#4285F4"
                })

            # -------------------------------
            # 🔥 Hour-wise 'heatmapData'
            # -------------------------------
            for hour, events_in_hour in sorted(hour_groups.items()):
                xs = np.array([float(r["centroid_x"]) for r in events_in_hour])
                ys = np.array([float(r["centroid_y"]) for r in events_in_hour])

                for i in range(len(events_in_hour)):
                    distances = np.sqrt((xs - xs[i]) ** 2 + (ys - ys[i]) ** 2)
                    density = np.sum(distances < 50)
                    radius = np.clip((density / 15) * 25, 8, 25)

                    if radius < 10:
                        color = "rgba(144, 238, 144, 0.6)"
                    elif radius < 18:
                        color = "rgba(255, 165, 0, 0.6)"
                    else:
                        color = "rgba(255, 69, 0, 0.7)"

                    heatmap_points.append({
                        "timestamp": hour,  # ✅ group by hour (e.g., "14:00")
                        "x": int(events_in_hour[i]["centroid_x"]),
                        "y": int(events_in_hour[i]["centroid_y"]),
                        "r": round(float(radius), 2),
                        "color": color
                    })
            camera_info = [
                {
                    "id": cam_id,
                    "name": f"Camera {cam_id}",
                    "peopleCount": len(unique_global_ids),
                    "img": f"https://placehold.co/400x250/f0f0f0/333333?text=CAM{cam_id}",
                    "videoUrl": f"http://127.0.0.1:8002/stream/{cam_id}",
                }
            ]
            # -------------------------------
            # 🧮 Count entries/exits
            # -------------------------------
            entries = len(set(r["global_id"] for r in cam_events if r["event_type"] == "enter"))
            exits = len(set(r["global_id"] for r in cam_events if r["event_type"] == "exit"))
            inside = max(entries - exits, 0)

            # -------------------------------
            # 📦 Build final structure
            # -------------------------------
            Data[cam_id] = {
                "camera_info": camera_info,
                "uniquePeople": len(unique_global_ids),
                "entries": entries,
                "exits": exits,
                "peopleInside": inside,
                "maxCapacity": 50 if cam_id == "cam1" else 60,
                "datas": datas,               # ✅ minute-level
                "heatmapData": heatmap_points # ✅ hour-level
            }

        cursor.close()
        conn.close()

        return {"data": Data}

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {err}")



@app.post("/fetch_by_date")
async def fetch_by_date(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """
    Fetch all camera-wise unique global IDs and minute-wise heatmap data for a specific date.
    Each 'datas' entry includes its minute (HH:MM) and color.
    Example: /fetch_by_date?date=2025-10-31
    """

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        start_time = f"{date} 00:00:00"
        end_time = f"{date} 23:59:59"

        query = """
        SELECT global_id, camera_id, event_type, ts, centroid_x, centroid_y
        FROM events
        WHERE ts BETWEEN %s AND %s
        AND centroid_x IS NOT NULL
        AND centroid_y IS NOT NULL
        ORDER BY ts ASC
        """
        cursor.execute(query, (start_time, end_time))
        rows = cursor.fetchall()

        if not rows:
            return {"status": "no_data", "message": f"No data found for {date}"}

        # Group by camera
        camera_groups: Dict[str, List[Dict]] = {}
        for r in rows:
            cam = r["camera_id"]
            if cam not in camera_groups:
                camera_groups[cam] = []
            camera_groups[cam].append(r)

        result = {}

        for cam_id, cam_events in camera_groups.items():
            # ✅ Get unique global_ids for this camera
            unique_global_ids = set(r["global_id"] for r in cam_events if r["global_id"] is not None)

            # Group by minute (HH:MM)
            minute_groups: Dict[str, List[Dict]] = {}
            for r in cam_events:
                minute_str = r["ts"].strftime("%H:%M")
                if minute_str not in minute_groups:
                    minute_groups[minute_str] = []
                minute_groups[minute_str].append(r)

            datas = []
            flat_heatmap_points = []

            for minute, events_in_minute in sorted(minute_groups.items()):
                xs = np.array([float(r["centroid_x"]) for r in events_in_minute])
                ys = np.array([float(r["centroid_y"]) for r in events_in_minute])

                count = len(set(r["global_id"] for r in events_in_minute))  # ✅ unique by global_id per minute
                datas.append({
                    "time": minute,
                    "count": count,
                    "color": "#4285F4"  # ✅ constant color
                })

                # ✅ Generate heatmap points with time included
                for i in range(len(events_in_minute)):
                    distances = np.sqrt((xs - xs[i]) ** 2 + (ys - ys[i]) ** 2)
                    density = np.sum(distances < 50)
                    radius = np.clip((density / 15) * 25, 8, 25)

                    if radius < 10:
                        color = "rgba(144, 238, 144, 0.6)"
                    elif radius < 18:
                        color = "rgba(255, 165, 0, 0.6)"
                    else:
                        color = "rgba(255, 69, 0, 0.7)"

                    flat_heatmap_points.append({
                        "timestamp": minute,
                        "centroid_x": int(events_in_minute[i]["centroid_x"]),
                        "centroid_y": int(events_in_minute[i]["centroid_y"]),
                        "r": round(float(radius), 2),
                        "color": color
                    })

            # ✅ Entries/Exits logic (unique by global_id per camera)
            entries = len(set(r["global_id"] for r in cam_events if r["event_type"] == "enter"))
            exits = len(set(r["global_id"] for r in cam_events if r["event_type"] == "exit"))
            people_inside = max(entries - exits, 0)

            result[cam_id] = {
                "uniquePeople": len(unique_global_ids),
                "entries": entries,
                "exits": exits,
                "peopleInside": people_inside,
                "maxCapacity": MAX_CAPACITY,
                "datas": datas,
                "heatmapData": flat_heatmap_points
            }

        cursor.close()
        conn.close()

        return {"data": result}

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {err}")
