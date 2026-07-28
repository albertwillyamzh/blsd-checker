import os
import pickle
import sqlite3
import cv2
import numpy as np
import uuid
import joblib
from rbf_model import RBFNN_Improved
from flask import Flask, render_template, request, redirect, url_for, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
from static.plots.generate_plots import (
    generate_heatmap,
    generate_confusion_matrix,
    generate_roc
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "DEVELOPMENT_SECRET_KEY")

app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)

DATABASE = "database.db"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
        g._database = db
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                password TEXT,
                role TEXT DEFAULT 'user'
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()

scaler = joblib.load("model/best_scaler_revisi.pkl")
pca = joblib.load("model/best_pca_revisi.pkl")
model = joblib.load("model/best_model_revisi.pkl")
label_encoder = joblib.load("model/label_encoder_revisi.pkl")

CLASSES = label_encoder.classes_.tolist()

IMG_SIZE = (64, 64)

def preprocess_image_flatten(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image cannot be read")

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img.astype(np.float32) / 255.0

    X_flat = img_norm.reshape(1, -1)
    X_scaled = scaler.transform(X_flat)
    X_pca = pca.transform(X_scaled)

    return X_pca

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        ).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not password or not confirm_password:
            return render_template("register.html", message="All fields are required")

        if password != confirm_password:
            return render_template("register.html", message="Passwords do not match")

        if len(password) < 6:
            return render_template("register.html", message="Password must be at least 6 characters")

        password_hash = generate_password_hash(password)

        db = get_db()

        existing = db.execute(
            "SELECT id FROM users WHERE username=?",
            (username,)
        ).fetchone()

        if existing:
            return render_template("register.html", message="Username already exists")

        try:
            db.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, password_hash, "user")
            )
            db.commit()
        except Exception as e:
            return render_template("register.html", message=str(e))

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def home():
    return render_template("home.html")

@app.route("/classification")
@login_required
def classification():
    return render_template("index.html", prediction=None, image_file=None)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return redirect(url_for("classification"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("classification"))

    if not allowed_file(file.filename):
        return redirect(url_for("classification"))

    ext = file.filename.rsplit(".", 1)[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        X = preprocess_image_flatten(save_path)
        probs = model.predict_proba(X)[0]

        top2_idx = np.argsort(probs)[-2:][::-1]

        top2_probs = probs[top2_idx]
        sum_top2 = np.sum(top2_probs)

        results = []
        for idx, p in zip(top2_idx, top2_probs):
            normalized_prob = (p / sum_top2) * 100
            results.append({
                "label": CLASSES[idx],
                "prob": round(float(normalized_prob), 2)
            })
        
        results = sorted(results, key=lambda x: x["prob"], reverse=True)

        wib_time = datetime.utcnow() + timedelta(hours=7)

        result_text = ", ".join([f"{r['label']} ({r['prob']}%)" for r in results])

        db = get_db()
        db.execute("""
            INSERT INTO history (user_id, filename, result, created_at)
            VALUES (?, ?, ?, ?)
        """, (session["user_id"], filename, result_text, wib_time))
        db.commit()

        return render_template("index.html", prediction=results, image_file=filename)

    except Exception as e:
        print("Prediction error:", e)
        return render_template("index.html", prediction=None, image_file=None)

@app.route("/history")
@login_required
def history():
    db = get_db()
    rows = db.execute("""
        SELECT * FROM history
        WHERE user_id=?
        ORDER BY id DESC
    """, (session["user_id"],)).fetchall()

    return render_template("history.html", history=rows)

@app.route("/delete_history", methods=["POST"])
@login_required
def delete_history():
    selected_ids = request.form.getlist("selected")
    if not selected_ids:
        return redirect(url_for("history"))

    db = get_db()
    placeholders = ",".join("?" * len(selected_ids))

    rows = db.execute(
        f"SELECT filename FROM history WHERE id IN ({placeholders}) AND user_id=?",
        (*selected_ids, session["user_id"])
    ).fetchall()

    for row in rows:
        path = os.path.join(UPLOAD_FOLDER, row["filename"])
        if os.path.exists(path):
            os.remove(path)

    db.execute(
        f"DELETE FROM history WHERE id IN ({placeholders}) AND user_id=?",
        (*selected_ids, session["user_id"])
    )
    db.commit()

    return redirect(url_for("history"))

@app.route("/documentation")
@login_required
def documentation():
    if session.get("role") != "admin":
        return redirect(url_for("home"))

    # contoh data
    cm = np.array([[45,2,1,0],[3,40,2,1],[0,2,43,1],[1,0,2,44]])
    fpr = [0,0.1,0.2,0.4,1]
    tpr = [0,0.6,0.8,0.9,1]

    heatmap_path = generate_heatmap(cm)
    cm_path = generate_confusion_matrix(cm)
    roc_path = generate_roc(fpr, tpr)

    return render_template(
        "documentation.html",
        heatmap_img=heatmap_path,
        cm_img=cm_path,
        roc_img=roc_path
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
