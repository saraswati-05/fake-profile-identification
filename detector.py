import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

# ---------- Flask Setup ----------
app = Flask(__name__)
app.secret_key = "change-me"

# ---------- Database (SQLite) ----------
Base = declarative_base()
DB_PATH = "app.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine))

# ---------- Auth ----------
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------- User Model ----------
class User(Base, UserMixin):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    full_name = Column(String(120))
    email = Column(String(120), unique=True)
    phone = Column(String(50))
    username = Column(String(80), unique=True)
    college = Column(String(120))
    semester = Column(String(50))
    password_hash = Column(String(255))
    is_admin = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=False)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    # Flask-Login requires these:
    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    return db.get(User,int(user_id))

# ---------- Utils ----------
FEATURES = ["followers","following","posts","account_age_days","is_verified","avg_likes","bio_length","has_profile_pic"]
MODEL_PATH = os.path.join("ml","model.pkl")
PLOTS_DIR = os.path.join("static","plots")
DATA_DIR = "data"
UPLOAD_DIR = "uploads"
os.makedirs("ml", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

def init_db():
    Base.metadata.create_all(engine)
    db = SessionLocal()
    # Create default admin if not exists
    admin = db.query(User).filter_by(email="admin@example.com").first()
    if not admin:
        admin = User(
            full_name="Admin",
            email="admin@example.com",
            phone="0000000000",
            username="admin",
            college="Admin College",
            semester="N/A",
            is_admin=True,
            is_approved=True
        )
        admin.set_password("admin123")
        db.add(admin)
        db.commit()
    db.close()

def load_or_generate_training_csv():
    csv_path = os.path.join(DATA_DIR, "training.csv")
    if os.path.exists(csv_path):
        return csv_path
    # Generate a small synthetic dataset as fallback
    rng = np.random.default_rng(42)
    n = 600
    followers = rng.integers(5, 5000, size=n)
    following = rng.integers(5, 5000, size=n)
    posts = rng.integers(0, 800, size=n)
    account_age_days = rng.integers(1, 4000, size=n)
    is_verified = rng.integers(0, 2, size=n)
    avg_likes = rng.integers(0, 1000, size=n)
    bio_length = rng.integers(0, 160, size=n)
    has_profile_pic = rng.integers(0, 2, size=n)

    # Heuristic: suspicious if following >> followers, very new, low posts/likes, no pic, not verified
    score = (
        (following / (followers+1)) * 0.8
        + (1/(account_age_days+1))*200
        + (1/(posts+1))*150
        + (1/(avg_likes+1))*200
        + (1-has_profile_pic)*0.8*100
        + (1-is_verified)*50
    )
    # Convert score to binary labels using a threshold
    label = (score > np.percentile(score, 65)).astype(int)

    df = pd.DataFrame({
        "followers": followers,
        "following": following,
        "posts": posts,
        "account_age_days": account_age_days,
        "is_verified": is_verified,
        "avg_likes": avg_likes,
        "bio_length": bio_length,
        "has_profile_pic": has_profile_pic,
        "label": label
    })
    df.to_csv(csv_path, index=False)
    # Also write a small test sample
    df.sample(12, random_state=7).drop(columns=["label"]).to_csv(os.path.join(DATA_DIR,"test_sample.csv"), index=False)
    return csv_path

def train_model():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.ensemble import RandomForestClassifier

    csv_path = load_or_generate_training_csv()
    df = pd.read_csv(csv_path)
    df = df.dropna()
    X = df[FEATURES]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Metrics
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix plot (single plot)
    plt.figure()
    # No specific colors set (defaults only)
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
    plt.close()

    # Feature importance
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure()
    plt.bar(range(len(FEATURES)), importances[idx])
    plt.xticks(range(len(FEATURES)), np.array(FEATURES)[idx], rotation=45, ha='right')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
    plt.close()

    return {"status":"ok","auc": float(roc_auc)}

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name","").strip()
        email = request.form.get("email","").strip().lower()
        phone = request.form.get("phone","").strip()
        username = request.form.get("username","").strip().lower()
        college = request.form.get("college","").strip()
        semester = request.form.get("semester","").strip()
        password = request.form.get("password","")

        if not (full_name and email and username and password):
            flash("Please fill in all required fields.")
            return redirect(url_for("register"))
        db = SessionLocal()
        if db.query(User).filter( (User.email==email) | (User.username==username) ).first():
            flash("Email or Username already exists.")
            db.close()
            return redirect(url_for("register"))
        user = User(full_name=full_name, email=email, phone=phone, username=username,
                    college=college, semester=semester, is_admin=False, is_approved=False)
        user.set_password(password)
        db.add(user)
        db.commit()
        db.close()
        flash("Registered! Wait for admin approval.")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").lower()
        password = request.form.get("password","")
        db = SessionLocal()
        user = db.query(User).filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            db.close()
            if not user.is_approved and not user.is_admin:
                flash("Login successful, but your account is pending admin approval.")
            return redirect(url_for("home"))
        db.close()
        flash("Invalid email or password.")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.")
    return redirect(url_for("home"))

def admin_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Admin only.")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

@app.route("/admin")
@login_required
@admin_required
def admin():
    db = SessionLocal()
    pending = db.query(User).filter_by(is_approved=False, is_admin=False).all()
    db.close()
    return render_template("admin.html", pending_users=pending)

@app.route("/users")
@login_required
@admin_required
def users():
    db = SessionLocal()
    all_users = db.query(User).all()
    db.close()
    return render_template("users.html", users=all_users)

@app.route("/approve/<int:user_id>")
@login_required
@admin_required
def approve_user(user_id):
    db = SessionLocal()
    u = db.get(User,user_id)
    if u:
        u.is_approved = True
        db.commit()
        flash(f"Approved {u.email}")
    db.close()
    return redirect(url_for("admin"))

@app.route("/reject/<int:user_id>")
@login_required
@admin_required
def reject_user(user_id):
    db = SessionLocal()
    u = db.query(User).get(user_id)
    if u:
        db.delete(u)
        db.commit()
        flash("User removed.")
    db.close()
    return redirect(url_for("admin"))

@app.route("/train")
@login_required
@admin_required
def train():
    return render_template("train.html")

@app.route("/train/run", methods=["POST"])
@login_required
@admin_required
def run_training():
    info = train_model()
    flash(f"Training complete. AUC={info['auc']:.3f}")
    return redirect(url_for("graphs"))

@app.route("/graphs")
@login_required
@admin_required
def graphs():
    return render_template("graphs.html")

@app.route("/predict", methods=["GET"])
@login_required
def predict():
    if not current_user.is_admin and not current_user.is_approved:
        flash("Your account needs admin approval to access this page.")
        return redirect(url_for("home"))
    return render_template("predict.html")

@app.route("/predict/csv", methods=["POST"])
@login_required
def predict_csv():
    if not current_user.is_admin and not current_user.is_approved:
        flash("Your account needs admin approval.")
        return redirect(url_for("home"))

    file = request.files.get("csv_file")
    if not file:
        flash("No file uploaded.")
        return redirect(url_for("predict"))
    path = os.path.join("uploads", "to_predict.csv")
    file.save(path)

    if not os.path.exists(MODEL_PATH):
        flash("Model not trained yet. Train first.")
        return redirect(url_for("train"))
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        flash(f"Missing columns: {missing}")
        return redirect(url_for("predict"))
    X = df[FEATURES]
    proba = model.predict_proba(X)[:,1]
    preds = (proba >= 0.5).astype(int)
    out = df.copy()
    out["prob_fake"] = proba.round(3)
    out["pred_label"] = preds
    # Render table
    return render_template("predict.html", table=out)

@app.route("/predict/one", methods=["POST"])
@login_required
def predict_single():
    if not current_user.is_admin and not current_user.is_approved:
        flash("Your account needs admin approval.")
        return redirect(url_for("home"))

    if not os.path.exists(MODEL_PATH):
        flash("Model not trained yet. Train first.")
        return redirect(url_for("train"))
    model = joblib.load(MODEL_PATH)

    def get_float(name, default=0.0):
        try:
            return float(request.form.get(name, default))
        except:
            return float(default)

    sample = np.array([[
        get_float("followers"),
        get_float("following"),
        get_float("posts"),
        get_float("account_age_days"),
        get_float("is_verified"),
        get_float("avg_likes"),
        get_float("bio_length"),
        get_float("has_profile_pic"),
    ]])
    prob_fake = float(model.predict_proba(sample)[0,1])
    pred = int(prob_fake >= 0.5)
    single_result = type("Obj", (), {"prob_fake": prob_fake, "pred_label": pred})
    return render_template("predict.html", single_result=single_result)

if __name__ == "__main__":
    init_db()
    # Start app
    app.run(debug=True)