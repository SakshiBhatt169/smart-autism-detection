from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from keras.models import load_model
from datetime import datetime
from io import BytesIO
import gdown
import mysql.connector
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import black, HexColor
from functools import wraps

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'autism_detection_model.h5')

# Google Drive file ID for your model file
# (extract from your shareable link)
GDRIVE_FILE_ID = '1s5OimbbO_ZRaRgUSTyETKdBeWK4tPeVU'

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Model file already exists, skipping download.")

# At app startup:
download_model()
model = load_model(MODEL_PATH)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# File Upload Config
UPLOAD_FOLDER = 'backend/uploads/'
HEATMAP_FOLDER = 'backend/heatmaps/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model('model/autism_detection_model.h5')

# MySQL Configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="smart_autism_db"
)
cursor = db.cursor()

# Quiz Questions
quiz_questions = [
    "Does your child show difficulties with social interactions?",
    "Does your child avoid eye contact?",
    "Does your child repeat certain actions or phrases?",
    "Does your child show unusual interests or behaviors?",
    "Does your child show sensitivity to sensory experiences?",
    "Does your child struggle with communication?",
    "Does your child demonstrate repetitive movements?"
]

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    image_uploaded = session.get('image_uploaded', False)
    quiz_attempted = session.get('quiz_attempted', False)
    both_done = image_uploaded and quiz_attempted
    return render_template('index.html', both_done=both_done)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        try:
            cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                           (username, email, password))
            db.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Username already exists.', 'danger')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    image = request.files.get('image')

    if not image or image.filename == '':
        flash('No image selected.', 'danger')
        return redirect(url_for('upload_page'))

    if allowed_file(image.filename):
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img_array = np.expand_dims(img, axis=0)

        prediction = model.predict(img_array)[0][0]
        session['image_result'] = float(prediction)
        session['image_uploaded'] = True

        flash('Image processed successfully!', 'success')
        return redirect(url_for('index'))
    else:
        flash('Invalid image format.', 'danger')
        return redirect(url_for('upload_page'))

@app.route('/quiz', methods=['GET', 'POST'])
@login_required
def quiz():
    if request.method == 'POST':
        answers = [request.form.get(f'q{i}') for i in range(1, 8)]
        if None in answers:
            flash('Please answer all questions.', 'danger')
            return redirect(url_for('quiz'))

        score = sum(1 if a == 'yes' else 0.5 if a == 'sometimes' else 0 for a in answers)
        session['quiz_result'] = round(score / 7, 2)
        session['quiz_attempted'] = True
        session['quiz_answers'] = answers

        flash('Quiz submitted successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('quiz.html', questions=quiz_questions)

@app.route('/result')
@login_required
def result():
    image_uploaded = session.get('image_uploaded', False)
    quiz_attempted = session.get('quiz_attempted', False)

    image_display = "Not Attempted"
    quiz_display = "Not Attempted"
    combined_result = "Not Available"
    guidance_message = "Please complete both the quiz and image upload to view results."

    if image_uploaded and 'image_result' in session:
        score = session['image_result']
        if score >= 0.7:
            image_display = 'High Risk'
        elif score >= 0.4:
            image_display = 'Moderate Risk'
        else:
            image_display = 'Low Risk'

    if quiz_attempted and 'quiz_result' in session:
        score = session['quiz_result']
        if score >= 0.7:
            quiz_display = 'High Risk'
        elif score >= 0.4:
            quiz_display = 'Moderate Risk'
        else:
            quiz_display = 'Low Risk'

    if image_display != "Not Attempted" or quiz_display != "Not Attempted":
        risk_map = {'Not Attempted': 0, 'Low Risk': 1, 'Moderate Risk': 2, 'High Risk': 3}
        combined = max(risk_map[image_display], risk_map[quiz_display])
        combined_result = [k for k, v in risk_map.items() if v == combined][0]
        session['combined_result'] = combined_result

        if combined_result == 'Low Risk':
            guidance_message = "No immediate concern. Regular monitoring is advised."
        elif combined_result == 'Moderate Risk':
            guidance_message = "Some signs detected. Monitoring and follow-up recommended."
        elif combined_result == 'High Risk':
            guidance_message = "High risk indicators detected. Please consult a healthcare professional."

    return render_template('result.html',
                           image_result=image_display,
                           quiz_result=quiz_display,
                           combined_result=combined_result,
                           guidance_message=guidance_message)

@app.route('/download_pdf')
@login_required
def download_pdf():
    image_score = session.get('image_result', 'Not Available')
    quiz_score = session.get('quiz_result', 'Not Available')
    combined_result = session.get('combined_result', 'Not Available')
    answers = session.get('quiz_answers', [])

    def get_label(score):
        if isinstance(score, (float, int)):
            if score >= 0.7:
                return 'High Risk'
            elif score >= 0.4:
                return 'Moderate Risk'
            else:
                return 'Low Risk'
        return 'Not Available'

    image_risk = get_label(image_score)
    quiz_risk = get_label(quiz_score)

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(HexColor("#003366"))
    c.drawCentredString(width / 2, height - 50, "Autism Detection Report")

    c.setStrokeColor(black)
    c.rect(50, 100, width - 100, height - 150, stroke=1)

    c.setFont("Helvetica", 12)
    c.setFillColor(black)
    y = height - 90
    c.drawString(70, y, f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    y -= 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(70, y, "Image Prediction:")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(90, y, f"Score: {image_score}  |  Risk Level: {image_risk}")

    y -= 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(70, y, "Quiz Result:")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(90, y, f"Score: {quiz_score}  |  Risk Level: {quiz_risk}")

    y -= 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(70, y, f"Combined Result: {combined_result}")

    if answers:
        y -= 40
        c.setFont("Helvetica-Bold", 13)
        c.drawString(70, y, "Quiz Answers:")
        c.setFont("Helvetica", 11)
        y -= 20
        for i, ans in enumerate(answers):
            c.drawString(90, y, f"Q{i+1}: {quiz_questions[i]} → {ans}")
            y -= 18
            if y < 120:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="autism_report.pdf", mimetype='application/pdf')
