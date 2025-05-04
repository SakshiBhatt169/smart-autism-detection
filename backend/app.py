from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, HexColor
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configuration
UPLOAD_FOLDER = 'backend/uploads/'
HEATMAP_FOLDER = 'backend/heatmaps/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

# Load the trained model
model = load_model('model/autism_detection_model.h5')

# Quiz questions
quiz_questions = [
    "Does your child show difficulties with social interactions?",
    "Does your child avoid eye contact?",
    "Does your child repeat certain actions or phrases?",
    "Does your child show unusual interests or behaviors?",
    "Does your child show sensitivity to sensory experiences?",
    "Does your child struggle with communication?",
    "Does your child demonstrate repetitive movements?"
]

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/')
def index():
    image_uploaded = session.get('image_uploaded', False)
    quiz_attempted = session.get('quiz_attempted', False)
    both_done = image_uploaded and quiz_attempted
    return render_template('index.html', both_done=both_done)

@app.route('/exit')
def exit_app():
    session.clear()
    return redirect(url_for('goodbye'))

@app.route('/goodbye')
def goodbye():
    return render_template('goodbye.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    image = request.files.get('image')

    if not image or image.filename == '':
        flash('No image selected.', 'danger')
        return redirect(url_for('upload_page'))

    if allowed_file(image.filename):
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        # Preprocess the image for prediction
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img_array = np.expand_dims(img, axis=0)

        prediction = model.predict(img_array)[0][0]
        session['image_result'] = float(prediction)
        session['image_uploaded'] = True

        flash('Image uploaded and processed successfully! Please complete the quiz.', 'success')
        return redirect(url_for('index'))
    else:
        flash('Invalid image format.', 'danger')
        return redirect(url_for('upload_page'))

@app.route('/quiz', methods=['GET', 'POST'])
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

        flash('Quiz submitted successfully! Please upload an image if not already done.', 'success')
        return redirect(url_for('index'))

    return render_template('quiz.html', questions=quiz_questions)

@app.route('/result')
def result():
    image_uploaded = session.get('image_uploaded', False)
    quiz_attempted = session.get('quiz_attempted', False)

    image_display = "Not Attempted"
    quiz_display = "Not Attempted"
    combined_result = "Not Available"
    guidance_message = "Please complete both the quiz and image upload to view results."

    if image_uploaded and 'image_result' in session:
        image_score = session['image_result']
        if image_score >= 0.7:
            image_display = 'High Risk'
        elif image_score >= 0.4:
            image_display = 'Moderate Risk'
        else:
            image_display = 'Low Risk'

    if quiz_attempted and 'quiz_result' in session:
        quiz_score = session['quiz_result']
        if quiz_score >= 0.7:
            quiz_display = 'High Risk'
        elif quiz_score >= 0.4:
            quiz_display = 'Moderate Risk'
        else:
            quiz_display = 'Low Risk'

    if image_display != "Not Attempted" or quiz_display != "Not Attempted":
        risk_levels = {'Not Attempted': 0, 'Low Risk': 1, 'Moderate Risk': 2, 'High Risk': 3}
        combined_score = max(risk_levels[image_display], risk_levels[quiz_display])
        combined_result = [k for k, v in risk_levels.items() if v == combined_score][0]
        session['combined_result'] = combined_result

        if combined_result == 'Low Risk':
            guidance_message = "No immediate concern. Regular monitoring is advised."
        elif combined_result == 'Moderate Risk':
            guidance_message = "Some signs detected. Monitoring and follow-up recommended."
        elif combined_result == 'High Risk':
            guidance_message = "High risk indicators detected. Please consult a healthcare professional."

    return render_template(
        'result.html',
        image_result=image_display,
        quiz_result=quiz_display,
        combined_result=combined_result,
        risk_level=combined_result,
        guidance_message=guidance_message
    )

@app.route('/download_pdf')
def download_pdf():
    image_score = session.get('image_result', 'Not Available')
    quiz_score = session.get('quiz_result', 'Not Available')
    combined_result = session.get('combined_result', 'Not Available')
    answers = session.get('quiz_answers', [])

    def get_risk_label(score):
        if isinstance(score, (float, int)):
            if score >= 0.7:
                return 'High Risk'
            elif score >= 0.4:
                return 'Moderate Risk'
            else:
                return 'Low Risk'
        return 'Not Available'

    image_risk = get_risk_label(image_score)
    quiz_risk = get_risk_label(quiz_score)

    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(HexColor("#003366"))
    c.drawCentredString(width / 2, height - 50, "Autism Detection Report")

    # Border
    c.setStrokeColor(black)
    c.rect(50, 100, width - 100, height - 150, stroke=1)

    # Report Content
    c.setFont("Helvetica", 12)
    c.setFillColor(black)
    y = height - 90
    c.drawString(70, y, f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    y -= 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(70, y, "Image Prediction:")

    c.setFont("Helvetica", 12)
    y -= 20
    c.drawString(90, y, f"Image Result Score: {image_score}")
    y -= 20
    c.drawString(90, y, f"Risk Level: {image_risk}")

    y -= 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(70, y, "Quiz Result:")

    c.setFont("Helvetica", 12)
    y -= 20
    c.drawString(90, y, f"Quiz Result Score: {quiz_score}")
    y -= 20
    c.drawString(90, y, f"Risk Level: {quiz_risk}")

    y -= 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(70, y, f"Combined Result: {combined_result}")

    if answers:
        y -= 40
        c.setFont("Helvetica-Bold", 13)
        c.drawString(70, y, "Quiz Answers:")

        c.setFont("Helvetica", 11)
        y -= 20
        for i, answer in enumerate(answers):
            question = quiz_questions[i]
            c.drawString(90, y, f"Q{i+1}: {question} Answer: {answer}")
            y -= 18
            if y < 120:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

    c.save()
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name="autism_detection_report.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    app.run(debug=True)
