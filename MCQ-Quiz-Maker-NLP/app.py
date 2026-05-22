from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import spacy
import random
import pdfplumber
import os
import nltk
import json
import re
from datetime import datetime
from nltk.corpus import wordnet

# Download WordNet resource
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'quizzable_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quizzes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

db = SQLAlchemy(app)

@app.context_processor
def inject_builtins():
    return dict(chr=chr)

# ==========================================
# DATABASE MODELS
# ==========================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    quizzes = db.relationship('Quiz', backref='user', lazy=True, cascade="all, delete-orphan")
    attempts = db.relationship('QuizAttempt', backref='user', lazy=True, cascade="all, delete-orphan")

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    title = db.Column(db.String(255), nullable=False)
    text_content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    questions = db.relationship('Question', backref='quiz', lazy=True, cascade="all, delete-orphan")
    attempts = db.relationship('QuizAttempt', backref='quiz', lazy=True, cascade="all, delete-orphan")

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    choices_json = db.Column(db.Text, nullable=False)  # JSON list
    correct_answer = db.Column(db.String(10), nullable=False)  # A, B, C, D or True/False
    type = db.Column(db.String(50), nullable=False)  # MCQ, TF, FIB
    context_sentence = db.Column(db.Text, nullable=True)

    @property
    def choices(self):
        return json.loads(self.choices_json)

    @choices.setter
    def choices(self, value):
        self.choices_json = json.dumps(value)

class QuizAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    score = db.Column(db.Integer, nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    answers_json = db.Column(db.Text, nullable=False)  # JSON dictionary of {question_id: selected_choice}
    taken_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def answers(self):
        return json.loads(self.answers_json)

    @answers.setter
    def answers(self, value):
        self.answers_json = json.dumps(value)

with app.app_context():
    db.create_all()
    # Migration helper to add missing columns in existing SQLite DB
    try:
        from sqlalchemy import text
        db.session.execute(text("SELECT user_id FROM quiz LIMIT 1"))
    except Exception:
        db.session.rollback()
        try:
            db.session.execute(text("ALTER TABLE quiz ADD COLUMN user_id INTEGER REFERENCES user(id)"))
            db.session.commit()
            print("Successfully migrated 'quiz' table to include 'user_id'")
        except Exception as e:
            print("Error altering quiz table:", e)
            db.session.rollback()

    try:
        from sqlalchemy import text
        db.session.execute(text("SELECT user_id FROM quiz_attempt LIMIT 1"))
    except Exception:
        db.session.rollback()
        try:
            db.session.execute(text("ALTER TABLE quiz_attempt ADD COLUMN user_id INTEGER REFERENCES user(id)"))
            db.session.commit()
            print("Successfully migrated 'quiz_attempt' table to include 'user_id'")
        except Exception as e:
            print("Error altering quiz_attempt table:", e)
            db.session.rollback()

# ==========================================
# AUTHENTICATION HELPERS
# ==========================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_user():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    return dict(current_user=user)

# ==========================================
# NLP QUIZ GENERATOR LOGIC
# ==========================================

def make_mcq(sent_obj, s_text, nlp, doc):
    nouns = [t.text for t in sent_obj if t.pos_ in ["NOUN", "PROPN"] and len(t.text) > 2]
    if not nouns:
        return None
    
    subject = random.choice(nouns)
    pattern = re.compile(r'\b' + re.escape(subject) + r'\b', re.IGNORECASE)
    if not pattern.search(s_text):
        return None
    question_text = pattern.sub("_______", s_text, count=1)
    
    choices = [subject]
    synonyms = []
    for syn in wordnet.synsets(subject):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != subject.lower() and len(name) > 1:
                synonyms.append(name)
    
    distractors = list(set(synonyms))
    all_doc_nouns = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and len(t.text) > 2 and t.text.lower() != subject.lower()]
    random.shuffle(all_doc_nouns)
    
    for n in all_doc_nouns:
        if len(distractors) >= 5:
            break
        if n.lower() not in [d.lower() for d in distractors] and n.lower() != subject.lower():
            distractors.append(n)
            
    if len(distractors) < 3:
        fallback = ["Option A", "Option B", "Option C"]
        for f in fallback:
            if len(distractors) >= 3:
                break
            distractors.append(f)
            
    selected_distractors = random.sample(distractors, 3)
    choices.extend(selected_distractors)
    random.shuffle(choices)
    
    correct_index = choices.index(subject)
    correct_letter = chr(65 + correct_index)
    
    return {
        'question_text': question_text,
        'choices': choices,
        'correct_answer': correct_letter,
        'type': 'MCQ',
        'context_sentence': s_text
    }

def make_fib(sent_obj, s_text):
    candidates = [t for t in sent_obj if t.pos_ in ["NUM", "PROPN", "NOUN", "ADJ"] and len(t.text) > 2]
    if not candidates:
        return None
    
    candidates.sort(key=lambda t: 0 if t.pos_ in ["NUM", "PROPN"] else 1)
    target_token = candidates[0]
    target_word = target_token.text
    
    pattern = re.compile(r'\b' + re.escape(target_word) + r'\b')
    if not pattern.search(s_text):
        return None
    question_text = pattern.sub("_______", s_text, count=1)
    
    choices = [target_word]
    distractors = []
    doc = sent_obj.doc
    same_pos_words = [t.text for t in doc if t.pos_ == target_token.pos_ and t.text.lower() != target_word.lower() and len(t.text) > 2]
    random.shuffle(same_pos_words)
    
    for w in same_pos_words:
        if len(distractors) >= 3:
            break
        if w.lower() not in [d.lower() for d in distractors]:
            distractors.append(w)
            
    fallback_words = ["detail", "process", "concept", "element"]
    for f in fallback_words:
        if len(distractors) >= 3:
            break
        if f.lower() != target_word.lower() and f.lower() not in [d.lower() for d in distractors]:
            distractors.append(f)
            
    choices.extend(distractors[:3])
    random.shuffle(choices)
    
    correct_index = choices.index(target_word)
    correct_letter = chr(65 + correct_index)
    
    return {
        'question_text': question_text,
        'choices': choices,
        'correct_answer': correct_letter,
        'type': 'FIB',
        'context_sentence': s_text
    }

def make_tf(sent_obj, s_text, doc, difficulty):
    is_true = random.choice([True, False])
    
    if is_true:
        return {
            'question_text': s_text,
            'choices': ["True", "False"],
            'correct_answer': 'True',
            'type': 'TF',
            'context_sentence': s_text
        }
    else:
        tokens = [t for t in sent_obj if t.pos_ in ["ADJ", "NOUN", "VERB"] and not t.is_stop and len(t.text) > 2]
        if not tokens:
            return None
        
        target_token = random.choice(tokens)
        target_word = target_token.text
        
        antonym = None
        for syn in wordnet.synsets(target_word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonym = lemma.antonyms()[0].name().replace('_', ' ')
                    break
            if antonym:
                break
                
        replacement = antonym
        if not replacement:
            same_pos = [t.text for t in doc if t.pos_ == target_token.pos_ and t.text.lower() != target_word.lower() and len(t.text) > 2]
            if same_pos:
                replacement = random.choice(same_pos)
                
        if not replacement:
            return None
            
        pattern = re.compile(r'\b' + re.escape(target_word) + r'\b')
        if not pattern.search(s_text):
            return None
        modified_sentence = pattern.sub(lambda m: replacement, s_text, count=1)
        
        return {
            'question_text': modified_sentence,
            'choices': ["True", "False"],
            'correct_answer': 'False',
            'type': 'TF',
            'context_sentence': s_text
        }

def generate_quiz_data(text, num_questions, question_types, difficulty):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    sentences = []
    for sent in doc.sents:
        s_text = sent.text.strip().replace('\n', ' ')
        s_text = ' '.join(s_text.split())
        if len(s_text) > 40 and len(s_text) < 250:
            sentences.append((sent, s_text))
            
    if not sentences:
        return []
        
    if not question_types:
        question_types = ['MCQ']
        
    questions_per_type = max(1, num_questions // len(question_types))
    generated_questions = []
    used_sentences = set()
    random.shuffle(sentences)
    
    for q_type in question_types:
        type_count = 0
        for sent_obj, s_text in sentences:
            if type_count >= questions_per_type or len(generated_questions) >= num_questions:
                break
            if s_text in used_sentences:
                continue
                
            q_data = None
            if q_type == 'MCQ':
                q_data = make_mcq(sent_obj, s_text, nlp, doc)
            elif q_type == 'FIB':
                q_data = make_fib(sent_obj, s_text)
            elif q_type == 'TF':
                q_data = make_tf(sent_obj, s_text, doc, difficulty)
                
            if q_data:
                generated_questions.append(q_data)
                used_sentences.add(s_text)
                type_count += 1

    if len(generated_questions) < num_questions:
        for sent_obj, s_text in sentences:
            if len(generated_questions) >= num_questions:
                break
            if s_text in used_sentences:
                continue
            q_type = random.choice(question_types)
            q_data = None
            if q_type == 'MCQ':
                q_data = make_mcq(sent_obj, s_text, nlp, doc)
            elif q_type == 'FIB':
                q_data = make_fib(sent_obj, s_text)
            elif q_type == 'TF':
                q_data = make_tf(sent_obj, s_text, doc, difficulty)
                
            if q_data:
                generated_questions.append(q_data)
                used_sentences.add(s_text)
                
    random.shuffle(generated_questions)
    return generated_questions

# ==========================================
# FLASK ROUTING
# ==========================================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password:
            flash('Please enter a username and password.', 'error')
            return render_template('signup.html')
            
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('signup.html')
            
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
            
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username is already taken. Please choose another.', 'error')
            return render_template('signup.html')
            
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
            
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash('Invalid username or password.', 'error')
            return render_template('login.html')
            
        session['user_id'] = user.id
        flash(f'Welcome back, {user.username}!', 'success')
        return redirect(url_for('dashboard'))
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    uid = session['user_id']
    total_quizzes = Quiz.query.filter_by(user_id=uid).count()
    total_attempts = QuizAttempt.query.filter_by(user_id=uid).count()
    
    avg_score = 0
    if total_attempts > 0:
        attempts = QuizAttempt.query.filter_by(user_id=uid).all()
        percentages = [(a.score / (a.total_questions * 10)) * 100 for a in attempts]
        avg_score = round(sum(percentages) / total_attempts, 1)
        
    recent_quizzes = Quiz.query.filter_by(user_id=uid).order_by(Quiz.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                           total_quizzes=total_quizzes, 
                           total_attempts=total_attempts, 
                           avg_score=avg_score, 
                           recent_quizzes=recent_quizzes)

@app.route('/create')
@login_required
def create_quiz_view():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    title = request.form.get('quiz_title', 'Untitled Quiz')
    num_questions = int(request.form.get('num_questions', 5))
    difficulty = request.form.get('difficulty', 'medium')
    question_types = request.form.getlist('question_types')
    
    text = ""
    
    # 1. Text Paste
    if 'text_paste' in request.form and request.form.get('text_paste').strip():
        text = request.form.get('text_paste').strip()
    # 2. File Upload
    elif 'pdf_file' in request.files:
        file = request.files['pdf_file']
        if file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            if file.filename.endswith('.pdf'):
                try:
                    with pdfplumber.open(file_path) as pdf:
                        text = " ".join([page.extract_text() or "" for page in pdf.pages])
                except Exception as e:
                    print("Error extracting PDF text:", e)
                    text = ""
            elif file.filename.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except Exception as e:
                    print("Error reading text file:", e)
                    text = ""
                
    if not text or len(text.strip()) < 50:
        flash('Please provide at least 50 characters of readable text.', 'error')
        return redirect(url_for('create_quiz_view'))
        
    # Generate
    questions_list = generate_quiz_data(text, num_questions, question_types, difficulty)
    
    if not questions_list:
        flash('Could not generate any quiz questions from the provided text. Try a longer or more descriptive text block!', 'error')
        return redirect(url_for('create_quiz_view'))
        
    # Save Quiz to DB
    quiz = Quiz(title=title, text_content=text, user_id=session['user_id'])
    db.session.add(quiz)
    db.session.commit()
    
    # Save Questions to DB
    for q in questions_list:
        question = Question(
            quiz_id=quiz.id,
            question_text=q['question_text'],
            choices_json=json.dumps(q['choices']),
            correct_answer=q['correct_answer'],
            type=q['type'],
            context_sentence=q['context_sentence']
        )
        db.session.add(question)
    db.session.commit()
    
    return redirect(url_for('quiz_view', quiz_id=quiz.id))

@app.route('/quiz/<int:quiz_id>')
@login_required
def quiz_view(quiz_id):
    quiz = Quiz.query.filter_by(id=quiz_id, user_id=session['user_id']).first_or_404()
    questions = Question.query.filter_by(quiz_id=quiz.id).all()
    
    # Convert list of Question objects to structured list for JS frontend
    questions_data = []
    for q in questions:
        questions_data.append({
            'id': q.id,
            'question_text': q.question_text,
            'choices': q.choices,
            'correct_answer': q.correct_answer,
            'type': q.type
        })
        
    return render_template('questions.html', quiz=quiz, questions=questions_data)

@app.route('/quiz/<int:quiz_id>/submit', methods=['POST'])
@login_required
def quiz_submit(quiz_id):
    quiz = Quiz.query.filter_by(id=quiz_id, user_id=session['user_id']).first_or_404()
    data = request.json
    user_answers = data.get('answers', {})
    
    # Calculate score
    questions = Question.query.filter_by(quiz_id=quiz.id).all()
    score = 0
    total = len(questions)
    
    for q in questions:
        user_ans = user_answers.get(str(q.id))
        if user_ans == q.correct_answer:
            score += 10
            
    # Save Attempt
    attempt = QuizAttempt(
        quiz_id=quiz.id,
        user_id=session['user_id'],
        score=score,
        total_questions=total,
        answers_json=json.dumps(user_answers)
    )
    db.session.add(attempt)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'attempt_id': attempt.id
    })

@app.route('/attempt/<int:attempt_id>')
@login_required
def attempt_view(attempt_id):
    attempt = QuizAttempt.query.filter_by(id=attempt_id, user_id=session['user_id']).first_or_404()
    quiz = Quiz.query.get(attempt.quiz_id)
    questions = Question.query.filter_by(quiz_id=quiz.id).all()
    
    user_answers = attempt.answers
    
    review_data = []
    for q in questions:
        user_choice = user_answers.get(str(q.id), "Skipped")
        is_correct = user_choice == q.correct_answer
        review_data.append({
            'question_text': q.question_text,
            'choices': q.choices,
            'correct_answer': q.correct_answer,
            'user_choice': user_choice,
            'is_correct': is_correct,
            'type': q.type,
            'context': q.context_sentence
        })
        
    return render_template('results.html', 
                           attempt=attempt, 
                           quiz=quiz, 
                           review_data=review_data, 
                           score=attempt.score, 
                           total_questions=attempt.total_questions)

@app.route('/history')
@login_required
def history():
    uid = session['user_id']
    attempts = QuizAttempt.query.filter_by(user_id=uid).order_by(QuizAttempt.taken_at.desc()).all()
    history_items = []
    
    for a in attempts:
        quiz = Quiz.query.get(a.quiz_id)
        if quiz:
            history_items.append({
                'attempt_id': a.id,
                'quiz_id': quiz.id,
                'title': quiz.title,
                'score': a.score,
                'total_questions': a.total_questions,
                'date': a.taken_at.strftime('%Y-%m-%d %H:%M')
            })
            
    return render_template('history.html', history_items=history_items)

@app.route('/delete-quiz/<int:quiz_id>', methods=['POST'])
@login_required
def delete_quiz(quiz_id):
    quiz = Quiz.query.filter_by(id=quiz_id, user_id=session['user_id']).first_or_404()
    db.session.delete(quiz)
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/export/<int:quiz_id>/<string:format_type>')
@login_required
def export_quiz(quiz_id, format_type):
    quiz = Quiz.query.filter_by(id=quiz_id, user_id=session['user_id']).first_or_404()
    questions = Question.query.filter_by(quiz_id=quiz.id).all()
    
    if format_type == 'json':
        data = {
            'quiz_title': quiz.title,
            'created_at': quiz.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'questions': []
        }
        for q in questions:
            data['questions'].append({
                'id': q.id,
                'question_text': q.question_text,
                'choices': q.choices,
                'correct_answer': q.correct_answer,
                'type': q.type
            })
        response = make_response(json.dumps(data, indent=4))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=quiz_{quiz_id}.json'
        return response
        
    elif format_type == 'txt':
        lines = []
        lines.append(f"Quiz Title: {quiz.title}")
        lines.append(f"Generated: {quiz.created_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append("="*40)
        lines.append("")
        
        for idx, q in enumerate(questions, 1):
            lines.append(f"Q{idx}. {q.question_text}")
            if q.type == 'TF':
                lines.append("  [ ] True    [ ] False")
            else:
                for opt_idx, choice in enumerate(q.choices):
                    letter = chr(65 + opt_idx)
                    lines.append(f"  {letter}) {choice}")
            lines.append(f"Correct Answer: {q.correct_answer}")
            if q.context_sentence:
                lines.append(f"Context: {q.context_sentence}")
            lines.append("-" * 30)
            lines.append("")
            
        response = make_response("\n".join(lines))
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Content-Disposition'] = f'attachment; filename=quiz_{quiz_id}.txt'
        return response
        
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)