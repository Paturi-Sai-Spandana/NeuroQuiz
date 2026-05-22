import unittest
from app import app, db, User, Quiz, QuizAttempt, Question

class MCQQuizMakerTestCase(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        self.app = app.test_client()
        
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_dashboard_redirects_to_login(self):
        """Unauthenticated user should be redirected to login page"""
        response = self.app.get('/', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.location)

    def test_signup_page_loads(self):
        """Signup page should load with status 200"""
        response = self.app.get('/signup')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Create Account', response.data)

    def test_successful_signup_and_login(self):
        """Creating an account and logging in should work seamlessly"""
        # 1. Sign up
        response = self.app.post('/signup', data={
            'username': 'testuser',
            'password': 'testpassword',
            'confirm_password': 'testpassword'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Account created successfully', response.data)

        # 2. Login
        response = self.app.post('/login', data={
            'username': 'testuser',
            'password': 'testpassword'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome back, testuser', response.data)
        self.assertIn(b'Dashboard', response.data)

        # 3. Log out
        response = self.app.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'You have been logged out', response.data)
        self.assertIn(b'Log In', response.data)

    def test_signup_validation(self):
        """Signup should validate password match, short password, and existing username"""
        # Passwords mismatch
        response = self.app.post('/signup', data={
            'username': 'testuser',
            'password': 'testpassword',
            'confirm_password': 'differentpassword'
        }, follow_redirects=True)
        self.assertIn(b'Passwords do not match', response.data)

        # Password too short
        response = self.app.post('/signup', data={
            'username': 'testuser',
            'password': '123',
            'confirm_password': '123'
        }, follow_redirects=True)
        self.assertIn(b'Password must be at least 6 characters long', response.data)

        # Duplicate username
        self.app.post('/signup', data={
            'username': 'duplicate',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        
        response = self.app.post('/signup', data={
            'username': 'duplicate',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        self.assertIn(b'Username is already taken', response.data)

if __name__ == '__main__':
    unittest.main()
