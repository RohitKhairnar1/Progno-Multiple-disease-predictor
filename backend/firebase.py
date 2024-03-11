
import firebase_admin

firebase_admin.initialize_app({
    'credential': firebase_admin.credential.certificate(
  {'apiKey': "AIzaSyCmRmbdb0J0ZmkrYJX9Q2VzWNoGdo77sho",
  'authDomain': "progno-app.firebaseapp.com",
  'projectId': "progno-app",
  'storageBucket': "progno-app.appspot.com",
  'messagingSenderId': "280979417527",
  'appId': "1:280979417527:web:ce04312654f05b52eea032"
  }
  ),
});


# app = firebase_admin.initializeApp(firebaseConfig);
# auth=firebase_admin.auth()