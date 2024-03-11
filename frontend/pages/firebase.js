// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
import firebase from "firebase/app";
import "firebase/auth";

// Your Firebase project configuration (replace with your actual credentials)
const firebaseConfig = {
  apiKey: "AIzaSyCmRmbdb0J0ZmkrYJX9Q2VzWNoGdo77sho",
  authDomain: "progno-app.firebaseapp.com",
  projectId: "progno-app",
  storageBucket: "progno-app.appspot.com",
  messagingSenderId: "280979417527",
  appId: "1:280979417527:web:ce04312654f05b52eea032"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Get a reference to the authentication service
const auth = firebase.auth();

// Login function using email and password (with error handling)
const login = async (email, password) => {
  try {
    const userCredential = await auth.signInWithEmailAndPassword(email, password);
    console.log("Logged in successfully:", userCredential.user);
    // Handle successful login (e.g., redirect to protected area)
  } catch (error) {
    console.error("Login failed:", error.message);
    // Handle login errors (e.g., display error message to user)
  }
};

// Logout function
const logout = async () => {
  try {
    await auth.signOut();
    console.log("Logged out successfully");
    // Handle successful logout (e.g., redirect to login page)
  } catch (error) {
    console.error("Logout failed:", error.message);
    // Handle logout errors
  }
};

export { login, logout };



// Initialize Firebase
// export const app = initializeApp(firebaseConfig);