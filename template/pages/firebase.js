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
// JavaScript code for profile page
firebase.auth().onAuthStateChanged(function(user) {
  if (user) {
      // User is signed in.
      document.getElementById("displayName").value = user.displayName;
      document.getElementById("email").value = user.email;
      
      // Display profile picture if available
      if (user.photoURL) {
          var profileImage = document.getElementById("profileImage");
          profileImage.innerHTML = `<img src="${user.photoURL}" alt="Profile Picture" style="max-width: 100px;">`;
      }
  } else {
      // User is signed out.
      window.location.href = "login.html"; // Redirect to login page if not logged in
  }
});

// Update profile function
// Update profile function
function updateProfile() {
  var user = firebase.auth().currentUser;
  var displayName = document.getElementById("displayName").value;
  var password = document.getElementById("password").value;

  // Update display name
  users.updateProfile({
      displayName: displayName
  }).then(function() {
      console.log("Display name updated successfully");
  }).catch(function(error) {
      console.error("Error updating display name:", error);
  });

  // Update password if provided
  if (password) {
      user.updatePassword(password).then(function() {
          console.log("Password updated successfully");
      }).catch(function(error) {
          console.error("Error updating password:", error);
      });
  }

  // Handle profile picture upload
  var profilePicture = document.getElementById("profilePicture").files[0];
  if (profilePicture) {
      var storageRef = firebase.storage().ref().child('profilePictures/' + user.uid);
      storageRef.put(profilePicture).then(function(snapshot) {
          console.log("Profile picture uploaded successfully");
          // Get the uploaded profile picture URL and update user's photoURL
          snapshot.ref.getDownloadURL().then(function(downloadURL) {
              user.updateProfile({
                  photoURL: downloadURL
              }).then(function() {
                  console.log("Profile picture URL updated successfully");
                  // Save the download URL in Firestore or Realtime Database
                  firebase.firestore().collection('users').doc(user.uid).update({
                      profilePictureURL: downloadURL
                  }).then(function() {
                      console.log("Profile picture URL saved to Firestore");
                      // Optionally, display the updated profile picture
                      var profileImage = document.getElementById("profileImage");
                      profileImage.innerHTML = `<img src="${downloadURL}" alt="Profile Picture" style="max-width: 100px;">`;
                  }).catch(function(error) {
                      console.error("Error saving profile picture URL to Firestore:", error);
                  });
              }).catch(function(error) {
                  console.error("Error updating profile picture URL:", error);
              });
          }).catch(function(error) {
              console.error("Error getting profile picture URL:", error);
          });
      }).catch(function(error) {
          console.error("Error uploading profile picture:", error);
      });
  }
}


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

export { login, logout,updateProfile };



// Initialize Firebase
// export const app = initializeApp(firebaseConfig);