
const loginBtn = document.querySelector('#login-btn');
loginBtn.addEventListener('click', e => {
    e.preventDefault();

    const email = document.querySelector('#email').value;
    const password = document.querySelector('#password').value;

    auth.signInWithEmailAndPassword(email, password)
        .then(cred => {
            console.log('Logged in user!');
            window.location.href = "PAGE1.html";
        })
        .catch(error => {
            console.log(error.message);
        })
});

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
  
  export { loginBtn, logout,updateProfile };



auth.onAuthStateChanged(user => {
    if (user) {
        console.log(user.email + " is logged in!");
    } else {
        console.log('User is logged out!');
    }
});