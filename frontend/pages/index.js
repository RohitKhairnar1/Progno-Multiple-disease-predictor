const loginBtn = document.querySelector('#login-btn');
  loginBtn.addEventListener('click', e => {
  e.preventDefault();

  const email = document.querySelector('#email').value;
  const password = document.querySelector('#password').value;

  auth.signInWithEmailAndPassword(email, password)
    .then(cred => {
      console.log('Logged in user!');
    })
    .catch(error => {
      console.log(error.message);
    })
});
const signupBtn = document.querySelector('#signup-btn');
    signupBtn.addEventListener('click', e => {
    e.preventDefault();

    const email = document.querySelector('#email').value;
    const password = document.querySelector('#password').value;

    auth.createUserWithEmailAndPassword(email, password).then(cred => {
    console.log(cred);
    console.log('User signed up!');
  });
});
auth.onAuthStateChanged(user => {
    if (user) {
      console.log(user.email + " is logged in!");
    } else {
      console.log('User is logged out!');
    }
  });