document.addEventListener('DOMContentLoaded', function () {
    // Ensure the form element is loaded before referencing it
    const form = document.getElementById('patient-form');

    // Check if the form is present on the page
    if (form) {
        form.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const fname = document.getElementById('fname').value;
            const lname = document.getElementById('lname').value;
            const email = document.getElementById('email').value;
            const gender = document.getElementById('gender').value === '1' ? 'Female' : 'Male';
            const phone = document.getElementById('phone').value;
            const address = document.getElementById('address').value;
            const city = document.getElementById('city').value;
            const zip = document.getElementById('zip').value;
            const dob = document.getElementById('brdy').value;

            // Create an object to store patient data
            const patientData = {
                fname,
                lname,
                email,
                gender,
                phone,
                address,
                city,
                zip,
                dob
            };

            // Save data to localStorage
            const patients = JSON.parse(localStorage.getItem('patients')) || [];
            patients.push(patientData);
            localStorage.setItem('patients', JSON.stringify(patients));

            // Log the saved data
            console.log('Data saved to localStorage:', JSON.parse(localStorage.getItem('patients')));

            // Reset the form after submission
            form.reset();
        });
    } else {
        console.error('Form element not found.');
    }
});
