document.addEventListener('DOMContentLoaded', function () {
    // Get the patient data from localStorage
    const patients = JSON.parse(localStorage.getItem('patients')) || [];
    const tableBody = document.getElementById('patient-table-body');

    if (patients.length === 0) {
        console.log('No patients found in localStorage.');
        return;
    }

    patients.forEach((patient, index) => {
        // Create a new row for each patient
        const newRow = document.createElement('div');
        newRow.classList.add('table__row');
        newRow.id = `row${index + 1}`; // Unique ID based on index

        // Set inner HTML for the new row
        newRow.innerHTML = `
            <div class="table__cell">
                <div class="avatar-thumb">
                    <img src="images/avatar-placeholder.jpg" alt="Avatar" title="${patient.fname} ${patient.lname}"/>
                </div>
            </div>
            <div class="table__cell">${patient.fname} ${patient.lname}</div>
            <div class="table__cell">
                <img src="images/icons/icons-20-gray/${patient.gender.toLowerCase()}.png" alt="${patient.gender}" title="${patient.gender}" />
            </div>
            <div class="table__cell">${patient.dob}</div>
            <div class="table__cell">${patient.email}</div>
            <div class="table__cell">${patient.phone}</div>
            <div class="table__cell">${patient.city}, ${patient.address}</div>
            <div class="table__cell">
                <a href="#" class="button button--general button--inline button--blue-bg modal-toggle" data-openpopup="appointment">+ Appointment</a>
            </div>
            <div class="table__cell">
                <span class="show-more show-more--ellipsis has-dropdown"></span>
                <nav class="dropdown-menu dropdown-menu--content dropdown-menu--table">             
                    <ul>
                        <li><a href="dashboard__patient-profile.html">View Profile</a></li>
                        <li><a href="#">Contact</a></li>
                        <li><a href="#" class="delete-row">Delete</a></li>
                    </ul>
                </nav>
            </div>
        `;

        // Append the new row to the table body
        tableBody.appendChild(newRow);

        // Optional: Log success message for each row added
        console.log('Patient row added successfully:', `${patient.fname} ${patient.lname}`);
    });
});
