document.addEventListener('DOMContentLoaded', function () {
    // Function to load the deleted row states from localStorage
    function loadDeletedRows() {
        const deletedRows = JSON.parse(localStorage.getItem('deletedRows')) || [];
        deletedRows.forEach(rowId => {
            const row = document.getElementById(rowId);
            if (row) {
                row.style.display = 'none'; // Hide the deleted row
            }
        });
    }

    // Add event listener to all delete links
    document.querySelectorAll('.delete-row').forEach(deleteLink => {
        deleteLink.addEventListener('click', function (event) {
            event.preventDefault(); // Prevent default link behavior
            const row = this.closest('.table__row'); // Find the closest row
            if (row) {
                const rowId = row.id; // Get the row ID

                // Store the deleted row ID in localStorage
                let deletedRows = JSON.parse(localStorage.getItem('deletedRows')) || [];
                deletedRows.push(rowId);
                localStorage.setItem('deletedRows', JSON.stringify(deletedRows));

                // Hide the row
                row.style.display = 'none';
            }
        });
    });

    // Load the deleted rows from localStorage
    loadDeletedRows();
});


